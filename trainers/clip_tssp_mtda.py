import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from clip import clip
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param

from models.clip_tssp import CLIPVisualWithHidden, MultiLayerStyleProjector
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import TextEncoder, load_clip_to_cpu
from trainers.mtda_base import MultiTargetTrainerXU


class CustomCLIPTSSPMTDA(nn.Module):
    """CLIP with target-set style tokens in the text prompt.

    The CLIP encoders are frozen. Training only updates the multi-layer style
    projector that maps visual hidden-state statistics into text prompt tokens.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        tssp_cfg = cfg.TRAINER.CLIP_TSSP_MTDA

        self.cfg = cfg
        self.dtype = clip_model.dtype
        self.target_domains = list(cfg.DATASET.TARGET_DOMAINS)
        self.use_gap_token = bool(tssp_cfg.USE_GAP_TOKEN)
        self.style_group_size = int(tssp_cfg.STYLE_GROUP_SIZE)
        self.gap_group_size = int(tssp_cfg.GAP_GROUP_SIZE)
        self.gap_position = str(tssp_cfg.GAP_POSITION).lower()
        self.proto_momentum = float(tssp_cfg.PROTO_MOMENTUM)
        self.debug_print_once = bool(tssp_cfg.DEBUG.PRINT_ONCE)
        self._has_printed_debug = False
        self._has_warned_empty_proto = False

        self.image_encoder = CLIPVisualWithHidden(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = self.image_encoder.width
        depth = self.image_encoder.depth
        self.style_projector = MultiLayerStyleProjector(
            visual_dim=visual_dim,
            text_dim=text_dim,
            depth=depth,
            hidden_dim=int(tssp_cfg.HIDDEN_DIM),
            eps=float(tssp_cfg.STYLE_EPS),
        )

        self.depth = depth
        self.style_depth = self._compressed_depth(
            depth, self.style_group_size, "STYLE_GROUP_SIZE"
        )
        self.gap_depth = self._compressed_depth(
            depth, self.gap_group_size, "GAP_GROUP_SIZE"
        )
        self.text_dim = text_dim
        self.n_ctx = self.style_depth * 2
        if self.use_gap_token:
            self.n_ctx += self.gap_depth

        self.register_buffer(
            "source_style_proto", torch.zeros(self.style_depth, text_dim)
        )
        self.register_buffer(
            "target_style_proto", torch.zeros(self.style_depth, text_dim)
        )
        self.register_buffer(
            "source_gap_proto", torch.zeros(self.gap_depth, text_dim)
        )
        self.register_buffer(
            "target_gap_proto", torch.zeros(self.gap_depth, text_dim)
        )
        self.register_buffer("proto_initialized", torch.tensor(False))

        template = str(tssp_cfg.PROMPT_TEMPLATE)
        if "{}" not in template:
            raise ValueError("TRAINER.CLIP_TSSP_MTDA.PROMPT_TEMPLATE must contain '{}'")

        prompt_prefix = " ".join(["X"] * self.n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + template.format(name) for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])
        self.n_cls = len(classnames)

        print("CLIPTSSPMTDA prompt setup:")
        print(f"  prompt template: {template}")
        print(f"  visual layers: {depth}")
        print(f"  style group size: {self.style_group_size}")
        print(f"  style token layers: {self.style_depth}")
        print(f"  gap group size: {self.gap_group_size}")
        print(f"  gap token layers: {self.gap_depth}")
        print(f"  use gap token: {self.use_gap_token}")
        print(f"  gap position: {self.gap_position}")
        print(f"  context tokens: {self.n_ctx}")

    @staticmethod
    def _compressed_depth(depth, group_size, config_name):
        if group_size <= 0:
            raise ValueError(f"{config_name} must be positive, got {group_size}")
        if depth % group_size != 0:
            raise ValueError(
                f"{config_name}={group_size} must divide visual depth {depth}"
            )
        return depth // group_size

    @staticmethod
    def _ensure_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values detected in {name}")

    @torch.no_grad()
    def _encode_visual(self, image):
        image_features, hidden_states = self.image_encoder(image)
        return image_features.float(), hidden_states

    def _compute_style_tokens(self, hidden_states):
        style_tokens = self.style_projector(hidden_states).float()
        self._ensure_finite("style_tokens", style_tokens)
        return style_tokens

    @staticmethod
    def _compress_tokens(tokens, group_size):
        if group_size == 1:
            return tokens
        batch_size, depth, text_dim = tokens.shape
        if depth % group_size != 0:
            raise ValueError(
                f"group_size={group_size} must divide token depth {depth}"
            )
        return tokens.reshape(
            batch_size, depth // group_size, group_size, text_dim
        ).mean(dim=2)

    def _assemble_context(self, source_tokens, target_tokens, gap_tokens=None):
        if gap_tokens is None:
            return torch.cat([source_tokens, target_tokens], dim=1)
        if self.gap_position == "middle":
            return torch.cat([source_tokens, gap_tokens, target_tokens], dim=1)
        if self.gap_position == "after_target":
            return torch.cat([source_tokens, target_tokens, gap_tokens], dim=1)
        raise ValueError(f"Unsupported GAP_POSITION={self.gap_position}")

    @torch.no_grad()
    def _update_prototypes(
        self,
        source_style_proto,
        target_style_proto,
        source_gap_proto,
        target_gap_proto,
    ):
        momentum = self.proto_momentum
        if not bool(self.proto_initialized.item()):
            self.source_style_proto.copy_(source_style_proto.detach())
            self.target_style_proto.copy_(target_style_proto.detach())
            self.source_gap_proto.copy_(source_gap_proto.detach())
            self.target_gap_proto.copy_(target_gap_proto.detach())
            self.proto_initialized.fill_(True)
            return

        self.source_style_proto.mul_(momentum).add_(
            source_style_proto.detach(), alpha=1.0 - momentum
        )
        self.target_style_proto.mul_(momentum).add_(
            target_style_proto.detach(), alpha=1.0 - momentum
        )
        self.source_gap_proto.mul_(momentum).add_(
            source_gap_proto.detach(), alpha=1.0 - momentum
        )
        self.target_gap_proto.mul_(momentum).add_(
            target_gap_proto.detach(), alpha=1.0 - momentum
        )

    def _build_train_context(self, source_raw_tokens, target_raw_tokens_by_domain):
        source_style_tokens = self._compress_tokens(
            source_raw_tokens, self.style_group_size
        )
        target_style_tokens_by_domain = OrderedDict(
            (
                domain_name,
                self._compress_tokens(tokens, self.style_group_size),
            )
            for domain_name, tokens in target_raw_tokens_by_domain.items()
        )
        source_style_proto = source_style_tokens.mean(dim=0)
        target_style_domain_protos = OrderedDict(
            (domain_name, tokens.mean(dim=0))
            for domain_name, tokens in target_style_tokens_by_domain.items()
        )
        target_style_proto = torch.stack(
            list(target_style_domain_protos.values())
        ).mean(dim=0)

        source_gap_tokens = self._compress_tokens(
            source_raw_tokens, self.gap_group_size
        )
        target_gap_tokens_by_domain = OrderedDict(
            (
                domain_name,
                self._compress_tokens(tokens, self.gap_group_size),
            )
            for domain_name, tokens in target_raw_tokens_by_domain.items()
        )
        source_gap_proto = source_gap_tokens.mean(dim=0)
        target_gap_domain_protos = OrderedDict(
            (domain_name, tokens.mean(dim=0))
            for domain_name, tokens in target_gap_tokens_by_domain.items()
        )
        target_gap_proto = torch.stack(
            list(target_gap_domain_protos.values())
        ).mean(dim=0)
        self._update_prototypes(
            source_style_proto=source_style_proto,
            target_style_proto=target_style_proto,
            source_gap_proto=source_gap_proto,
            target_gap_proto=target_gap_proto,
        )

        batch_size = source_raw_tokens.shape[0]
        target_style_tokens = target_style_proto.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        gap_tokens = None
        if self.use_gap_token:
            gap_proto = target_gap_proto - source_gap_proto
            gap_tokens = gap_proto.unsqueeze(0).expand(batch_size, -1, -1)

        context = self._assemble_context(
            source_tokens=source_style_tokens,
            target_tokens=target_style_tokens,
            gap_tokens=gap_tokens,
        )
        self._ensure_finite("context_train", context)
        return context, {
            "source_style_tokens": source_style_tokens,
            "target_style_tokens_by_domain": target_style_tokens_by_domain,
            "source_style_proto": source_style_proto,
            "target_style_proto": target_style_proto,
            "target_style_domain_protos": target_style_domain_protos,
            "source_gap_proto": source_gap_proto,
            "target_gap_proto": target_gap_proto,
            "target_gap_domain_protos": target_gap_domain_protos,
            "gap_tokens": gap_tokens,
        }

    def _build_eval_context(self, batch_size, device):
        if not bool(self.proto_initialized.item()):
            if not self._has_warned_empty_proto:
                print(
                    "Warning: CLIPTSSPMTDA style prototypes are not initialized; "
                    "using zero style tokens for this eval call."
                )
                self._has_warned_empty_proto = True
            source_proto = torch.zeros(
                self.style_depth, self.text_dim, device=device, dtype=torch.float32
            )
            target_proto = torch.zeros_like(source_proto)
            source_gap_proto = torch.zeros(
                self.gap_depth, self.text_dim, device=device, dtype=torch.float32
            )
            target_gap_proto = torch.zeros_like(source_gap_proto)
        else:
            source_proto = self.source_style_proto.to(device=device, dtype=torch.float32)
            target_proto = self.target_style_proto.to(device=device, dtype=torch.float32)
            source_gap_proto = self.source_gap_proto.to(
                device=device, dtype=torch.float32
            )
            target_gap_proto = self.target_gap_proto.to(
                device=device, dtype=torch.float32
            )

        source_tokens = source_proto.unsqueeze(0).expand(batch_size, -1, -1)
        target_tokens = target_proto.unsqueeze(0).expand(batch_size, -1, -1)
        gap_tokens = None
        if self.use_gap_token:
            gap_tokens = (target_gap_proto - source_gap_proto).unsqueeze(0).expand(
                batch_size, -1, -1
            )

        context = self._assemble_context(
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            gap_tokens=gap_tokens,
        )
        self._ensure_finite("context_eval", context)
        return context

    def _construct_prompts(self, ctx_i):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx_i = ctx_i.to(device=prefix.device, dtype=prefix.dtype)
        ctx_i = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([prefix, ctx_i, suffix], dim=1)

    def _compute_logits(self, image_features, context):
        image_features = image_features.float()
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        logit_scale = self.logit_scale.float().exp()
        tokenized_prompts = self.tokenized_prompts.to(image_features.device)

        logits = []
        for image_feature_i, ctx_i in zip(image_features, context):
            prompts_i = self._construct_prompts(ctx_i)
            text_features = self.text_encoder(prompts_i, tokenized_prompts).float()
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            logits_i = logit_scale * image_feature_i @ text_features.t()
            logits.append(logits_i)

        logits = torch.stack(logits)
        self._ensure_finite("logits", logits)
        return logits

    def _log_debug_once(
        self,
        image_s,
        image_u_dict,
        hidden_s,
        source_raw_tokens,
        target_raw_tokens_by_domain,
        context,
        stats,
        logits,
        loss,
    ):
        if not self.debug_print_once or self._has_printed_debug:
            return

        print("[CLIPTSSPMTDA debug]")
        print("source domain:", self.cfg.DATASET.SOURCE_DOMAINS[0])
        print("target domains:", list(image_u_dict.keys()))
        print("source batch shape:", tuple(image_s.shape))
        for domain_name, image_u in image_u_dict.items():
            print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
        print("hidden layers:", len(hidden_s))
        print("first hidden shape:", tuple(hidden_s[0].shape))
        print("source raw style tokens shape:", tuple(source_raw_tokens.shape))
        for domain_name, tokens in target_raw_tokens_by_domain.items():
            print(
                f"target raw style tokens shape [{domain_name}]:",
                tuple(tokens.shape),
            )
        print(
            "source compressed style tokens shape:",
            tuple(stats["source_style_tokens"].shape),
        )
        print(
            "source style proto shape:",
            tuple(stats["source_style_proto"].shape),
        )
        print(
            "target-set style proto shape:",
            tuple(stats["target_style_proto"].shape),
        )
        print("source gap proto shape:", tuple(stats["source_gap_proto"].shape))
        print("target-set gap proto shape:", tuple(stats["target_gap_proto"].shape))
        if stats["gap_tokens"] is not None:
            print("gap style tokens shape:", tuple(stats["gap_tokens"].shape))
        print("context shape:", tuple(context.shape))
        print("style_group_size:", self.style_group_size)
        print("gap_group_size:", self.gap_group_size)
        print("use_gap_token:", self.use_gap_token)
        print("gap_position:", self.gap_position)
        print("logits shape:", tuple(logits.shape))
        print("loss:", float(loss.detach().item()))
        print(
            "source token norm:",
            float(stats["source_style_tokens"].detach().norm().item()),
        )
        print(
            "target proto norm:",
            float(stats["target_style_proto"].detach().norm().item()),
        )
        if stats["gap_tokens"] is not None:
            print("gap token norm:", float(stats["gap_tokens"].detach().norm().item()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_u_dict):
        image_features_s, hidden_s = self._encode_visual(image_s)
        source_raw_tokens = self._compute_style_tokens(hidden_s)

        target_raw_tokens_by_domain = OrderedDict()
        for domain_name, image_u in image_u_dict.items():
            _, hidden_u = self._encode_visual(image_u)
            target_raw_tokens_by_domain[domain_name] = self._compute_style_tokens(
                hidden_u
            )

        context, stats = self._build_train_context(
            source_raw_tokens, target_raw_tokens_by_domain
        )
        logits = self._compute_logits(image_features_s, context)
        loss_ce = F.cross_entropy(logits, label_s)
        self._ensure_finite("loss_ce", loss_ce)

        self._log_debug_once(
            image_s=image_s,
            image_u_dict=image_u_dict,
            hidden_s=hidden_s,
            source_raw_tokens=source_raw_tokens,
            target_raw_tokens_by_domain=target_raw_tokens_by_domain,
            context=context,
            stats=stats,
            logits=logits,
            loss=loss_ce,
        )

        acc = compute_accuracy(logits, label_s)[0].item()
        source_norm = stats["source_style_tokens"].detach().float().norm()
        target_norm = stats["target_style_proto"].detach().float().norm()
        gap_norm = torch.zeros((), device=loss_ce.device)
        if stats["gap_tokens"] is not None:
            gap_norm = stats["gap_tokens"].detach().float().norm()

        return {
            "loss": loss_ce,
            "loss_ce": loss_ce.detach(),
            "acc_src": torch.tensor(acc, device=loss_ce.device),
            "source_style_norm": source_norm,
            "target_style_norm": target_norm,
            "gap_style_norm": gap_norm,
        }

    def forward_inference(self, image, domain_name=None):
        del domain_name
        image_features, _ = self._encode_visual(image)
        context = self._build_eval_context(
            batch_size=image_features.shape[0],
            device=image_features.device,
        )
        return self._compute_logits(image_features, context)

    def forward(self, image, domain_name=None):
        return self.forward_inference(image, domain_name=domain_name)


@TRAINER_REGISTRY.register()
class CLIPTSSPMTDA(MultiTargetTrainerXU):
    def check_cfg(self, cfg):
        tssp_cfg = cfg.TRAINER.CLIP_TSSP_MTDA
        assert tssp_cfg.PREC in ["fp16", "fp32", "amp"]
        assert tssp_cfg.HIDDEN_DIM > 0
        assert tssp_cfg.STYLE_GROUP_SIZE in [1, 2, 3, 4]
        assert tssp_cfg.GAP_GROUP_SIZE in [1, 2, 3, 4]
        assert tssp_cfg.GAP_POSITION in ["after_target", "middle"]
        assert 0.0 <= tssp_cfg.PROTO_MOMENTUM < 1.0
        assert tssp_cfg.STYLE_EPS > 0.0
        assert "{}" in tssp_cfg.PROMPT_TEMPLATE

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.CLIP_TSSP_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CLIPTSSPMTDA")
        self.model = CustomCLIPTSSPMTDA(cfg, classnames, clip_model)

        print("Freezing CLIP; updating only the multi-layer style projector")
        for name, param in self.model.named_parameters():
            param.requires_grad_(name.startswith("style_projector."))

        enabled = [
            name for name, param in self.model.named_parameters() if param.requires_grad
        ]
        print("Parameters to be updated:")
        for name in enabled:
            print(f"  - {name}")

        self.model.to(self.device)
        total_params = count_num_param(self.model)
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        print(f"# params: {total_params:,}")
        print(f"# trainable params: {trainable_params:,}")

        self.optim = build_optimizer(self.model.style_projector, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("clip_tssp_mtda", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP_TSSP_MTDA.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _model_ref(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)
        model = self._model_ref()
        prec = self.cfg.TRAINER.CLIP_TSSP_MTDA.PREC

        if prec == "amp":
            with autocast():
                outputs = model.forward_train(image_x, label_x, image_u)
                loss = outputs["loss"]
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            outputs = model.forward_train(image_x, label_x, image_u)
            loss = outputs["loss"]
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {"loss": float(loss.item())}
        for key, value in outputs.items():
            if key == "loss":
                continue
            loss_summary[key] = value.item() if torch.is_tensor(value) else float(value)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input_tensor, domain_name=None):
        return self._model_ref().forward_inference(input_tensor, domain_name=domain_name)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint_compat(model_path)
            state_dict = checkpoint["state_dict"]
            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)
            state_dict.pop("tokenized_prompts", None)
            model_ref = self._models[name]
            if (
                "source_gap_proto" not in state_dict
                and "source_style_proto" in state_dict
                and state_dict["source_style_proto"].shape
                == model_ref.source_gap_proto.shape
            ):
                state_dict["source_gap_proto"] = state_dict[
                    "source_style_proto"
                ].clone()
            if (
                "target_gap_proto" not in state_dict
                and "target_style_proto" in state_dict
                and state_dict["target_style_proto"].shape
                == model_ref.target_gap_proto.shape
            ):
                state_dict["target_gap_proto"] = state_dict[
                    "target_style_proto"
                ].clone()

            loaded_epoch = checkpoint["epoch"]
            print(f'Loading weights to {name} from "{model_path}" (epoch = {loaded_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
