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

from models.clip_tssp import (
    CLIPVisualWithHidden,
    MultiLayerImageProjector,
    MultiLayerStyleProjector,
)
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
        self.use_image_tokens = bool(tssp_cfg.USE_IMAGE_TOKENS)
        self.image_group_size = int(tssp_cfg.IMAGE_GROUP_SIZE)
        self.enable_vpt = bool(tssp_cfg.ENABLE_VPT)
        self.n_vctx = int(tssp_cfg.N_VCTX)
        self.gap_position = str(tssp_cfg.GAP_POSITION).lower()
        self.proto_momentum = float(tssp_cfg.PROTO_MOMENTUM)
        self.lambda_em = float(tssp_cfg.LAMBDA_EM)
        self.detach_entropy_text = bool(tssp_cfg.DETACH_ENTROPY_TEXT)
        self.entropy_eps = float(tssp_cfg.ENTROPY_EPS)
        self.lambda_kl = float(tssp_cfg.LAMBDA_KL)
        self.kl_temperature = float(tssp_cfg.KL_TEMPERATURE)
        self.lambda_pl = float(tssp_cfg.LAMBDA_PL)
        self.pl_threshold = float(tssp_cfg.PL_THRESHOLD)
        self.pl_student_threshold = float(tssp_cfg.PL_STUDENT_THRESHOLD)
        self.pl_use_student_low_conf_mask = bool(
            tssp_cfg.PL_USE_STUDENT_LOW_CONF_MASK
        )
        self.debug_print_once = bool(tssp_cfg.DEBUG.PRINT_ONCE)
        self._has_printed_debug = False
        self._has_warned_empty_proto = False

        self.image_encoder = CLIPVisualWithHidden(
            clip_model.visual,
            enable_vpt=self.enable_vpt,
            n_vctx=self.n_vctx,
            vctx_init_std=float(tssp_cfg.VCTX_INIT_STD),
        )
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
        if self.use_image_tokens:
            self.image_projector = MultiLayerImageProjector(
                visual_dim=visual_dim,
                text_dim=text_dim,
                depth=depth,
            )
        else:
            self.image_projector = None

        self.depth = depth
        self.style_depth = self._compressed_depth(
            depth, self.style_group_size, "STYLE_GROUP_SIZE"
        )
        self.gap_depth = self._compressed_depth(
            depth, self.gap_group_size, "GAP_GROUP_SIZE"
        )
        self.image_depth = self._compressed_depth(
            depth, self.image_group_size, "IMAGE_GROUP_SIZE"
        )
        self.text_dim = text_dim
        self.n_ctx = self.style_depth * 2
        if self.use_gap_token:
            self.n_ctx += self.gap_depth
        if self.use_image_tokens:
            self.n_ctx += self.image_depth

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
        zs_template = str(tssp_cfg.ZS_PROMPT_TEMPLATE)
        if "{}" not in zs_template:
            raise ValueError(
                "TRAINER.CLIP_TSSP_MTDA.ZS_PROMPT_TEMPLATE must contain '{}'"
            )

        prompt_prefix = " ".join(["X"] * self.n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + template.format(name) for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        zs_prompts = [zs_template.format(name) for name in classnames]
        zs_tokenized_prompts = torch.cat(
            [clip.tokenize(prompt) for prompt in zs_prompts]
        )

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("zs_tokenized_prompts", zs_tokenized_prompts)
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
        print(f"  use image tokens: {self.use_image_tokens}")
        print(f"  image group size: {self.image_group_size}")
        print(f"  image token layers: {self.image_depth}")
        print(f"  persistent VCTX: {self.enable_vpt}")
        print(f"  VCTX tokens: {self.n_vctx if self.enable_vpt else 0}")
        print(f"  target entropy weight: {self.lambda_em}")
        print(f"  detach entropy text: {self.detach_entropy_text}")
        print(f"  KL weight: {self.lambda_kl}")
        print(f"  KL temperature: {self.kl_temperature}")
        print(f"  pseudo-label weight: {self.lambda_pl}")
        print(f"  pseudo-label threshold: {self.pl_threshold}")
        print(f"  pseudo-label student threshold: {self.pl_student_threshold}")
        print(f"  pseudo-label low-conf only: {self.pl_use_student_low_conf_mask}")
        print(f"  zero-shot prompt template: {zs_template}")
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

    def _encode_clean_visual(self, image):
        with torch.no_grad():
            image_features, hidden_states = self.image_encoder(image)
        return image_features.float(), hidden_states

    def _encode_adapted_visual(self, image, clean_features=None):
        if not self.enable_vpt:
            if clean_features is None:
                clean_features, _ = self._encode_clean_visual(image)
            return clean_features
        return self.image_encoder.forward_vpt(image).float()

    def _compute_style_tokens(self, hidden_states):
        style_tokens = self.style_projector(hidden_states).float()
        self._ensure_finite("style_tokens", style_tokens)
        return style_tokens

    def _compute_image_tokens(self, hidden_states):
        if self.image_projector is None:
            return None
        image_tokens = self.image_projector(hidden_states).float()
        image_tokens = self._compress_tokens(
            image_tokens, self.image_group_size
        )
        self._ensure_finite("image_tokens", image_tokens)
        return image_tokens

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

    def _assemble_context(
        self,
        source_tokens,
        target_tokens,
        gap_tokens=None,
        image_tokens=None,
    ):
        if gap_tokens is None:
            pieces = [source_tokens, target_tokens]
        elif self.gap_position == "middle":
            pieces = [source_tokens, gap_tokens, target_tokens]
        elif self.gap_position == "after_target":
            pieces = [source_tokens, target_tokens, gap_tokens]
        else:
            raise ValueError(f"Unsupported GAP_POSITION={self.gap_position}")
        if image_tokens is not None:
            pieces.append(image_tokens)
        return torch.cat(pieces, dim=1)

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

    def _build_train_context(
        self,
        source_raw_tokens,
        target_raw_tokens_by_domain,
        source_image_tokens=None,
    ):
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
            image_tokens=source_image_tokens,
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
            "image_tokens": source_image_tokens,
        }

    def _build_target_contexts(
        self,
        stats,
        target_batch_sizes,
        target_image_tokens_by_domain,
    ):
        contexts = OrderedDict()
        for domain_name, batch_size in target_batch_sizes.items():
            source_tokens = stats["source_style_proto"].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            target_tokens = stats["target_style_domain_protos"][
                domain_name
            ].unsqueeze(0).expand(batch_size, -1, -1)

            gap_tokens = None
            if self.use_gap_token:
                gap_proto = (
                    stats["target_gap_domain_protos"][domain_name]
                    - stats["source_gap_proto"]
                )
                gap_tokens = gap_proto.unsqueeze(0).expand(batch_size, -1, -1)

            context = self._assemble_context(
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                gap_tokens=gap_tokens,
                image_tokens=target_image_tokens_by_domain.get(domain_name),
            )
            self._ensure_finite(f"context_target[{domain_name}]", context)
            contexts[domain_name] = context
        return contexts

    def _build_eval_context(self, batch_size, device, image_tokens=None):
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
            image_tokens=image_tokens,
        )
        self._ensure_finite("context_eval", context)
        return context

    def _construct_prompts(self, ctx_i):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx_i = ctx_i.to(device=prefix.device, dtype=prefix.dtype)
        ctx_i = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([prefix, ctx_i, suffix], dim=1)

    def _compute_logits(self, image_features, context, detach_text_features=False):
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
            if detach_text_features:
                text_features = text_features.detach()
            logits_i = logit_scale * image_feature_i @ text_features.t()
            logits.append(logits_i)

        logits = torch.stack(logits)
        self._ensure_finite("logits", logits)
        return logits

    def _uses_target_logits(self):
        return (
            self.lambda_em > 0.0
            or self.lambda_kl > 0.0
            or self.lambda_pl > 0.0
        )

    def _zero_shot_text_features(self, device):
        tokenized_prompts = self.zs_tokenized_prompts.to(device)
        prompts = self.token_embedding(tokenized_prompts).type(self.dtype)
        text_features = self.text_encoder(prompts, tokenized_prompts).float()
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        self._ensure_finite("zero_shot_text_features", text_features)
        return text_features

    @torch.no_grad()
    def _compute_reference_logits(self, image_features):
        image_features = image_features.float()
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        text_features = self._zero_shot_text_features(image_features.device)
        logits = self.logit_scale.float().exp() * image_features @ text_features.t()
        self._ensure_finite("reference_logits", logits)
        return logits.detach()

    def _conditional_entropy(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = probs.clamp_min(self.entropy_eps).log()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        self._ensure_finite("conditional_entropy", entropy)
        return entropy

    def _reference_kl_loss(self, student_logits, reference_logits):
        temperature = self.kl_temperature
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
        student_probs = student_log_probs.exp()
        reference_log_probs = F.log_softmax(
            reference_logits.float() / temperature, dim=-1
        ).detach()
        kl = (
            student_probs * (student_log_probs - reference_log_probs)
        ).sum(dim=-1).mean()
        kl = kl * (temperature ** 2)
        self._ensure_finite("reference_kl_loss", kl)
        return kl

    def _pseudo_label_loss(self, student_logits, reference_logits):
        reference_probs = F.softmax(reference_logits.float(), dim=-1)
        reference_conf, reference_label = reference_probs.max(dim=-1)
        student_probs = F.softmax(student_logits.detach().float(), dim=-1)
        student_conf, student_label = student_probs.max(dim=-1)

        mask = reference_conf.ge(self.pl_threshold)
        if self.pl_use_student_low_conf_mask:
            mask = mask & student_conf.lt(self.pl_student_threshold)

        mask_float = mask.float()
        ce = F.cross_entropy(student_logits, reference_label, reduction="none")
        loss = (ce * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        self._ensure_finite("pseudo_label_loss", loss)
        stats = {
            "coverage": mask_float.mean(),
            "clip_conf": reference_conf.mean(),
            "student_conf": student_conf.mean(),
            "agreement": (student_label == reference_label).float().mean(),
        }
        return loss, stats

    def _log_debug_once(
        self,
        image_s,
        image_u_dict,
        hidden_s,
        source_raw_tokens,
        target_raw_tokens_by_domain,
        source_image_tokens,
        context,
        stats,
        logits,
        target_logits_by_domain,
        reference_logits_by_domain,
        loss_ce,
        loss_em,
        loss_kl,
        loss_pl,
        pl_coverage,
        loss_total,
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
        if source_image_tokens is not None:
            print("source image tokens shape:", tuple(source_image_tokens.shape))
        print("context shape:", tuple(context.shape))
        print("style_group_size:", self.style_group_size)
        print("gap_group_size:", self.gap_group_size)
        print("use_gap_token:", self.use_gap_token)
        print("gap_position:", self.gap_position)
        print("use_image_tokens:", self.use_image_tokens)
        print("image_group_size:", self.image_group_size)
        print("enable_vpt:", self.enable_vpt)
        if self.enable_vpt:
            print("vctx shape:", tuple(self.image_encoder.vctx.shape))
            print(
                "vctx norm:",
                float(self.image_encoder.vctx.detach().float().norm().item()),
            )
        print("lambda_em:", self.lambda_em)
        print("detach_entropy_text:", self.detach_entropy_text)
        print("lambda_kl:", self.lambda_kl)
        print("kl_temperature:", self.kl_temperature)
        print("lambda_pl:", self.lambda_pl)
        print("pl_threshold:", self.pl_threshold)
        print("pl_student_threshold:", self.pl_student_threshold)
        print("pl_low_conf_only:", self.pl_use_student_low_conf_mask)
        print("logits shape:", tuple(logits.shape))
        for domain_name, target_logits in target_logits_by_domain.items():
            print(
                f"target logits shape [{domain_name}]:",
                tuple(target_logits.shape),
            )
        for domain_name, reference_logits in reference_logits_by_domain.items():
            print(
                f"reference logits shape [{domain_name}]:",
                tuple(reference_logits.shape),
            )
        print("loss_ce:", float(loss_ce.detach().item()))
        print("loss_em:", float(loss_em.detach().item()))
        print("loss_kl:", float(loss_kl.detach().item()))
        print("loss_pl:", float(loss_pl.detach().item()))
        print("pl_coverage:", float(pl_coverage.detach().item()))
        print("loss_total:", float(loss_total.detach().item()))
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
        if source_image_tokens is not None:
            print(
                "image token norm:",
                float(source_image_tokens.detach().norm().item()),
            )
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_u_dict):
        need_target_logits = self._uses_target_logits()
        clean_features_s, hidden_s = self._encode_clean_visual(image_s)
        image_features_s = self._encode_adapted_visual(
            image_s, clean_features=clean_features_s
        )
        source_raw_tokens = self._compute_style_tokens(hidden_s)
        source_image_tokens = self._compute_image_tokens(hidden_s)

        target_raw_tokens_by_domain = OrderedDict()
        target_features_by_domain = OrderedDict()
        target_image_tokens_by_domain = OrderedDict()
        reference_logits_by_domain = OrderedDict()
        for domain_name, image_u in image_u_dict.items():
            clean_features_u, hidden_u = self._encode_clean_visual(image_u)
            target_raw_tokens_by_domain[domain_name] = self._compute_style_tokens(
                hidden_u
            )
            if need_target_logits:
                target_features_by_domain[domain_name] = self._encode_adapted_visual(
                    image_u, clean_features=clean_features_u
                )
                target_image_tokens_by_domain[domain_name] = (
                    self._compute_image_tokens(hidden_u)
                )
                reference_logits_by_domain[domain_name] = (
                    self._compute_reference_logits(clean_features_u)
                )

        context, stats = self._build_train_context(
            source_raw_tokens,
            target_raw_tokens_by_domain,
            source_image_tokens=source_image_tokens,
        )
        logits = self._compute_logits(image_features_s, context)
        loss_ce = F.cross_entropy(logits, label_s)
        self._ensure_finite("loss_ce", loss_ce)

        target_logits_by_domain = OrderedDict()
        entropy_by_domain = OrderedDict()
        if need_target_logits:
            target_contexts = self._build_target_contexts(
                stats=stats,
                target_batch_sizes=OrderedDict(
                    (domain_name, features.shape[0])
                    for domain_name, features in target_features_by_domain.items()
                ),
                target_image_tokens_by_domain=target_image_tokens_by_domain,
            )
            for domain_name, image_features_u in target_features_by_domain.items():
                target_logits = self._compute_logits(
                    image_features_u,
                    target_contexts[domain_name],
                    detach_text_features=(
                        self.lambda_em > 0.0 and self.detach_entropy_text
                    ),
                )
                target_logits_by_domain[domain_name] = target_logits
                if self.lambda_em > 0.0:
                    entropy_by_domain[domain_name] = self._conditional_entropy(
                        target_logits
                    )
        if self.lambda_em > 0.0:
            loss_em = torch.stack(list(entropy_by_domain.values())).mean()
        else:
            loss_em = loss_ce.new_zeros(())

        kl_by_domain = OrderedDict()
        if self.lambda_kl > 0.0:
            for domain_name, target_logits in target_logits_by_domain.items():
                kl_by_domain[domain_name] = self._reference_kl_loss(
                    target_logits,
                    reference_logits_by_domain[domain_name],
                )
            loss_kl = torch.stack(list(kl_by_domain.values())).mean()
        else:
            loss_kl = loss_ce.new_zeros(())

        pl_by_domain = OrderedDict()
        pl_stats_by_domain = OrderedDict()
        if self.lambda_pl > 0.0:
            for domain_name, target_logits in target_logits_by_domain.items():
                pl_loss, pl_stats = self._pseudo_label_loss(
                    target_logits,
                    reference_logits_by_domain[domain_name],
                )
                pl_by_domain[domain_name] = pl_loss
                pl_stats_by_domain[domain_name] = pl_stats
            loss_pl = torch.stack(list(pl_by_domain.values())).mean()
            pl_coverage = torch.stack(
                [stats_i["coverage"] for stats_i in pl_stats_by_domain.values()]
            ).mean()
            pl_clip_conf = torch.stack(
                [stats_i["clip_conf"] for stats_i in pl_stats_by_domain.values()]
            ).mean()
            pl_student_conf = torch.stack(
                [stats_i["student_conf"] for stats_i in pl_stats_by_domain.values()]
            ).mean()
            clip_student_agreement = torch.stack(
                [stats_i["agreement"] for stats_i in pl_stats_by_domain.values()]
            ).mean()
        else:
            loss_pl = loss_ce.new_zeros(())
            pl_coverage = loss_ce.new_zeros(())
            pl_clip_conf = loss_ce.new_zeros(())
            pl_student_conf = loss_ce.new_zeros(())
            clip_student_agreement = loss_ce.new_zeros(())

        loss_total = (
            loss_ce
            + self.lambda_em * loss_em
            + self.lambda_kl * loss_kl
            + self.lambda_pl * loss_pl
        )
        self._ensure_finite("loss_total", loss_total)

        self._log_debug_once(
            image_s=image_s,
            image_u_dict=image_u_dict,
            hidden_s=hidden_s,
            source_raw_tokens=source_raw_tokens,
            target_raw_tokens_by_domain=target_raw_tokens_by_domain,
            source_image_tokens=source_image_tokens,
            context=context,
            stats=stats,
            logits=logits,
            target_logits_by_domain=target_logits_by_domain,
            reference_logits_by_domain=reference_logits_by_domain,
            loss_ce=loss_ce,
            loss_em=loss_em,
            loss_kl=loss_kl,
            loss_pl=loss_pl,
            pl_coverage=pl_coverage,
            loss_total=loss_total,
        )

        acc = compute_accuracy(logits, label_s)[0].item()
        source_norm = stats["source_style_tokens"].detach().float().norm()
        target_norm = stats["target_style_proto"].detach().float().norm()
        gap_norm = torch.zeros((), device=loss_ce.device)
        if stats["gap_tokens"] is not None:
            gap_norm = stats["gap_tokens"].detach().float().norm()
        image_norm = torch.zeros((), device=loss_ce.device)
        if source_image_tokens is not None:
            image_norm = source_image_tokens.detach().float().norm()
        vctx_norm = torch.zeros((), device=loss_ce.device)
        if self.enable_vpt:
            vctx_norm = self.image_encoder.vctx.detach().float().norm()

        outputs = {
            "loss": loss_total,
            "loss_ce": loss_ce.detach(),
            "loss_em": loss_em.detach(),
            "weighted_loss_em": (self.lambda_em * loss_em).detach(),
            "loss_kl": loss_kl.detach(),
            "weighted_loss_kl": (self.lambda_kl * loss_kl).detach(),
            "loss_pl": loss_pl.detach(),
            "weighted_loss_pl": (self.lambda_pl * loss_pl).detach(),
            "pl_coverage": pl_coverage.detach(),
            "pl_clip_conf": pl_clip_conf.detach(),
            "pl_student_conf": pl_student_conf.detach(),
            "clip_student_agreement": clip_student_agreement.detach(),
            "acc_src": torch.tensor(acc, device=loss_ce.device),
            "source_style_norm": source_norm,
            "target_style_norm": target_norm,
            "gap_style_norm": gap_norm,
            "image_token_norm": image_norm,
            "vctx_norm": vctx_norm,
        }
        for domain_name, entropy in entropy_by_domain.items():
            outputs[f"loss_em_{domain_name}"] = entropy.detach()
        for domain_name, kl in kl_by_domain.items():
            outputs[f"loss_kl_{domain_name}"] = kl.detach()
        for domain_name, pl in pl_by_domain.items():
            outputs[f"loss_pl_{domain_name}"] = pl.detach()
        for domain_name, stats_i in pl_stats_by_domain.items():
            outputs[f"pl_coverage_{domain_name}"] = stats_i["coverage"].detach()
            outputs[f"pl_clip_conf_{domain_name}"] = stats_i["clip_conf"].detach()
            outputs[f"pl_student_conf_{domain_name}"] = stats_i[
                "student_conf"
            ].detach()
            outputs[f"clip_student_agreement_{domain_name}"] = stats_i[
                "agreement"
            ].detach()
        return outputs

    def forward_inference(self, image, domain_name=None):
        del domain_name
        if self.use_image_tokens or not self.enable_vpt:
            clean_features, hidden_states = self._encode_clean_visual(image)
        else:
            clean_features = None
            hidden_states = None
        image_features = self._encode_adapted_visual(
            image, clean_features=clean_features
        )
        image_tokens = (
            self._compute_image_tokens(hidden_states)
            if hidden_states is not None
            else None
        )
        context = self._build_eval_context(
            batch_size=image_features.shape[0],
            device=image_features.device,
            image_tokens=image_tokens,
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
        assert tssp_cfg.IMAGE_GROUP_SIZE in [1, 2, 3, 4]
        assert tssp_cfg.N_VCTX > 0
        assert tssp_cfg.VCTX_INIT_STD > 0.0
        assert tssp_cfg.GAP_POSITION in ["after_target", "middle"]
        assert 0.0 <= tssp_cfg.PROTO_MOMENTUM < 1.0
        assert tssp_cfg.STYLE_EPS > 0.0
        assert tssp_cfg.LAMBDA_EM >= 0.0
        assert tssp_cfg.ENTROPY_EPS > 0.0
        assert tssp_cfg.LAMBDA_KL >= 0.0
        assert tssp_cfg.KL_TEMPERATURE > 0.0
        assert tssp_cfg.LAMBDA_PL >= 0.0
        assert 0.0 <= tssp_cfg.PL_THRESHOLD <= 1.0
        assert 0.0 <= tssp_cfg.PL_STUDENT_THRESHOLD <= 1.0
        if tssp_cfg.DETACH_ENTROPY_TEXT:
            assert tssp_cfg.ENABLE_VPT
            assert tssp_cfg.LAMBDA_EM > 0.0
        assert "{}" in tssp_cfg.PROMPT_TEMPLATE
        assert "{}" in tssp_cfg.ZS_PROMPT_TEMPLATE

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.CLIP_TSSP_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CLIPTSSPMTDA")
        self.model = CustomCLIPTSSPMTDA(cfg, classnames, clip_model)

        print("Freezing CLIP; updating TSSP projectors and optional VCTX")
        for name, param in self.model.named_parameters():
            trainable = (
                name.startswith("style_projector.")
                or name.startswith("image_projector.")
                or (self.model.enable_vpt and name == "image_encoder.vctx")
            )
            param.requires_grad_(trainable)

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

        trainable_params = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
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
            state_dict.pop("zs_tokenized_prompts", None)
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
