import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_pretrained_weights

from models.clip_vit import VisualEncoderAdapter, compute_patch_style, patch_tokens
from models.style_prompt import StyleMLP, TargetStyleQueues
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import PromptLearner, TextEncoder, load_clip_to_cpu
from trainers.mtda_base import MultiTargetTrainerXU


class StylePromptLearner(PromptLearner):
    def __init__(self, cfg, classnames, clip_model, style_dim):
        super().__init__(cfg, classnames, clip_model)
        style_cfg = cfg.TRAINER.STYLE_PROMPT
        ctx_dim = self.ctx.shape[-1]
        hidden_dim = int(style_cfg.STYLE_MLP_HIDDEN)
        if hidden_dim <= 0:
            hidden_dim = max(ctx_dim // 4, 1)

        self.style_mlp = StyleMLP(
            input_dim=style_dim,
            hidden_dim=hidden_dim,
            output_dim=ctx_dim,
            dtype=self.ctx.dtype if cfg.TRAINER.COCOOP.PREC == "fp16" else None,
        )

        beta_tensor = torch.tensor(float(style_cfg.BETA_INIT), dtype=self.ctx.dtype)
        if style_cfg.BETA_LEARNABLE:
            self.beta = nn.Parameter(beta_tensor)
        else:
            self.register_buffer("beta", beta_tensor)

    def forward(self, im_features, style_gap=None):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        pi_img = self.meta_net(im_features)
        if style_gap is None:
            pi_style = torch.zeros_like(pi_img)
        else:
            pi_style = self.style_mlp(style_gap.to(pi_img.dtype))

        beta = self.beta.to(pi_img.dtype)
        ctx_shifted = (
            ctx.unsqueeze(0)
            + pi_img.unsqueeze(1)
            + beta * pi_style.unsqueeze(1)
        )

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            prompts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(prompts_i)
        prompts = torch.stack(prompts)

        return prompts, {"pi_img": pi_img, "pi_style": pi_style, "beta": beta}


class CustomCLIPStylePromptMTDA(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.dtype = clip_model.dtype
        self.token_scope = cfg.TRAINER.STYLE_PROMPT.TOKEN_SCOPE
        self.style_layer = cfg.TRAINER.STYLE_PROMPT.STYLE_LAYER
        self.style_eps = cfg.TRAINER.STYLE_PROMPT.EPS
        self.target_domains = list(cfg.DATASET.TARGET_DOMAINS)
        self.debug_print_once = cfg.TRAINER.STYLE_PROMPT_MTDA.DEBUG.PRINT_ONCE
        self._has_printed_debug = False
        self._latest_selection_distribution = OrderedDict()

        self.image_encoder = clip_model.visual
        self.visual_adapter = VisualEncoderAdapter(clip_model.visual)
        self.prompt_learner = StylePromptLearner(
            cfg,
            classnames,
            clip_model,
            style_dim=self.visual_adapter.hidden_dim * 2,
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.style_queues = TargetStyleQueues(
            self.target_domains,
            style_dim=self.visual_adapter.hidden_dim * 2,
            queue_size=cfg.TRAINER.STYLE_PROMPT.STYLE_QUEUE_SIZE,
        )

    def _extract_style(self, image):
        hidden = self.visual_adapter.forward_until(image, self.style_layer)
        if self.token_scope != "patch":
            raise ValueError(f"Unsupported TOKEN_SCOPE={self.token_scope}")
        style = compute_patch_style(patch_tokens(hidden), eps=self.style_eps)
        return style

    def _select_style_gap(self, style_source):
        selected_styles, selected_indices, _ = self.style_queues.select(style_source)
        if not self.target_domains:
            return (
                torch.zeros_like(style_source),
                torch.zeros_like(style_source),
                selected_indices,
            )

        selected_stack = torch.stack(
            [selected_styles[domain_name].to(style_source.device) for domain_name in self.target_domains],
            dim=0,
        )
        selected_mean = selected_stack.mean(dim=0)
        style_gap = (selected_stack - style_source.unsqueeze(0)).mean(dim=0)
        return style_gap, selected_mean, selected_indices

    def _selection_distribution(self, selected_indices):
        distribution = OrderedDict()
        for domain_name, indices in selected_indices.items():
            if indices.numel() == 0:
                distribution[domain_name] = {}
                continue
            valid = indices[indices >= 0]
            if valid.numel() == 0:
                distribution[domain_name] = {}
                continue
            counts = torch.bincount(valid, minlength=int(valid.max().item()) + 1)
            distribution[domain_name] = {
                int(idx): int(count.item()) for idx, count in enumerate(counts) if count.item() > 0
            }
        return distribution

    def _compute_logits(self, image_features, prompts):
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = []
        for prompts_i, image_feature_i in zip(prompts, image_features_norm):
            text_features = self.text_encoder(prompts_i, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_i = logit_scale * image_feature_i @ text_features.t()
            logits.append(logits_i)
        return torch.stack(logits)

    def _log_debug_once(
        self,
        image_s,
        image_u_dict,
        style_s,
        target_batch_styles,
        queue_lengths,
        selected_style,
        style_gap,
        pi_img,
        pi_style,
        beta,
        logits,
        loss,
        selection_distribution,
    ):
        if not self.debug_print_once or self._has_printed_debug:
            return

        print("[StylePromptMTDA debug]")
        print("source domain:", self.cfg.DATASET.SOURCE_DOMAINS[0])
        print("target domains:", self.target_domains)
        print("source batch shape:", tuple(image_s.shape))
        for domain_name, image_u in image_u_dict.items():
            print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
        print("style_s shape:", tuple(style_s.shape))
        for domain_name, batch_style in target_batch_styles.items():
            print(f"target batch style shape [{domain_name}]:", tuple(batch_style.shape))
        print("queue length per target domain:", queue_lengths)
        print("selected style shape:", tuple(selected_style.shape))
        print("style_gap shape:", tuple(style_gap.shape))
        print("pi_img shape:", tuple(pi_img.shape))
        print("pi_style shape:", tuple(pi_style.shape))
        print("beta:", float(beta.detach().item()))
        print("logits shape:", tuple(logits.shape))
        print("loss:", float(loss.detach().item()))
        print("pi_style norm:", float(pi_style.norm(dim=-1).mean().detach().item()))
        print("style_gap norm:", float(style_gap.norm(dim=-1).mean().detach().item()))
        print("selected queue index distribution:", selection_distribution)
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_u_dict):
        image_features = self.image_encoder(image_s.type(self.dtype))
        style_s = self._extract_style(image_s)

        target_batch_styles = OrderedDict()
        for domain_name, image_u in image_u_dict.items():
            style_t = self._extract_style(image_u)
            batch_style = style_t.mean(dim=0)
            target_batch_styles[domain_name] = batch_style
            self.style_queues.enqueue(domain_name, batch_style)

        style_gap, selected_style, selected_indices = self._select_style_gap(style_s)
        selection_distribution = self._selection_distribution(selected_indices)
        self._latest_selection_distribution = selection_distribution

        prompts, prompt_info = self.prompt_learner(image_features, style_gap=style_gap)
        pi_img = prompt_info["pi_img"]
        pi_style = prompt_info["pi_style"]
        beta = prompt_info["beta"]

        assert pi_style.shape == pi_img.shape
        assert torch.isfinite(pi_style).all()

        logits = self._compute_logits(image_features, prompts)
        loss_ce = F.cross_entropy(logits, label_s)
        assert torch.isfinite(loss_ce).all()

        self._log_debug_once(
            image_s=image_s,
            image_u_dict=image_u_dict,
            style_s=style_s,
            target_batch_styles=target_batch_styles,
            queue_lengths=self.style_queues.lengths(),
            selected_style=selected_style,
            style_gap=style_gap,
            pi_img=pi_img,
            pi_style=pi_style,
            beta=beta,
            logits=logits,
            loss=loss_ce,
            selection_distribution=selection_distribution,
        )

        acc = compute_accuracy(logits, label_s)[0].item()
        return {
            "loss": loss_ce,
            "loss_ce": loss_ce.detach(),
            "acc_src": torch.tensor(acc, device=loss_ce.device),
            "beta": beta.detach(),
            "pi_style_norm": pi_style.norm(dim=-1).mean().detach(),
            "style_gap_norm": style_gap.norm(dim=-1).mean().detach(),
        }

    def forward_inference(self, image, domain_name=None):
        image_features = self.image_encoder(image.type(self.dtype))
        style = self._extract_style(image)
        if all(length == 0 for length in self.style_queues.lengths().values()):
            style_gap = torch.zeros_like(style)
        else:
            style_gap, _, _ = self._select_style_gap(style)
        prompts, _ = self.prompt_learner(image_features, style_gap=style_gap)
        return self._compute_logits(image_features, prompts)

    def forward(self, image, domain_name=None):
        return self.forward_inference(image, domain_name=domain_name)


@TRAINER_REGISTRY.register()
class StylePromptMTDA(MultiTargetTrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.STYLE_PROMPT_MTDA.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.STYLE_PROMPT.SELECTION == "domainwise_top1"
        assert cfg.TRAINER.STYLE_PROMPT.DISTANCE == "cosine"
        assert cfg.TRAINER.STYLE_PROMPT.TOKEN_SCOPE == "patch"

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.STYLE_PROMPT_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building StylePromptMTDA")
        self.model = CustomCLIPStylePromptMTDA(cfg, classnames, clip_model)

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = sorted(
            name for name, param in self.model.named_parameters() if param.requires_grad
        )
        print("Parameters to be updated:")
        for name in enabled:
            print(f"  - {name}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("style_prompt_mtda", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.STYLE_PROMPT_MTDA.PREC == "amp" else None

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
        prec = self.cfg.TRAINER.STYLE_PROMPT_MTDA.PREC

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

        if model.debug_print_once and model._latest_selection_distribution:
            print("selection distribution:", model._latest_selection_distribution)

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
            state_dict.pop("prompt_learner.token_prefix", None)
            state_dict.pop("prompt_learner.token_suffix", None)

            loaded_epoch = checkpoint["epoch"]
            print(f'Loading weights to {name} from "{model_path}" (epoch = {loaded_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

