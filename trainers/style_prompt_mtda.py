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
from models.style_prompt import DomainStyleMLP
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import PromptLearner, TextEncoder, load_clip_to_cpu
from trainers.mtda_base import MultiTargetTrainerXU


class StylePromptLearner(PromptLearner):
    def __init__(self, cfg, classnames, clip_model, style_dim):
        super().__init__(cfg, classnames, clip_model)
        style_cfg = cfg.TRAINER.STYLE_PROMPT
        ctx_dim = self.ctx.shape[-1]
        hidden_dim = int(style_cfg.DOMAIN_STYLE_MLP_HIDDEN)
        if hidden_dim <= 0:
            hidden_dim = int(style_cfg.STYLE_MLP_HIDDEN)
        if hidden_dim <= 0:
            hidden_dim = max(ctx_dim // 4, 1)

        self.domain_style_mlp = DomainStyleMLP(
            input_dim=style_dim,
            hidden_dim=hidden_dim,
            output_dim=ctx_dim,
        )

        beta_tensor = torch.tensor(float(style_cfg.BETA_INIT), dtype=self.ctx.dtype)
        if style_cfg.BETA_LEARNABLE:
            self.beta = nn.Parameter(beta_tensor)
        else:
            self.register_buffer("beta", beta_tensor)

    def compute_pi_img(self, im_features):
        return self.meta_net(im_features)

    def compute_pi_domain(self, domain_style):
        squeeze = False
        if domain_style.dim() == 1:
            domain_style = domain_style.unsqueeze(0)
            squeeze = True

        pi_domain = self.domain_style_mlp(domain_style.float())
        if squeeze:
            pi_domain = pi_domain[0]
        return pi_domain

    def build_prompts(self, pi_img, pi_domain=None):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        if pi_domain is None:
            pi_domain_batch = torch.zeros_like(pi_img)
        elif pi_domain.dim() == 1:
            pi_domain_batch = pi_domain.unsqueeze(0).expand(pi_img.shape[0], -1)
        else:
            pi_domain_batch = pi_domain

        pi_domain_batch = pi_domain_batch.to(pi_img.dtype)
        beta = self.beta.to(pi_img.dtype)

        ctx_shifted = (
            ctx.unsqueeze(0)
            + pi_img.unsqueeze(1)
            + beta * pi_domain_batch.unsqueeze(1)
        )

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            prompts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(prompts_i)
        prompts = torch.stack(prompts)

        return prompts, {"pi_img": pi_img, "pi_domain": pi_domain_batch, "beta": beta}

    def forward(self, im_features, pi_domain=None):
        pi_img = self.compute_pi_img(im_features)
        return self.build_prompts(pi_img, pi_domain=pi_domain)


class CustomCLIPStylePromptMTDA(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.dtype = clip_model.dtype
        self.token_scope = cfg.TRAINER.STYLE_PROMPT.TOKEN_SCOPE
        self.style_layer = cfg.TRAINER.STYLE_PROMPT.STYLE_LAYER
        self.style_eps = cfg.TRAINER.STYLE_PROMPT.EPS
        self.lambda_ent = float(cfg.TRAINER.STYLE_PROMPT.LAMBDA_ENT)
        self.target_domains = list(cfg.DATASET.TARGET_DOMAINS)
        self.debug_print_once = cfg.TRAINER.STYLE_PROMPT_MTDA.DEBUG.PRINT_ONCE
        self._has_printed_debug = False

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

    @staticmethod
    def _ensure_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values detected in {name}")

    def _extract_style(self, image):
        hidden = self.visual_adapter.forward_until(image, self.style_layer)
        if self.token_scope != "patch":
            raise ValueError(f"Unsupported TOKEN_SCOPE={self.token_scope}")
        return compute_patch_style(patch_tokens(hidden), eps=self.style_eps)

    def _compute_domain_statistics(self, image_u_dict):
        target_batch_styles = OrderedDict()
        pi_domain_by_target = OrderedDict()

        for domain_name, image_u in image_u_dict.items():
            style_t = self._extract_style(image_u)
            style_domain = style_t.mean(dim=0)
            target_batch_styles[domain_name] = style_domain
            pi_domain_by_target[domain_name] = self.prompt_learner.compute_pi_domain(style_domain)

        return target_batch_styles, pi_domain_by_target

    def _compute_logits(self, image_features, prompts):
        image_features_norm = image_features.float()
        image_features_norm = image_features_norm / image_features_norm.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        logit_scale = self.logit_scale.float().exp()
        logits = []
        for prompts_i, image_feature_i in zip(prompts, image_features_norm):
            text_features = self.text_encoder(prompts_i, self.tokenized_prompts).float()
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            logits_i = logit_scale * image_feature_i @ text_features.t()
            logits.append(logits_i)
        return torch.stack(logits)

    @staticmethod
    def _entropy_from_logits(logits):
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=1).mean()

    def _log_debug_once(
        self,
        image_s,
        image_u_dict,
        target_batch_styles,
        pi_domain_by_target,
        pi_img_s,
        logits_example,
        beta,
        loss_src,
        loss_ent,
        loss_total,
    ):
        if not self.debug_print_once or self._has_printed_debug:
            return

        print("[StylePromptMTDA debug]")
        print("source domain:", self.cfg.DATASET.SOURCE_DOMAINS[0])
        print("target domains:", self.target_domains)
        print("source batch shape:", tuple(image_s.shape))
        for domain_name, image_u in image_u_dict.items():
            print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
        for domain_name, style_domain in target_batch_styles.items():
            print(f"target batch style shape [{domain_name}]:", tuple(style_domain.shape))
        for domain_name, pi_domain in pi_domain_by_target.items():
            print(f"pi_domain shape [{domain_name}]:", tuple(pi_domain.shape))
        print("pi_img shape:", tuple(pi_img_s.shape))
        print("beta:", float(beta.detach().item()))
        print(
            "pi_domain_norm:",
            float(torch.stack([v.norm() for v in pi_domain_by_target.values()]).mean().detach().item()),
        )
        print("logits shape:", tuple(logits_example.shape))
        print("loss_src:", float(loss_src.detach().item()))
        print("loss_ent:", float(loss_ent.detach().item()))
        print("loss_total:", float(loss_total.detach().item()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_u_dict):
        image_features_s = self.image_encoder(image_s.type(self.dtype))
        self._ensure_finite("image_features_s", image_features_s)
        pi_img_s = self.prompt_learner.compute_pi_img(image_features_s)
        self._ensure_finite("pi_img_s", pi_img_s)

        target_batch_styles, pi_domain_by_target = self._compute_domain_statistics(image_u_dict)
        for domain_name, style_domain in target_batch_styles.items():
            self._ensure_finite(f"style_domain[{domain_name}]", style_domain)
        for domain_name, pi_domain in pi_domain_by_target.items():
            self._ensure_finite(f"pi_domain[{domain_name}]", pi_domain)

        loss_src_terms = []
        acc_terms = []
        logits_example = None

        for domain_name in self.target_domains:
            pi_domain = pi_domain_by_target[domain_name]
            prompts_s, prompt_info_s = self.prompt_learner.build_prompts(pi_img_s, pi_domain=pi_domain)
            logits_s = self._compute_logits(image_features_s, prompts_s)
            self._ensure_finite(f"logits_s[{domain_name}]", logits_s)
            if logits_example is None:
                logits_example = logits_s
            loss_ce_d = F.cross_entropy(logits_s, label_s)
            self._ensure_finite(f"loss_ce[{domain_name}]", loss_ce_d)
            loss_src_terms.append(loss_ce_d)
            acc_terms.append(compute_accuracy(logits_s, label_s)[0].item())

            assert prompt_info_s["pi_domain"].shape == pi_img_s.shape
            self._ensure_finite(f"prompt_info_s.pi_domain[{domain_name}]", prompt_info_s["pi_domain"])

        loss_src = torch.stack(loss_src_terms).mean()
        self._ensure_finite("loss_src", loss_src)

        if self.lambda_ent > 0:
            loss_ent_terms = []
            for domain_name, image_u in image_u_dict.items():
                pi_domain = pi_domain_by_target[domain_name]
                image_features_t = self.image_encoder(image_u.type(self.dtype))
                self._ensure_finite(f"image_features_t[{domain_name}]", image_features_t)
                pi_img_t = self.prompt_learner.compute_pi_img(image_features_t)
                self._ensure_finite(f"pi_img_t[{domain_name}]", pi_img_t)
                prompts_t, _ = self.prompt_learner.build_prompts(pi_img_t, pi_domain=pi_domain)
                logits_t = self._compute_logits(image_features_t, prompts_t)
                self._ensure_finite(f"logits_t[{domain_name}]", logits_t)
                loss_ent_d = self._entropy_from_logits(logits_t)
                self._ensure_finite(f"loss_ent[{domain_name}]", loss_ent_d)
                loss_ent_terms.append(loss_ent_d)
            loss_ent = torch.stack(loss_ent_terms).mean()
        else:
            loss_ent = torch.zeros((), device=loss_src.device, dtype=loss_src.dtype)
        self._ensure_finite("loss_ent", loss_ent)

        beta = self.prompt_learner.beta.to(loss_src.dtype)
        loss_total = loss_src + self.lambda_ent * loss_ent
        self._ensure_finite("loss_total", loss_total)

        self._log_debug_once(
            image_s=image_s,
            image_u_dict=image_u_dict,
            target_batch_styles=target_batch_styles,
            pi_domain_by_target=pi_domain_by_target,
            pi_img_s=pi_img_s,
            logits_example=logits_example,
            beta=beta,
            loss_src=loss_src,
            loss_ent=loss_ent,
            loss_total=loss_total,
        )

        pi_domain_norm = torch.stack([v.norm() for v in pi_domain_by_target.values()]).mean().detach()
        acc_src = torch.tensor(sum(acc_terms) / len(acc_terms), device=loss_total.device)

        return {
            "loss": loss_total,
            "loss_src": loss_src.detach(),
            "loss_ent": loss_ent.detach(),
            "loss_total": loss_total.detach(),
            "acc_src": acc_src,
            "beta": beta.detach(),
            "pi_domain_norm": pi_domain_norm,
        }

    def forward_inference(self, image, domain_name=None):
        image_features = self.image_encoder(image.type(self.dtype))
        style_domain = self._extract_style(image).mean(dim=0)
        pi_domain = self.prompt_learner.compute_pi_domain(style_domain)
        prompts, _ = self.prompt_learner(image_features, pi_domain=pi_domain)
        return self._compute_logits(image_features, prompts)

    def forward(self, image, domain_name=None):
        return self.forward_inference(image, domain_name=domain_name)


@TRAINER_REGISTRY.register()
class StylePromptMTDA(MultiTargetTrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.STYLE_PROMPT_MTDA.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.STYLE_PROMPT.TOKEN_SCOPE == "patch"
        assert cfg.TRAINER.STYLE_PROMPT.LAMBDA_ENT >= 0.0

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
