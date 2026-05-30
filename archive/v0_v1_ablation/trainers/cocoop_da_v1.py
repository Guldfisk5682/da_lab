import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_pretrained_weights

from archive.v0_v1_ablation.models.shallow_adapt import (
    DomainStatsBank,
    FinalFeatureGate,
    ShallowAdaptation,
    compute_patch_stats,
)
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import PromptLearner, TextEncoder, load_clip_to_cpu
from archive.v0_v1_ablation.trainers.cocoop_da_v0 import VisualEncoderAdapter


class CustomCLIPDAV1(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.visual_adapter = VisualEncoderAdapter(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        da_cfg = cfg.TRAINER.COCOOP_DA
        dim = self.visual_adapter.hidden_dim
        feat_dim = self.visual_adapter.output_dim

        self.inject_layer = da_cfg.INJECT_LAYER
        self.adapt_mode = da_cfg.ADAPT_MODE
        self.eps = da_cfg.STATS.EPS
        self.use_adapted_target_eval = da_cfg.EVAL.USE_ADAPTED_TARGET
        self.debug_print_once = da_cfg.DEBUG.PRINT_ONCE
        self.force_alpha = da_cfg.GATE.FORCE_ALPHA
        self._has_printed_debug = False

        self.shallow_adapt = ShallowAdaptation(dim)
        self.final_gate = FinalFeatureGate(
            feat_dim,
            hidden_ratio=da_cfg.GATE.HIDDEN_RATIO,
            init_bias=da_cfg.GATE.INIT_BIAS,
        )
        self.source_stats_bank = DomainStatsBank(
            dim, momentum=da_cfg.STATS.MOMENTUM, eps=da_cfg.STATS.EPS
        )
        self.target_stats_bank = DomainStatsBank(
            dim, momentum=da_cfg.STATS.MOMENTUM, eps=da_cfg.STATS.EPS
        )

    def _split_hidden(self, hidden_tokens):
        return hidden_tokens[:, :1, :], hidden_tokens[:, 1:, :]

    def _adapt_patch_tokens(self, patch_tokens, ref_mu, ref_std):
        ref_mu = ref_mu.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        ref_std = ref_std.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        mu, std = compute_patch_stats(patch_tokens, eps=self.eps)
        normalized = (patch_tokens - mu) / (std + self.eps)
        normalized = normalized.to(patch_tokens.dtype)
        adapted = self.shallow_adapt(normalized, ref_mu, ref_std)
        adapted = adapted.to(patch_tokens.dtype)
        return adapted, mu, std

    def _normalize_feature(self, feat):
        return feat / feat.norm(dim=-1, keepdim=True)

    def _compute_alpha(self, feat_adapted):
        if self.force_alpha >= 0.0:
            alpha = torch.full(
                (feat_adapted.shape[0], 1),
                float(self.force_alpha),
                device=feat_adapted.device,
                dtype=feat_adapted.dtype,
            )
            return alpha

        alpha = self.final_gate(feat_adapted)
        return alpha.to(dtype=feat_adapted.dtype)

    def _compose_final_feature(self, feat_normal, feat_adapted):
        alpha = self._compute_alpha(feat_adapted)
        feat_final = (torch.ones_like(alpha) - alpha) * feat_normal + alpha * feat_adapted
        return feat_final, alpha

    def _encode_logits(self, image_features):
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner(image_features)

        logits = []
        for prompts_i, image_feature_i in zip(prompts, image_features):
            text_features = self.text_encoder(prompts_i, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_i = logit_scale * image_feature_i @ text_features.t()
            logits.append(logits_i)

        return torch.stack(logits)

    def _forward_source_features(self, image_s, image_t):
        hidden_s = self.visual_adapter.forward_until(image_s, self.inject_layer)
        hidden_t = self.visual_adapter.forward_until(image_t, self.inject_layer)

        cls_s, patch_s = self._split_hidden(hidden_s)
        _, patch_t = self._split_hidden(hidden_t)

        mu_s, std_s = compute_patch_stats(patch_s, eps=self.eps)
        mu_t, std_t = compute_patch_stats(patch_t, eps=self.eps)

        self.source_stats_bank.update(mu_s.detach(), std_s.detach())
        self.target_stats_bank.update(mu_t.detach(), std_t.detach())

        target_mu, target_std = self.target_stats_bank.get()
        patch_s_adapted, _, _ = self._adapt_patch_tokens(patch_s, target_mu, target_std)

        hidden_s_adapted = torch.cat([cls_s, patch_s_adapted], dim=1).to(hidden_s.dtype)
        feat_normal = self.visual_adapter.forward_from(
            hidden_s, start_layer=self.inject_layer + 1
        )
        feat_adapted = self.visual_adapter.forward_from(
            hidden_s_adapted, start_layer=self.inject_layer + 1
        )
        feat_final, alpha = self._compose_final_feature(feat_normal, feat_adapted)

        debug = {
            "hidden_s": hidden_s,
            "hidden_t": hidden_t,
            "patch_s": patch_s,
            "patch_t": patch_t,
            "mu_s": mu_s,
            "std_s": std_s,
            "mu_t": mu_t,
            "std_t": std_t,
            "target_mu_bank": target_mu,
            "target_std_bank": target_std,
            "feat_normal": feat_normal,
            "feat_adapted": feat_adapted,
            "feat_final": feat_final,
            "alpha": alpha,
        }
        return feat_normal, feat_adapted, feat_final, debug

    def _forward_target_features(self, image):
        hidden_t = self.visual_adapter.forward_until(image, self.inject_layer)
        cls_t, patch_t = self._split_hidden(hidden_t)
        feat_normal = self.visual_adapter.forward_from(
            hidden_t, start_layer=self.inject_layer + 1
        )

        if self.use_adapted_target_eval and bool(self.source_stats_bank.initialized.item()):
            source_mu, source_std = self.source_stats_bank.get()
            patch_t_adapted, _, _ = self._adapt_patch_tokens(patch_t, source_mu, source_std)
            hidden_t_adapted = torch.cat([cls_t, patch_t_adapted], dim=1).to(hidden_t.dtype)
            feat_adapted = self.visual_adapter.forward_from(
                hidden_t_adapted, start_layer=self.inject_layer + 1
            )
            feat_final, alpha = self._compose_final_feature(feat_normal, feat_adapted)
        else:
            feat_adapted = feat_normal
            feat_final = feat_normal
            alpha = self._compute_alpha(feat_adapted)

        return feat_normal, feat_adapted, feat_final, alpha

    def _log_debug_once(self, debug, logits_s, loss_src):
        if not self.debug_print_once or self._has_printed_debug:
            return

        print("[CoCoOpDAV1 debug]")
        print("h_s shape:", tuple(debug["hidden_s"].shape))
        print("h_t shape:", tuple(debug["hidden_t"].shape))
        print("p_s shape:", tuple(debug["patch_s"].shape))
        print("p_t shape:", tuple(debug["patch_t"].shape))
        print("feat_normal shape:", tuple(debug["feat_normal"].shape))
        print("feat_adapted shape:", tuple(debug["feat_adapted"].shape))
        print("feat_final shape:", tuple(debug["feat_final"].shape))
        print("alpha shape:", tuple(debug["alpha"].shape))
        print("logits_s shape:", tuple(logits_s.shape))
        print("loss_src:", float(loss_src.detach().item()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_t):
        feat_normal, feat_adapted, feat_final, debug = self._forward_source_features(
            image_s, image_t
        )

        feat_normal = self._normalize_feature(feat_normal)
        feat_adapted = self._normalize_feature(feat_adapted)
        feat_final = self._normalize_feature(feat_final)

        logits_s = self._encode_logits(feat_final)
        loss_src = F.cross_entropy(logits_s, label_s)

        alpha = debug["alpha"]
        self._log_debug_once(debug, logits_s, loss_src)

        acc = compute_accuracy(logits_s, label_s)[0].item()
        return {
            "loss": loss_src,
            "loss_src": loss_src.detach(),
            "acc_src": torch.tensor(acc, device=loss_src.device),
            "alpha_mean": alpha.mean().detach(),
            "alpha_std": alpha.std(unbiased=False).detach(),
            "alpha_min": alpha.min().detach(),
            "alpha_max": alpha.max().detach(),
            "source_mu_norm": self.source_stats_bank.running_mu.norm().detach(),
            "source_std_mean": self.source_stats_bank.running_std.mean().detach(),
            "target_mu_norm": self.target_stats_bank.running_mu.norm().detach(),
            "target_std_mean": self.target_stats_bank.running_std.mean().detach(),
        }

    def forward_inference(self, image):
        feat_normal, feat_adapted, feat_final, _ = self._forward_target_features(image)
        feat_normal = self._normalize_feature(feat_normal)
        feat_adapted = self._normalize_feature(feat_adapted)
        feat_final = self._normalize_feature(feat_final)
        return self._encode_logits(feat_final)


@TRAINER_REGISTRY.register()
class CoCoOpDAV1(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.COCOOP_DA.ADAPT_MODE in ["s2t", "bidirect"]
        assert cfg.TRAINER.COCOOP_DA.INJECT_LAYER >= 1

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CoCoOpDAV1")
        self.model = CustomCLIPDAV1(cfg, classnames, clip_model)
        self._freeze_parameters()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")

        param_groups = []
        adaptation_params = [
            p for p in self.model.shallow_adapt.parameters() if p.requires_grad
        ]
        gate_params = [p for p in self.model.final_gate.parameters() if p.requires_grad]
        prompt_params = [
            p for p in self.model.prompt_learner.parameters() if p.requires_grad
        ]

        if adaptation_params:
            param_groups.append({"params": adaptation_params})
        if gate_params:
            param_groups.append({"params": gate_params})
        if prompt_params:
            param_groups.append(
                {
                    "params": prompt_params,
                    "lr": cfg.OPTIM.LR * cfg.TRAINER.COCOOP_DA.TRAIN.PROMPT_LR_MULT,
                }
            )

        self.optim = build_optimizer(self.model, cfg.OPTIM, param_groups=param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("cocoop_da_v1", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _freeze_parameters(self):
        cfg = self.cfg

        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        for param in self.model.shallow_adapt.parameters():
            param.requires_grad_(True)
        for param in self.model.final_gate.parameters():
            param.requires_grad_(True)

        if cfg.TRAINER.COCOOP_DA.TRAIN.TRAIN_PROMPT_LEARNER:
            for param in self.model.prompt_learner.parameters():
                param.requires_grad_(True)

        enabled = sorted(
            name for name, param in self.model.named_parameters() if param.requires_grad
        )
        print("Parameters to be updated:")
        for name in enabled:
            print(f"  - {name}")

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                outputs = self.model.forward_train(image_x, label_x, image_u)
                loss = outputs["loss"]
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            outputs = self.model.forward_train(image_x, label_x, image_u)
            loss = outputs["loss"]
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {}
        for key, value in outputs.items():
            if key == "loss":
                continue
            if torch.is_tensor(value):
                loss_summary[key] = value.item()
            else:
                loss_summary[key] = float(value)
        loss_summary["loss"] = loss.item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.model.forward_inference(input)

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
            epoch = checkpoint["epoch"]

            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
