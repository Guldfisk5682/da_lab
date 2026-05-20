import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_pretrained_weights

from models.shallow_adapt import (
    DomainStatsBank,
    ShallowAdaptation,
    ShallowGate,
    compute_patch_stats,
    softmax_entropy,
)
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import PromptLearner, TextEncoder, load_clip_to_cpu


class VisualEncoderAdapter(nn.Module):
    """Expose shallow-token forward helpers on CLIP ViT."""

    def __init__(self, visual):
        super().__init__()
        if not hasattr(visual, "transformer") or not hasattr(
            visual.transformer, "resblocks"
        ):
            raise TypeError("CoCoOpDAV0 currently supports CLIP ViT backbones only")

        self.visual = visual
        self.num_layers = len(self.visual.transformer.resblocks)
        self.output_dim = self.visual.output_dim
        self.hidden_dim = self.visual.conv1.out_channels
        self.dtype = self.visual.conv1.weight.dtype

    def patch_embed(self, image):
        x = self.visual.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_token = self.visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        return x

    def tokens_forward(self, tokens, start_layer=1, end_layer=None):
        if end_layer is None:
            end_layer = self.num_layers

        x = tokens.permute(1, 0, 2)
        for layer_idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            if layer_idx < start_layer:
                continue
            if layer_idx > end_layer:
                break
            x = block(x)
        return x.permute(1, 0, 2)

    def forward_until(self, image, layer_idx):
        tokens = self.patch_embed(image)
        if layer_idx <= 0:
            return tokens
        return self.tokens_forward(tokens, start_layer=1, end_layer=layer_idx)

    def forward_from(self, hidden_tokens, start_layer):
        tokens = self.tokens_forward(hidden_tokens, start_layer=start_layer)
        features = self.visual.ln_post(tokens[:, 0, :])
        if self.visual.proj is not None:
            features = features @ self.visual.proj
        return features

    def forward(self, image):
        hidden = self.patch_embed(image)
        return self.forward_from(hidden, start_layer=1)


class CustomCLIPDA(nn.Module):
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
        self.inject_layer = da_cfg.INJECT_LAYER
        self.adapt_mode = da_cfg.ADAPT_MODE
        self.modify_cls = da_cfg.MODIFY_CLS
        self.eps = da_cfg.STATS.EPS
        self.lambda_cons = da_cfg.LOSS.LAMBDA_CONS
        self.lambda_ent = da_cfg.LOSS.LAMBDA_ENT
        self.use_adapted_target_eval = da_cfg.EVAL.USE_ADAPTED_TARGET
        self.debug_print_once = da_cfg.DEBUG.PRINT_ONCE
        self._has_printed_debug = False

        self.shallow_adapt = ShallowAdaptation(dim)
        self.gate = ShallowGate(dim, init_bias=da_cfg.GATE.INIT_BIAS)
        self.source_stats_bank = DomainStatsBank(
            dim, momentum=da_cfg.STATS.MOMENTUM, eps=da_cfg.STATS.EPS
        )
        self.target_stats_bank = DomainStatsBank(
            dim, momentum=da_cfg.STATS.MOMENTUM, eps=da_cfg.STATS.EPS
        )

    def _split_hidden(self, hidden_tokens):
        cls_token = hidden_tokens[:, :1, :]
        patch_tokens = hidden_tokens[:, 1:, :]
        return cls_token, patch_tokens

    def _fuse_tokens(self, patch_tokens, ref_mu, ref_std):
        ref_mu = ref_mu.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        ref_std = ref_std.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        mu, std = compute_patch_stats(patch_tokens, eps=self.eps)
        normalized = (patch_tokens - mu) / (std + self.eps)
        normalized = normalized.to(patch_tokens.dtype)
        adapted = self.shallow_adapt(normalized, ref_mu, ref_std)
        adapted = adapted.to(patch_tokens.dtype)
        alpha = self.gate(patch_tokens, adapted)
        fused = (torch.ones_like(alpha) - alpha) * patch_tokens + alpha * adapted
        return {
            "mu": mu,
            "std": std,
            "adapted": adapted,
            "alpha": alpha,
            "fused": fused,
        }

    def _encode_logits(self, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner(image_features)

        logits = []
        for prompts_i, image_feature_i in zip(prompts, image_features):
            text_features = self.text_encoder(prompts_i, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_i = logit_scale * image_feature_i @ text_features.t()
            logits.append(logits_i)

        logits = torch.stack(logits)
        return logits, image_features

    def _forward_source_path(self, image_s, image_t):
        hidden_s = self.visual_adapter.forward_until(image_s, self.inject_layer)
        hidden_t = self.visual_adapter.forward_until(image_t, self.inject_layer)

        cls_s, patch_s = self._split_hidden(hidden_s)
        _, patch_t = self._split_hidden(hidden_t)

        mu_s, std_s = compute_patch_stats(patch_s, eps=self.eps)
        mu_t, std_t = compute_patch_stats(patch_t, eps=self.eps)

        self.source_stats_bank.update(mu_s.detach(), std_s.detach())
        self.target_stats_bank.update(mu_t.detach(), std_t.detach())

        target_mu, target_std = self.target_stats_bank.get()
        source_state = self._fuse_tokens(patch_s, target_mu, target_std)
        hidden_s_fused = torch.cat([cls_s, source_state["fused"]], dim=1)
        hidden_s_fused = hidden_s_fused.to(hidden_s.dtype)
        feat_s_fused = self.visual_adapter.forward_from(
            hidden_s_fused, start_layer=self.inject_layer + 1
        )

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
            "alpha_s": source_state["alpha"],
            "hidden_s_fused": hidden_s_fused,
            "feat_s_fused": feat_s_fused,
        }

        return feat_s_fused, hidden_s, hidden_t, debug

    def _forward_target_path(self, hidden_t, use_adapt):
        cls_t, patch_t = self._split_hidden(hidden_t)

        if use_adapt and bool(self.source_stats_bank.initialized.item()):
            source_mu, source_std = self.source_stats_bank.get()
            target_state = self._fuse_tokens(patch_t, source_mu, source_std)
            patch_t_out = target_state["fused"]
            alpha_t = target_state["alpha"]
        else:
            patch_t_out = patch_t
            alpha_t = None

        hidden_t_out = torch.cat([cls_t, patch_t_out], dim=1)
        hidden_t_out = hidden_t_out.to(hidden_t.dtype)
        feat_t = self.visual_adapter.forward_from(
            hidden_t_out, start_layer=self.inject_layer + 1
        )
        return feat_t, alpha_t

    def _log_debug_once(self, debug, logits_s, loss_src):
        if not self.debug_print_once or self._has_printed_debug:
            return

        print("[CoCoOpDAV0 debug]")
        print("x_s shape:", tuple(debug["hidden_s"].shape))
        print("x_t shape:", tuple(debug["hidden_t"].shape))
        print("h_s shape:", tuple(debug["hidden_s"].shape))
        print("h_t shape:", tuple(debug["hidden_t"].shape))
        print("p_s shape:", tuple(debug["patch_s"].shape))
        print("p_t shape:", tuple(debug["patch_t"].shape))
        print("mu_s shape:", tuple(debug["mu_s"].shape))
        print("std_s shape:", tuple(debug["std_s"].shape))
        print("mu_t_bank shape:", tuple(debug["target_mu_bank"].shape))
        print("std_t_bank shape:", tuple(debug["target_std_bank"].shape))
        print("alpha_s shape:", tuple(debug["alpha_s"].shape))
        print("h_s_fused shape:", tuple(debug["hidden_s_fused"].shape))
        print("feat_s_fused shape:", tuple(debug["feat_s_fused"].shape))
        print("logits_s shape:", tuple(logits_s.shape))
        print("loss_src:", float(loss_src.detach().item()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_t):
        feat_s_fused, hidden_s, hidden_t, debug = self._forward_source_path(
            image_s, image_t
        )
        logits_s_fused, _ = self._encode_logits(feat_s_fused)

        with torch.no_grad():
            feat_s_normal = self.visual_adapter.forward_from(
                hidden_s, start_layer=self.inject_layer + 1
            )
            logits_s_normal, _ = self._encode_logits(feat_s_normal)

        target_use_adapt = self.adapt_mode == "bidirect"
        feat_t, alpha_t = self._forward_target_path(hidden_t, use_adapt=target_use_adapt)
        logits_t, _ = self._encode_logits(feat_t)

        loss_src = F.cross_entropy(logits_s_fused, label_s)
        loss_cons = F.kl_div(
            F.log_softmax(logits_s_fused, dim=-1),
            F.softmax(logits_s_normal.detach(), dim=-1),
            reduction="batchmean",
        )
        loss_ent = softmax_entropy(logits_t).mean()

        loss = loss_src
        if self.lambda_cons > 0:
            loss = loss + self.lambda_cons * loss_cons
        if self.lambda_ent > 0:
            loss = loss + self.lambda_ent * loss_ent

        alpha_s = debug["alpha_s"]
        assert debug["hidden_s_fused"].shape == hidden_s.shape
        assert torch.isfinite(debug["hidden_s_fused"]).all()
        assert torch.isfinite(logits_s_fused).all()
        assert alpha_s.min().item() >= 0 and alpha_s.max().item() <= 1

        self._log_debug_once(debug, logits_s_fused, loss_src)

        acc = compute_accuracy(logits_s_fused, label_s)[0].item()
        output = {
            "loss": loss,
            "loss_src": loss_src.detach(),
            "loss_cons": loss_cons.detach(),
            "loss_ent": loss_ent.detach(),
            "acc_src": torch.tensor(acc, device=loss.device),
            "alpha_mean": alpha_s.mean().detach(),
            "alpha_std": alpha_s.std(unbiased=False).detach(),
            "alpha_min": alpha_s.min().detach(),
            "alpha_max": alpha_s.max().detach(),
            "source_mu_norm": self.source_stats_bank.running_mu.norm().detach(),
            "source_std_mean": self.source_stats_bank.running_std.mean().detach(),
            "target_mu_norm": self.target_stats_bank.running_mu.norm().detach(),
            "target_std_mean": self.target_stats_bank.running_std.mean().detach(),
        }

        if alpha_t is not None:
            output["alpha_t_mean"] = alpha_t.mean().detach()

        return output

    def forward_inference(self, image):
        hidden = self.visual_adapter.forward_until(image, self.inject_layer)
        feat, _ = self._forward_target_path(
            hidden, use_adapt=self.use_adapted_target_eval
        )
        logits, _ = self._encode_logits(feat)
        return logits


@TRAINER_REGISTRY.register()
class CoCoOpDAV0(TrainerXU):
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

        print("Building CoCoOpDAV0")
        self.model = CustomCLIPDA(cfg, classnames, clip_model)

        self._freeze_parameters()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")

        param_groups = []
        adaptation_params = [p for p in self.model.shallow_adapt.parameters() if p.requires_grad]
        gate_params = [p for p in self.model.gate.parameters() if p.requires_grad]
        prompt_params = [p for p in self.model.prompt_learner.parameters() if p.requires_grad]

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
        self.register_model("cocoop_da_v0", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

    def _freeze_parameters(self):
        cfg = self.cfg

        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        for param in self.model.shallow_adapt.parameters():
            param.requires_grad_(True)
        for param in self.model.gate.parameters():
            param.requires_grad_(True)

        if cfg.TRAINER.COCOOP_DA.TRAIN.TRAIN_PROMPT_LEARNER:
            for param in self.model.prompt_learner.parameters():
                param.requires_grad_(True)

        enabled = sorted(name for name, param in self.model.named_parameters() if param.requires_grad)
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
