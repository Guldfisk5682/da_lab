import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_pretrained_weights

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import TextEncoder, load_clip_to_cpu
from trainers.cocoop_da_v0 import VisualEncoderAdapter

_tokenizer = _Tokenizer()


class LegacyPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        gspa_cfg = cfg.TRAINER.GSPA_LEGACY

        n_cls = len(classnames)
        n_ctx = gspa_cfg.N_CTX
        ctx_init = gspa_cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vision_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vision_dim, max(vision_dim // 16, 1))),
                    ("silu", nn.SiLU()),
                    ("linear2", nn.Linear(max(vision_dim // 16, 1), ctx_dim)),
                ]
            )
        )
        self.meta_net.to(dtype=dtype)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, vision_feats):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        bias = self.meta_net(vision_feats)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            prompts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(prompts_i)
        prompts = torch.stack(prompts)
        return prompts


class LegacyFeatureGate(nn.Module):
    def __init__(self, dim, hidden_ratio=16, init_bias=3.0, dtype=None):
        super().__init__()
        hidden_dim = max(dim // hidden_ratio, 1)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.fc2.bias, init_bias)
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.act(self.fc1(x))))


class CustomCLIPGSPALegacy(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = LegacyPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.visual_adapter = VisualEncoderAdapter(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.visual = clip_model.visual
        self.visual_projection = clip_model.visual.proj

        gspa_cfg = cfg.TRAINER.GSPA_LEGACY
        vision_dim = self.visual.ln_post.weight.shape[0]
        self.inject_after_block = gspa_cfg.INJECT_AFTER_BLOCK
        self.eps = gspa_cfg.EPS
        self.label_smoothing = gspa_cfg.LABEL_SMOOTHING
        self.debug_print_once = gspa_cfg.DEBUG.PRINT_ONCE
        self._has_printed_debug = False

        self.gate = LegacyFeatureGate(
            vision_dim,
            hidden_ratio=gspa_cfg.GATE_HIDDEN_RATIO,
            init_bias=gspa_cfg.GATE_INIT_BIAS,
            dtype=self.dtype,
        )

    def _compute_stats(self, hidden_states):
        hidden_float = hidden_states.float()
        mu = hidden_float.mean(dim=1, keepdim=True)
        std = hidden_float.std(dim=1, keepdim=True) + self.eps
        hidden_norm = (hidden_float - mu) / std
        return mu, std, hidden_norm

    def _cross_style_swap(self, hidden_s, hidden_t):
        mu_s, std_s, h_s_norm = self._compute_stats(hidden_s)
        mu_t, std_t, h_t_norm = self._compute_stats(hidden_t)

        h_s_adapted = (h_s_norm * std_t + mu_t).to(hidden_s.dtype)
        h_t_adapted = (h_t_norm * std_s + mu_s).to(hidden_t.dtype)

        return {
            "mu_s": mu_s,
            "std_s": std_s,
            "mu_t": mu_t,
            "std_t": std_t,
            "h_s_adapted": h_s_adapted,
            "h_t_adapted": h_t_adapted,
        }

    def _forward_visual_tail(self, hidden_states):
        tokens = self.visual_adapter.tokens_forward(
            hidden_states, start_layer=self.inject_after_block + 1
        )
        cls_hidden = self.visual.ln_post(tokens[:, 0, :]).type(self.dtype)
        return tokens, cls_hidden

    def _project_image_feature(self, vision_feats):
        if self.visual_projection is None:
            return vision_feats
        return vision_feats @ self.visual_projection

    def _encode_logits(self, vision_feats):
        prompts = self.prompt_learner(vision_feats)
        image_features = self._project_image_feature(vision_feats)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits = []
        for prompts_i, image_feature_i in zip(prompts, image_features):
            text_features = self.text_encoder(prompts_i, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_i = logit_scale * image_feature_i @ text_features.t()
            logits.append(logits_i)

        return torch.stack(logits)

    def _encode_image_normal(self, image):
        hidden = self.visual_adapter.patch_embed(image)
        _, vision_feats = self._forward_visual_tail(hidden)
        return vision_feats

    def _log_debug_once(self, image_s, image_t, debug, logits_s, loss):
        if not self.debug_print_once or self._has_printed_debug:
            return

        print("[GSPALegacy debug]")
        print("x_s shape:", tuple(image_s.shape))
        print("x_t shape:", tuple(image_t.shape))
        print("h_s shape:", tuple(debug["h_s"].shape))
        print("h_t shape:", tuple(debug["h_t"].shape))
        print("h_s_adapted shape:", tuple(debug["h_s_adapted"].shape))
        print("last_hidden_s_normal shape:", tuple(debug["last_hidden_s_normal"].shape))
        print("last_hidden_s_adapted shape:", tuple(debug["last_hidden_s_adapted"].shape))
        print("gate_s shape:", tuple(debug["gate_s"].shape))
        print("fused_s shape:", tuple(debug["fused_s"].shape))
        print("logits_s shape:", tuple(logits_s.shape))
        print("loss:", float(loss.detach().item()))
        print("gate_mean:", float(debug["gate_s"].mean().detach().item()))
        print(
            "gate_std:",
            float(debug["gate_s"].std(unbiased=False).detach().item()),
        )
        print("gate_min:", float(debug["gate_s"].min().detach().item()))
        print("gate_max:", float(debug["gate_s"].max().detach().item()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_t):
        h_s = self.visual_adapter.forward_until(image_s, self.inject_after_block)
        h_t = self.visual_adapter.forward_until(image_t, self.inject_after_block)
        swapped = self._cross_style_swap(h_s, h_t)

        _, last_hidden_s_normal = self._forward_visual_tail(h_s)
        _, last_hidden_s_adapted = self._forward_visual_tail(swapped["h_s_adapted"])
        _, last_hidden_t_normal = self._forward_visual_tail(h_t)
        _, last_hidden_t_adapted = self._forward_visual_tail(swapped["h_t_adapted"])

        gate_s = self.gate(last_hidden_s_adapted).to(last_hidden_s_adapted.dtype)
        fused_s = gate_s * last_hidden_s_normal + (
            torch.ones_like(gate_s) - gate_s
        ) * last_hidden_s_adapted

        logits_s = self._encode_logits(fused_s)
        loss = F.cross_entropy(
            logits_s, label_s, label_smoothing=self.label_smoothing
        )

        debug = {
            "h_s": h_s,
            "h_t": h_t,
            "h_s_adapted": swapped["h_s_adapted"],
            "last_hidden_s_normal": last_hidden_s_normal,
            "last_hidden_s_adapted": last_hidden_s_adapted,
            "last_hidden_t_normal": last_hidden_t_normal,
            "last_hidden_t_adapted": last_hidden_t_adapted,
            "gate_s": gate_s,
            "fused_s": fused_s,
        }
        self._log_debug_once(image_s, image_t, debug, logits_s, loss)

        acc = compute_accuracy(logits_s, label_s)[0].item()
        return {
            "loss": loss,
            "acc_src": torch.tensor(acc, device=loss.device),
            "gate_mean": gate_s.mean().detach(),
            "gate_std": gate_s.std(unbiased=False).detach(),
            "gate_min": gate_s.min().detach(),
            "gate_max": gate_s.max().detach(),
        }

    def forward_inference(self, image):
        vision_feats = self._encode_image_normal(image)
        return self._encode_logits(vision_feats)


@TRAINER_REGISTRY.register()
class GSPALegacy(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.GSPA_LEGACY.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.GSPA_LEGACY.STATS_SCOPE == "all_tokens"
        assert cfg.TRAINER.GSPA_LEGACY.INJECT_AFTER_BLOCK >= 0

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.GSPA_LEGACY.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building GSPALegacy")
        self.model = CustomCLIPGSPALegacy(cfg, classnames, clip_model)
        self._freeze_parameters()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")

        param_groups = self._build_param_groups()
        self.optim = build_optimizer(self.model, cfg.OPTIM, param_groups=param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("gspa_legacy", self.model, self.optim, self.sched)

        self.scaler = (
            GradScaler() if cfg.TRAINER.GSPA_LEGACY.PREC == "amp" else None
        )

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _freeze_parameters(self):
        cfg = self.cfg
        last_n = cfg.TRAINER.GSPA_LEGACY.TRAIN_VISUAL_LAST_N
        visual_blocks = self.model.visual.transformer.resblocks
        start_block = max(len(visual_blocks) - last_n, 0)

        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        self.model.prompt_learner.ctx.requires_grad_(True)
        for param in self.model.prompt_learner.meta_net.parameters():
            param.requires_grad_(True)
        for param in self.model.gate.parameters():
            param.requires_grad_(True)

        for block in list(visual_blocks)[start_block:]:
            for param in block.parameters():
                param.requires_grad_(True)

        for param in self.model.visual.ln_post.parameters():
            param.requires_grad_(True)
        if self.model.visual_projection is not None:
            self.model.visual_projection.requires_grad_(True)

        if cfg.TRAINER.GSPA_LEGACY.TRAIN_LOGIT_SCALE:
            self.model.logit_scale.requires_grad_(True)

        enabled = sorted(
            name for name, param in self.model.named_parameters() if param.requires_grad
        )
        print("Trainable parameters:")
        for name in enabled:
            print(f"  - {name}")

    def _collect_trainable(self, params_or_modules):
        params = []
        seen = set()
        for item in params_or_modules:
            if item is None:
                continue
            if isinstance(item, nn.Parameter):
                iterable = [item]
            else:
                iterable = list(item.parameters())
            for param in iterable:
                if param.requires_grad and id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        return params

    def _build_param_groups(self):
        cfg = self.cfg
        gspa_cfg = cfg.TRAINER.GSPA_LEGACY
        base_lr = cfg.OPTIM.LR
        param_groups = []
        visual_blocks = list(self.model.visual.transformer.resblocks)
        start_block = max(len(visual_blocks) - gspa_cfg.TRAIN_VISUAL_LAST_N, 0)

        specs = [
            (
                "ctx",
                [self.model.prompt_learner.ctx],
                base_lr * gspa_cfg.LR_RATIO.CTX,
            ),
            (
                "metanet",
                [self.model.prompt_learner.meta_net],
                base_lr * gspa_cfg.LR_RATIO.METANET,
            ),
            (
                "gate",
                [self.model.gate],
                base_lr * gspa_cfg.LR_RATIO.GATE,
            ),
            (
                "vision_last",
                visual_blocks[start_block:],
                base_lr * gspa_cfg.LR_RATIO.VISION_LAST,
            ),
            (
                "vision_post_ln",
                [self.model.visual.ln_post],
                base_lr * gspa_cfg.LR_RATIO.VISION_POST_LN,
            ),
            (
                "visual_proj",
                [self.model.visual_projection],
                base_lr * gspa_cfg.LR_RATIO.VISUAL_PROJ,
            ),
        ]

        if gspa_cfg.TRAIN_LOGIT_SCALE:
            specs.append(
                (
                    "logit_scale",
                    [self.model.logit_scale],
                    base_lr * gspa_cfg.LR_RATIO.LOGIT_SCALE,
                )
            )

        for group_name, names, lr in specs:
            params = self._collect_trainable(names)
            if not params or lr <= 0:
                continue
            num_params = sum(param.numel() for param in params)
            print(
                f"Param group {group_name}: params={num_params:,}, lr={lr:.8f}"
            )
            param_groups.append({"params": params, "lr": lr})

        return param_groups

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.GSPA_LEGACY.PREC

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

        loss_summary = {"loss": loss.item()}
        for key, value in outputs.items():
            if key == "loss":
                continue
            loss_summary[key] = value.item() if torch.is_tensor(value) else float(value)

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
