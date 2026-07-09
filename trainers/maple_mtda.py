import copy
import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer

from trainers.cocoop import load_clip_to_cpu as load_base_clip_to_cpu
from trainers.mtda_base import MultiTargetTrainerXU

_tokenizer = _Tokenizer()


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def load_clip_to_cpu(cfg):
    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE_MTDA.N_CTX,
    }

    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoderMaPLe(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer([x, compound_prompts_deeper_text, 0])[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        n_cls = len(classnames)
        n_ctx = maple_cfg.N_CTX
        ctx_init = maple_cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.conv1.out_channels
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        if cfg_imsize != clip_imsize:
            raise ValueError(f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})")
        if maple_cfg.PROMPT_DEPTH < 1:
            raise ValueError("TRAINER.MAPLE_MTDA.PROMPT_DEPTH should be >= 1")

        self.compound_prompts_depth = maple_cfg.PROMPT_DEPTH

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        self.proj = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)

        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for prompt in self.compound_prompts_text:
            nn.init.normal_(prompt, std=0.02)

        single_layer = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        print("MaPLeMTDA design: source-only multi-modal prompt baseline")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        print(f"Prompt depth: {self.compound_prompts_depth}")

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        visual_deep_prompts = [
            layer(self.compound_prompts_text[index])
            for index, layer in enumerate(self.compound_prompt_projections)
        ]
        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


class CustomMaPLeMTDA(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoderMaPLe(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.lambda_pl = float(maple_cfg.LAMBDA_PL)
        self.pl_threshold = float(maple_cfg.PL_THRESHOLD)
        self.pl_student_threshold = float(maple_cfg.PL_STUDENT_THRESHOLD)
        self.pl_use_student_low_conf_mask = bool(
            maple_cfg.PL_USE_STUDENT_LOW_CONF_MASK
        )
        self.debug_print_once = bool(maple_cfg.DEBUG.PRINT_ONCE)
        self._debug_printed = False

        zs_template = str(maple_cfg.ZS_PROMPT_TEMPLATE)
        if "{}" not in zs_template:
            raise ValueError("TRAINER.MAPLE_MTDA.ZS_PROMPT_TEMPLATE must contain '{}'")
        classnames = [name.replace("_", " ") for name in classnames]
        zs_prompts = [zs_template.format(name) for name in classnames]
        self.register_buffer(
            "zs_tokenized_prompts",
            torch.cat([clip.tokenize(prompt) for prompt in zs_prompts]),
        )

        print(f"MaPLeMTDA pseudo-label weight: {self.lambda_pl}")
        print(f"MaPLeMTDA pseudo-label threshold: {self.pl_threshold}")
        print(f"MaPLeMTDA pseudo-label student threshold: {self.pl_student_threshold}")
        print(
            "MaPLeMTDA pseudo-label low-conf only: "
            f"{self.pl_use_student_low_conf_mask}"
        )
        print(f"MaPLeMTDA zero-shot prompt template: {zs_template}")

    @staticmethod
    def _ensure_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values detected in {name}")

    def forward(self, image, domain_name=None):
        del domain_name
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner()

        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        )
        image_features = self.image_encoder(
            image.type(self.dtype), shared_ctx, deep_compound_prompts_vision
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        return logits

    @torch.no_grad()
    def _encode_reference_image(self, image):
        visual = self.image_encoder
        x = visual.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer([x, [], 0])[0]
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        x = x.float()
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._ensure_finite("maple_reference_image_features", x)
        return x

    @torch.no_grad()
    def _zero_shot_text_features(self, device):
        tokenized_prompts = self.zs_tokenized_prompts.to(device)
        prompts = self.token_embedding(tokenized_prompts).type(self.dtype)
        text_features = self.text_encoder(prompts, tokenized_prompts, []).float()
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        self._ensure_finite("maple_zero_shot_text_features", text_features)
        return text_features

    @torch.no_grad()
    def _compute_reference_logits(self, image):
        image_features = self._encode_reference_image(image)
        text_features = self._zero_shot_text_features(image_features.device)
        logits = self.logit_scale.float().exp() * image_features @ text_features.t()
        self._ensure_finite("maple_reference_logits", logits)
        return logits.detach()

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
        self._ensure_finite("maple_pseudo_label_loss", loss)

        stats = {
            "coverage": mask_float.mean(),
            "clip_conf": reference_conf.mean(),
            "student_conf": student_conf.mean(),
            "agreement": (student_label == reference_label).float().mean(),
        }
        return loss, stats

    def forward_train(self, image_s, label_s, image_u_dict):
        logits_s = self(image_s)
        loss_ce = F.cross_entropy(logits_s, label_s)
        self._ensure_finite("maple_source_ce", loss_ce)

        pl_by_domain = {}
        pl_stats_by_domain = {}
        if self.lambda_pl > 0.0:
            for domain_name, image_u in image_u_dict.items():
                target_logits = self(image_u)
                reference_logits = self._compute_reference_logits(image_u)
                pl_loss, pl_stats = self._pseudo_label_loss(
                    target_logits, reference_logits
                )
                pl_by_domain[domain_name] = pl_loss
                pl_stats_by_domain[domain_name] = pl_stats
            loss_pl = torch.stack(list(pl_by_domain.values())).mean()
            pl_coverage = torch.stack(
                [stats["coverage"] for stats in pl_stats_by_domain.values()]
            ).mean()
            pl_clip_conf = torch.stack(
                [stats["clip_conf"] for stats in pl_stats_by_domain.values()]
            ).mean()
            pl_student_conf = torch.stack(
                [stats["student_conf"] for stats in pl_stats_by_domain.values()]
            ).mean()
            clip_student_agreement = torch.stack(
                [stats["agreement"] for stats in pl_stats_by_domain.values()]
            ).mean()
        else:
            loss_pl = loss_ce.new_zeros(())
            pl_coverage = loss_ce.new_zeros(())
            pl_clip_conf = loss_ce.new_zeros(())
            pl_student_conf = loss_ce.new_zeros(())
            clip_student_agreement = loss_ce.new_zeros(())

        loss_total = loss_ce + self.lambda_pl * loss_pl
        self._ensure_finite("maple_loss_total", loss_total)

        if self.debug_print_once and not self._debug_printed:
            print("[MaPLeMTDA PL debug]")
            print("source batch shape:", tuple(image_s.shape))
            for domain_name, image_u in image_u_dict.items():
                print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
            print("source logits shape:", tuple(logits_s.shape))
            print("lambda_pl:", self.lambda_pl)
            print("loss_ce:", float(loss_ce.detach().item()))
            print("loss_pl:", float(loss_pl.detach().item()))
            print("pl_coverage:", float(pl_coverage.detach().item()))
            print("loss_total:", float(loss_total.detach().item()))
            self._debug_printed = True

        outputs = {
            "loss": loss_total,
            "source_ce": loss_ce.detach(),
            "loss_pl": loss_pl.detach(),
            "weighted_loss_pl": (self.lambda_pl * loss_pl).detach(),
            "pl_coverage": pl_coverage.detach(),
            "pl_clip_conf": pl_clip_conf.detach(),
            "pl_student_conf": pl_student_conf.detach(),
            "clip_student_agreement": clip_student_agreement.detach(),
        }
        for domain_name, pl_loss in pl_by_domain.items():
            outputs[f"loss_pl_{domain_name}"] = pl_loss.detach()
        for domain_name, stats in pl_stats_by_domain.items():
            outputs[f"pl_coverage_{domain_name}"] = stats["coverage"].detach()
            outputs[f"pl_clip_conf_{domain_name}"] = stats["clip_conf"].detach()
            outputs[f"pl_student_conf_{domain_name}"] = stats[
                "student_conf"
            ].detach()
            outputs[f"clip_student_agreement_{domain_name}"] = stats[
                "agreement"
            ].detach()
        return outputs


@TRAINER_REGISTRY.register()
class MaPLeMTDA(MultiTargetTrainerXU):
    """MaPLe source-only baseline under the Office-Home SS-MTDA protocol."""

    def check_cfg(self, cfg):
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        assert maple_cfg.PREC in ["fp16", "fp32", "amp"]
        assert maple_cfg.LAMBDA_PL >= 0.0
        assert 0.0 <= maple_cfg.PL_THRESHOLD <= 1.0
        assert 0.0 <= maple_cfg.PL_STUDENT_THRESHOLD <= 1.0
        assert "{}" in maple_cfg.ZS_PROMPT_TEMPLATE

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        if cfg.TRAINER.MAPLE_MTDA.USE_MAPLE_CLIP_BUILD:
            clip_model = load_clip_to_cpu(cfg)
        else:
            clip_model = load_base_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building MaPLeMTDA custom CLIP")
        self.model = CustomMaPLeMTDA(cfg, classnames, clip_model)

        print("Freezing CLIP image/text encoders; updating MaPLe prompt learner only")
        for name, param in self.model.named_parameters():
            param.requires_grad_("prompt_learner" in name)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print("Trainable parameters:")
        for name in enabled:
            print(f"  {name}")
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        print(f"Trainable parameter count: {trainable_params:,}")

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MaPLeMTDA", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.MAPLE_MTDA.PREC == "amp" else None

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)

        if (
            bool(self.cfg.TRAINER.MAPLE_MTDA.DEBUG.PRINT_ONCE)
            and not getattr(self, "_debug_printed", False)
        ):
            print("MaPLeMTDA debug:")
            print(f"  source domains: {self.cfg.DATASET.SOURCE_DOMAINS}")
            print(f"  target domains: {list(image_u.keys())}")
            print(f"  source batch shape: {tuple(image_x.shape)}")
            for domain_name, image in image_u.items():
                print(f"  target batch shape[{domain_name}]: {tuple(image.shape)}")
            self._debug_printed = True

        prec = self.cfg.TRAINER.MAPLE_MTDA.PREC
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

        loss_summary = {"loss": float(loss.item())}
        for key, value in outputs.items():
            if key == "loss":
                continue
            loss_summary[key] = value.item() if torch.is_tensor(value) else float(value)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        model_path = osp.join(directory, "MaPLeMTDA", model_file)
        if not osp.exists(model_path):
            raise FileNotFoundError(f'Model not found at "{model_path}"')

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            state_dict.pop(key, None)

        print(f'Loading weights to MaPLeMTDA from "{model_path}"')
        self._models["MaPLeMTDA"].load_state_dict(state_dict, strict=False)
