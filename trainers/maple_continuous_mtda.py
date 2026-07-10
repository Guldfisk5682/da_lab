import os.path as osp

import torch
import torch.nn as nn

from clip import clip
from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer

from trainers.maple_mtda import (
    CustomMaPLeMTDA,
    MaPLeMTDA,
    TextEncoderMaPLe,
    _get_clones,
    load_base_clip_to_cpu,
    load_clip_to_cpu,
)


class ContinuousMultiModalPromptLearner(nn.Module):
    """MaPLe prompt learner with continuous text prompt tokens.

    The text branch receives only the shallow prompt insertion, so those prompt
    token states propagate through all text transformer layers. The visual
    branch keeps MaPLe-style depth-specific projections, but every projection is
    applied to the same trainable text prompt parameter set.
    """

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

        single_layer = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        print("ContinuousMaPLeMTDA design: continuous text prompts + per-layer visual projections")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context tokens: {n_ctx}")
        print(f"Prompt depth: {self.compound_prompts_depth}")

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        visual_deep_prompts = [
            layer(self.ctx) for layer in self.compound_prompt_projections
        ]
        return (
            prompts,
            self.proj(self.ctx),
            [],
            visual_deep_prompts,
        )


class ContinuousSharedProjPromptLearner(ContinuousMultiModalPromptLearner):
    """Continuous text prompts with one shared deep visual projection."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.conv1.out_channels

        del self.compound_prompt_projections
        self.compound_prompt_projection = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)

        print(
            "ContinuousSharedProjMaPLeMTDA design: continuous text prompts "
            "+ shared deep visual projection"
        )
        print(f"Shared projection reused for {maple_cfg.PROMPT_DEPTH - 1} deep visual layers")

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        shared_deep_prompt = self.compound_prompt_projection(self.ctx)
        visual_deep_prompts = [
            shared_deep_prompt for _ in range(self.compound_prompts_depth - 1)
        ]
        return (
            prompts,
            self.proj(self.ctx),
            [],
            visual_deep_prompts,
        )


class CustomContinuousMaPLeMTDA(CustomMaPLeMTDA):
    prompt_learner_cls = ContinuousMultiModalPromptLearner
    log_prefix = "ContinuousMaPLeMTDA"

    def __init__(self, cfg, classnames, clip_model):
        nn.Module.__init__(self)
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        self.prompt_learner = self.prompt_learner_cls(
            cfg, classnames, clip_model
        )
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

        print(f"{self.log_prefix} pseudo-label weight: {self.lambda_pl}")
        print(f"{self.log_prefix} pseudo-label threshold: {self.pl_threshold}")
        print(f"{self.log_prefix} pseudo-label student threshold: {self.pl_student_threshold}")
        print(
            f"{self.log_prefix} pseudo-label low-conf only: "
            f"{self.pl_use_student_low_conf_mask}"
        )
        print(f"{self.log_prefix} zero-shot prompt template: {zs_template}")


class CustomContinuousSharedProjMaPLeMTDA(CustomContinuousMaPLeMTDA):
    prompt_learner_cls = ContinuousSharedProjPromptLearner
    log_prefix = "ContinuousSharedProjMaPLeMTDA"


@TRAINER_REGISTRY.register()
class ContinuousMaPLeMTDA(MaPLeMTDA):
    """Continuous-text MaPLe ablation under the Office-Home SS-MTDA protocol."""

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

        print("Building ContinuousMaPLeMTDA custom CLIP")
        self.model = CustomContinuousMaPLeMTDA(cfg, classnames, clip_model)

        print("Freezing CLIP image/text encoders; updating continuous MaPLe prompt learner only")
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
        self.register_model("ContinuousMaPLeMTDA", self.model, self.optim, self.sched)
        self.scaler = None
        if cfg.TRAINER.MAPLE_MTDA.PREC == "amp":
            from torch.cuda.amp import GradScaler

            self.scaler = GradScaler()

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        model_path = osp.join(directory, "ContinuousMaPLeMTDA", model_file)
        if not osp.exists(model_path):
            raise FileNotFoundError(f'Model not found at "{model_path}"')

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            state_dict.pop(key, None)

        print(f'Loading weights to ContinuousMaPLeMTDA from "{model_path}"')
        self._models["ContinuousMaPLeMTDA"].load_state_dict(state_dict, strict=False)


@TRAINER_REGISTRY.register()
class ContinuousSharedProjMaPLeMTDA(ContinuousMaPLeMTDA):
    """Continuous-text MaPLe with one shared deep visual projection."""

    model_name = "ContinuousSharedProjMaPLeMTDA"
    custom_model_cls = CustomContinuousSharedProjMaPLeMTDA

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

        print("Building ContinuousSharedProjMaPLeMTDA custom CLIP")
        self.model = self.custom_model_cls(cfg, classnames, clip_model)

        print("Freezing CLIP image/text encoders; updating shared-proj continuous MaPLe prompt learner only")
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
        self.register_model(self.model_name, self.model, self.optim, self.sched)
        self.scaler = None
        if cfg.TRAINER.MAPLE_MTDA.PREC == "amp":
            from torch.cuda.amp import GradScaler

            self.scaler = GradScaler()

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        model_path = osp.join(directory, self.model_name, model_file)
        if not osp.exists(model_path):
            raise FileNotFoundError(f'Model not found at "{model_path}"')

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            state_dict.pop(key, None)

        print(f'Loading weights to {self.model_name} from "{model_path}"')
        self._models[self.model_name].load_state_dict(state_dict, strict=False)
