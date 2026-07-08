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
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoderMaPLe(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, domain_name=None):
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


@TRAINER_REGISTRY.register()
class MaPLeMTDA(MultiTargetTrainerXU):
    """MaPLe source-only baseline under the Office-Home SS-MTDA protocol."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE_MTDA.PREC in ["fp16", "fp32", "amp"]

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
                logits = self.model(image_x)
                loss = F.cross_entropy(logits, label_x)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits = self.model(image_x)
            loss = F.cross_entropy(logits, label_x)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "source_ce": loss.item(),
        }

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
