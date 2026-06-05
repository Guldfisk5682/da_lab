import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_pretrained_weights

from models.visual_prompt import DomainTextVCTXGenerator, ShallowVPTVisualEncoder
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import PromptLearner, TextEncoder, load_clip_to_cpu
from trainers.mtda_base import MultiTargetTrainerXU


class CustomCLIPVPTMTDA(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        vpt_cfg = cfg.TRAINER.COCOOP_VPT_MTDA

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ShallowVPTVisualEncoder(
            clip_model.visual,
            n_vctx=vpt_cfg.N_VCTX,
            init_std=vpt_cfg.VCTX_INIT_STD,
            prompt_depth=vpt_cfg.VISION_PROMPT_DEPTH,
            prompt_position=vpt_cfg.VCTX_POSITION,
        )
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.debug_print_once = vpt_cfg.DEBUG.PRINT_ONCE
        self._has_printed_debug = False
        self.domain_vctx = None
        self.domain_aware_enabled = vpt_cfg.DOMAIN_AWARE.ENABLED
        self.domain_hidden_dim = vpt_cfg.DOMAIN_AWARE.HIDDEN_DIM
        self.domain_gamma_init = vpt_cfg.DOMAIN_AWARE.GAMMA_INIT
        self.domain_gamma_learnable = vpt_cfg.DOMAIN_AWARE.GAMMA_LEARNABLE
        self.domain_names = list(cfg.DATASET.TARGET_DOMAINS)

        template = str(vpt_cfg.DOMAIN_AWARE.TEXT_TEMPLATE)
        prompts = [
            template.format(domain=domain_name.replace("_", " "))
            for domain_name in self.domain_names
        ]
        self.register_buffer(
            "domain_tokenized_prompts",
            torch.cat([clip.tokenize(prompt) for prompt in prompts]),
        )

    def initialize_domain_vctx(self):
        if not self.domain_aware_enabled or self.domain_vctx is not None:
            return

        with torch.no_grad():
            tokenized = self.domain_tokenized_prompts.to(self.token_embedding.weight.device)
            embedding = self.token_embedding(tokenized).type(self.dtype)
            domain_text_features = self.text_encoder(embedding, tokenized).float()
            domain_text_features = F.normalize(domain_text_features, dim=-1)

        self.domain_vctx = DomainTextVCTXGenerator(
            domain_names=self.domain_names,
            domain_text_features=domain_text_features,
            prompt_depth=self.image_encoder.prompt_depth,
            n_vctx=self.image_encoder.n_vctx,
            vision_dim=self.image_encoder.visual.conv1.out_channels,
            hidden_dim=self.domain_hidden_dim,
            gamma_init=self.domain_gamma_init,
            gamma_learnable=self.domain_gamma_learnable,
        ).to(self.token_embedding.weight.device)

    def _domain_residual(self, domain_names=None):
        if self.domain_vctx is None:
            return None
        return self.domain_vctx(domain_names=domain_names)

    def _encode_logits(self, image_features):
        image_features_norm = image_features.float()
        image_features_norm = image_features_norm / image_features_norm.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        prompts = self.prompt_learner(image_features)
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

    def _build_debug_snapshot(
        self, image_s, image_u_dict, image_features, logits, loss, vctx_residual
    ):
        if not self.debug_print_once or self._has_printed_debug:
            return

        pi_img = self.prompt_learner.meta_net(image_features)
        print("[CoCoOpVPTMTDA debug]")
        print("source domain:", self.cfg.DATASET.SOURCE_DOMAINS[0])
        print("target domains:", list(image_u_dict.keys()))
        print("source batch shape:", tuple(image_s.shape))
        for domain_name, image_u in image_u_dict.items():
            print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
        print("vctx shape:", tuple(self.image_encoder.vctx.shape))
        print("vision prompt depth:", self.image_encoder.prompt_depth)
        print("vctx position:", self.image_encoder.prompt_position)
        print("vctx norm:", float(self.image_encoder.vctx.detach().float().norm().item()))
        print("domain aware enabled:", self.domain_vctx is not None)
        if self.domain_vctx is not None:
            print("domain gamma:", float(self.domain_vctx.gamma.detach().float().item()))
            print(
                "domain residual norm:",
                float(vctx_residual.detach().float().norm().item()),
            )
        print("image feature shape:", tuple(image_features.shape))
        print("pi_img shape:", tuple(pi_img.shape))
        print("logits shape:", tuple(logits.shape))
        print("loss:", float(loss.detach().item()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_u_dict):
        vctx_residual = self._domain_residual(domain_names=list(image_u_dict.keys()))
        image_features = self.image_encoder(image_s.type(self.dtype), vctx_residual=vctx_residual)
        logits = self._encode_logits(image_features)
        loss_ce = F.cross_entropy(logits, label_s)
        self._build_debug_snapshot(
            image_s, image_u_dict, image_features, logits, loss_ce, vctx_residual
        )

        acc = compute_accuracy(logits, label_s)[0].item()
        domain_gamma = torch.zeros((), device=loss_ce.device)
        domain_residual_norm = torch.zeros((), device=loss_ce.device)
        if self.domain_vctx is not None:
            domain_gamma = self.domain_vctx.gamma.detach().float()
            domain_residual_norm = vctx_residual.detach().float().norm()

        return {
            "loss": loss_ce,
            "loss_ce": loss_ce.detach(),
            "acc_src": torch.tensor(acc, device=loss_ce.device),
            "vctx_norm": self.image_encoder.vctx.detach().float().norm(),
            "domain_gamma": domain_gamma,
            "domain_residual_norm": domain_residual_norm,
        }

    def forward_inference(self, image, domain_name=None):
        vctx_residual = self._domain_residual(domain_names=domain_name)
        image_features = self.image_encoder(image.type(self.dtype), vctx_residual=vctx_residual)
        return self._encode_logits(image_features)

    def forward(self, image, domain_name=None):
        return self.forward_inference(image, domain_name=domain_name)


@TRAINER_REGISTRY.register()
class CoCoOpVPTMTDA(MultiTargetTrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP_VPT_MTDA.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.COCOOP_VPT_MTDA.N_VCTX > 0
        assert cfg.TRAINER.COCOOP_VPT_MTDA.VISION_PROMPT_DEPTH > 0
        assert cfg.TRAINER.COCOOP_VPT_MTDA.VCTX_POSITION in ["append", "insert"]
        assert cfg.TRAINER.COCOOP_VPT_MTDA.DOMAIN_AWARE.HIDDEN_DIM >= 0

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COCOOP_VPT_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CoCoOpVPTMTDA")
        self.model = CustomCLIPVPTMTDA(cfg, classnames, clip_model)
        self.model.to(self.device)
        self.model.initialize_domain_vctx()

        print("Turning off CLIP weights; updating CoCoOp prompt learner and VPT")
        for name, param in self.model.named_parameters():
            trainable = (
                name.startswith("prompt_learner.")
                or name == "image_encoder.vctx"
                or name.startswith("domain_vctx.")
            )
            param.requires_grad_(trainable)

        enabled = sorted(
            name for name, param in self.model.named_parameters() if param.requires_grad
        )
        print("Parameters to be updated:")
        for name in enabled:
            print(f"  - {name}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("cocoop_vpt_mtda", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_VPT_MTDA.PREC == "amp" else None

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
        prec = self.cfg.TRAINER.COCOOP_VPT_MTDA.PREC

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
