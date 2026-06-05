import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param

from models.clip_vpt import CLIPVPTVisualEncoder
from trainers.checkpoint_utils import load_checkpoint_compat
from trainers.cocoop import TextEncoder, load_clip_to_cpu
from trainers.mtda_base import MultiTargetTrainerXU


class CustomCLIPVPTMTDA(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        vpt_cfg = cfg.TRAINER.CLIP_VPT_MTDA

        self.cfg = cfg
        self.image_encoder = CLIPVPTVisualEncoder(
            clip_model.visual,
            enable_vpt=vpt_cfg.ENABLE_VPT,
            n_vctx=vpt_cfg.N_VCTX,
            init_std=vpt_cfg.VCTX_INIT_STD,
        )
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.debug_print_once = vpt_cfg.DEBUG.PRINT_ONCE
        self._has_printed_debug = False

        template = str(vpt_cfg.PROMPT_TEMPLATE)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [template.format(name) for name in classnames]
        self.register_buffer(
            "tokenized_prompts",
            torch.cat([clip.tokenize(prompt) for prompt in prompts]),
        )

    @torch.no_grad()
    def _text_features(self):
        tokenized = self.tokenized_prompts.to(self.token_embedding.weight.device)
        embedding = self.token_embedding(tokenized).type(self.dtype)
        text_features = self.text_encoder(embedding, tokenized).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _encode_logits(self, image):
        image_features = self.image_encoder(image.type(self.dtype)).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self._text_features()
        logit_scale = self.logit_scale.float().exp()
        return logit_scale * image_features @ text_features.t()

    def _build_debug_snapshot(self, image_s, image_u_dict, logits, loss):
        if not self.debug_print_once or self._has_printed_debug:
            return

        print("[CLIPVPTMTDA debug]")
        print("source domain:", self.cfg.DATASET.SOURCE_DOMAINS[0])
        print("target domains:", list(image_u_dict.keys()))
        print("source batch shape:", tuple(image_s.shape))
        for domain_name, image_u in image_u_dict.items():
            print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
        print("enable_vpt:", self.image_encoder.enable_vpt)
        if self.image_encoder.enable_vpt:
            print("vctx shape:", tuple(self.image_encoder.vctx.shape))
            print("vctx norm:", float(self.image_encoder.vctx.detach().float().norm()))
        print("tokenized prompts shape:", tuple(self.tokenized_prompts.shape))
        print("logits shape:", tuple(logits.shape))
        print("loss:", float(loss.detach()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_u_dict):
        logits = self._encode_logits(image_s)
        loss_ce = F.cross_entropy(logits, label_s)
        self._build_debug_snapshot(image_s, image_u_dict, logits, loss_ce)

        acc = compute_accuracy(logits, label_s)[0].item()
        vctx_norm = torch.zeros((), device=loss_ce.device)
        if self.image_encoder.enable_vpt:
            vctx_norm = self.image_encoder.vctx.detach().float().norm()

        return {
            "loss": loss_ce,
            "loss_ce": loss_ce.detach(),
            "acc_src": torch.tensor(acc, device=loss_ce.device),
            "vctx_norm": vctx_norm,
        }

    def forward_inference(self, image, domain_name=None):
        del domain_name
        return self._encode_logits(image)

    def forward(self, image, domain_name=None):
        return self.forward_inference(image, domain_name=domain_name)


@TRAINER_REGISTRY.register()
class CLIPVPTMTDA(MultiTargetTrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CLIP_VPT_MTDA.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.CLIP_VPT_MTDA.N_VCTX > 0
        assert "{}" in cfg.TRAINER.CLIP_VPT_MTDA.PROMPT_TEMPLATE

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.CLIP_VPT_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CLIPVPTMTDA")
        self.model = CustomCLIPVPTMTDA(cfg, classnames, clip_model)

        print("Freezing CLIP; updating only VCTX when enabled")
        for name, param in self.model.named_parameters():
            trainable = (
                cfg.TRAINER.CLIP_VPT_MTDA.ENABLE_VPT
                and name == "image_encoder.vctx"
            )
            param.requires_grad_(trainable)

        enabled = sorted(
            name for name, param in self.model.named_parameters() if param.requires_grad
        )
        print("Parameters to be updated:")
        if enabled:
            for name in enabled:
                print(f"  - {name}")
        else:
            print("  <none> (zero-shot CLIP eval mode)")

        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")

        self.optim = None
        self.sched = None
        if enabled:
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
            self.optim = build_optimizer(trainable_params, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("clip_vpt_mtda", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP_VPT_MTDA.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _model_ref(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def forward_backward(self, batch_x, batch_u):
        if self.optim is None:
            raise RuntimeError("CLIPVPTMTDA has no trainable parameters; use eval-only")

        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)
        model = self._model_ref()
        prec = self.cfg.TRAINER.CLIP_VPT_MTDA.PREC

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
            loaded_epoch = checkpoint["epoch"]
            print(f'Loading weights to {name} from "{model_path}" (epoch = {loaded_epoch})')
            self._models[name].load_state_dict(checkpoint["state_dict"], strict=False)
