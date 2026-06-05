import torch
import torch.nn as nn


class CLIPVPTVisualEncoder(nn.Module):
    """CLIP ViT visual encoder with optional persistent appended VCTX tokens."""

    def __init__(self, visual, enable_vpt=False, n_vctx=8, init_std=0.02):
        super().__init__()
        if not hasattr(visual, "transformer") or not hasattr(
            visual.transformer, "resblocks"
        ):
            raise TypeError("CLIPVPTVisualEncoder supports CLIP ViT backbones only")

        self.visual = visual
        self.enable_vpt = bool(enable_vpt)
        self.n_vctx = int(n_vctx)
        if self.enable_vpt and self.n_vctx <= 0:
            raise ValueError(f"n_vctx must be positive when VPT is enabled, got {n_vctx}")

        width = visual.conv1.out_channels
        if self.enable_vpt:
            self.vctx = nn.Parameter(torch.empty(self.n_vctx, width))
            nn.init.normal_(self.vctx, std=float(init_std))
        else:
            self.register_parameter("vctx", None)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _base_tokens(self, image):
        x = self.visual.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = self.visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        return x

    def forward(self, image):
        x = self._base_tokens(image)

        if self.enable_vpt:
            visual_ctx = self.vctx.to(dtype=x.dtype, device=x.device)
            visual_ctx = visual_ctx.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x
