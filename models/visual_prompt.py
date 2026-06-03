import torch
import torch.nn as nn


class ShallowVPTVisualEncoder(nn.Module):
    """CLIP ViT visual encoder with shallow/deep visual prompt tokens."""

    def __init__(self, visual, n_vctx=4, init_std=0.02, prompt_depth=1):
        super().__init__()
        if not hasattr(visual, "transformer") or not hasattr(
            visual.transformer, "resblocks"
        ):
            raise TypeError("ShallowVPTVisualEncoder supports CLIP ViT backbones only")

        self.visual = visual
        self.n_vctx = int(n_vctx)
        self.prompt_depth = int(prompt_depth)
        if self.n_vctx <= 0:
            raise ValueError(f"n_vctx must be positive, got {n_vctx}")
        if self.prompt_depth <= 0:
            raise ValueError(f"prompt_depth must be positive, got {prompt_depth}")
        if self.prompt_depth > len(self.visual.transformer.resblocks):
            raise ValueError(
                f"prompt_depth={prompt_depth} exceeds visual depth "
                f"{len(self.visual.transformer.resblocks)}"
            )

        width = visual.conv1.out_channels
        self.vctx = nn.Parameter(torch.empty(self.prompt_depth, self.n_vctx, width))
        nn.init.normal_(self.vctx, std=float(init_std))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image):
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
        x = x.permute(1, 0, 2)

        base_len = x.shape[0]
        batch_size = x.shape[1]
        for layer_idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            if layer_idx <= self.prompt_depth:
                x = x[:base_len]
                visual_ctx = self.vctx[layer_idx - 1].to(dtype=x.dtype, device=x.device)
                visual_ctx = visual_ctx.unsqueeze(0).expand(batch_size, -1, -1)
                visual_ctx = visual_ctx.permute(1, 0, 2)
                x = torch.cat([x, visual_ctx], dim=0)

            x = block(x)

        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x
