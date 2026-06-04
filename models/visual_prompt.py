import torch
import torch.nn as nn


class ShallowVPTVisualEncoder(nn.Module):
    """CLIP ViT visual encoder with shallow/deep visual prompt tokens."""

    def __init__(
        self,
        visual,
        n_vctx=4,
        init_std=0.02,
        prompt_depth=1,
        prompt_position="append",
    ):
        super().__init__()
        if not hasattr(visual, "transformer") or not hasattr(
            visual.transformer, "resblocks"
        ):
            raise TypeError("ShallowVPTVisualEncoder supports CLIP ViT backbones only")

        self.visual = visual
        self.n_vctx = int(n_vctx)
        self.prompt_depth = int(prompt_depth)
        self.prompt_position = str(prompt_position).lower()
        if self.n_vctx <= 0:
            raise ValueError(f"n_vctx must be positive, got {n_vctx}")
        if self.prompt_depth <= 0:
            raise ValueError(f"prompt_depth must be positive, got {prompt_depth}")
        if self.prompt_position not in {"append", "insert"}:
            raise ValueError(
                f"prompt_position must be 'append' or 'insert', got {prompt_position}"
            )
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

    def _add_prompt_bld(self, tokens, prompt):
        if self.prompt_position == "append":
            return torch.cat([tokens, prompt], dim=1)
        return torch.cat([tokens[:, :1], prompt, tokens[:, 1:]], dim=1)

    def _strip_prompt_lbd(self, tokens, base_len):
        if self.prompt_position == "append":
            return tokens[:base_len]
        return torch.cat([tokens[:1], tokens[1 + self.n_vctx:]], dim=0)

    def _add_prompt_lbd(self, tokens, prompt):
        if self.prompt_position == "append":
            return torch.cat([tokens, prompt], dim=0)
        return torch.cat([tokens[:1], prompt, tokens[1:]], dim=0)

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

        base_len = x.shape[1]
        visual_ctx = self.vctx[0].unsqueeze(0).expand(x.shape[0], -1, -1)
        visual_ctx = visual_ctx.to(dtype=x.dtype, device=x.device)
        x = self._add_prompt_bld(x, visual_ctx)

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        batch_size = x.shape[1]
        for layer_idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            if 1 < layer_idx <= self.prompt_depth:
                x = self._strip_prompt_lbd(x, base_len)
                visual_ctx = self.vctx[layer_idx - 1].to(dtype=x.dtype, device=x.device)
                visual_ctx = visual_ctx.unsqueeze(0).expand(batch_size, -1, -1)
                visual_ctx = visual_ctx.permute(1, 0, 2)
                x = self._add_prompt_lbd(x, visual_ctx)

            x = block(x)

        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x
