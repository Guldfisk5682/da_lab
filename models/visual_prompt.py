import math

import torch
import torch.nn as nn


class InstanceVCTXGenerator(nn.Module):
    """Generate instance-specific VCTX residuals from early patch tokens."""

    def __init__(
        self,
        prompt_depth,
        n_vctx,
        vision_dim,
        hidden_dim=512,
        beta_init=0.0,
        beta_learnable=True,
        log_std_min=-5.0,
        log_std_max=2.0,
        fixed_eval_seed=0,
        mode="residual",
    ):
        super().__init__()
        self.prompt_depth = int(prompt_depth)
        self.n_vctx = int(n_vctx)
        self.vision_dim = int(vision_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.mode = str(mode).lower()
        if self.mode not in {"residual", "append"}:
            raise ValueError(f"Unsupported instance mode: {mode}")

        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            hidden_dim = max(256, self.vision_dim // 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.vision_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        output_dim = self.prompt_depth * self.n_vctx * self.vision_dim
        self.to_stats = nn.Linear(hidden_dim, output_dim * 2)

        generator = torch.Generator().manual_seed(int(fixed_eval_seed))
        fixed_eps = torch.randn(
            self.prompt_depth,
            self.n_vctx,
            self.vision_dim,
            generator=generator,
        )
        self.register_buffer("fixed_eval_eps", fixed_eps)

        beta = torch.tensor(float(beta_init), dtype=torch.float32)
        if beta_learnable:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _tokens_to_map(self, patch_tokens):
        batch_size, num_tokens, dim = patch_tokens.shape
        if dim != self.vision_dim:
            raise ValueError(f"Expected patch dim {self.vision_dim}, got {dim}")
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError("Patch tokens must form a square grid.")
        return patch_tokens.transpose(1, 2).reshape(batch_size, dim, side, side)

    def forward(self, patch_tokens):
        token_map = self._tokens_to_map(patch_tokens.float())
        features = self.encoder(token_map).flatten(1)
        mean, log_std = self.to_stats(features).chunk(2, dim=-1)
        std = log_std.clamp(min=self.log_std_min, max=self.log_std_max).exp()

        mean = mean.view(-1, self.prompt_depth, self.n_vctx, self.vision_dim)
        std = std.view(-1, self.prompt_depth, self.n_vctx, self.vision_dim)

        if self.training:
            eps = torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
            )
        else:
            eps = self.fixed_eval_eps.to(device=mean.device, dtype=mean.dtype)
            eps = eps.unsqueeze(0).expand_as(mean)

        residual = mean + eps * std
        return self.beta.float() * residual, mean, std


class ShallowVPTVisualEncoder(nn.Module):
    """CLIP ViT visual encoder with persistent visual prompt tokens."""

    def __init__(
        self,
        visual,
        n_vctx=4,
        init_std=0.02,
        prompt_depth=1,
    ):
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

    def _add_prompt_bld(self, tokens, prompt):
        return torch.cat([tokens, prompt], dim=1)

    def _strip_prompt_lbd(self, tokens, base_len, prompt_len):
        del prompt_len
        return tokens[:base_len]

    def _add_prompt_lbd(self, tokens, prompt):
        return torch.cat([tokens, prompt], dim=0)

    def _split_instance_prompt(self, vctx_residual):
        if isinstance(vctx_residual, dict):
            mode = vctx_residual.get("mode", "residual")
            residual = vctx_residual.get("tokens")
        else:
            mode = "residual"
            residual = vctx_residual
        return mode, residual

    def _prompt_tokens(self, layer_idx, batch_size, dtype, device, vctx_residual=None):
        mode, residual = self._split_instance_prompt(vctx_residual)
        prompt = self.vctx[layer_idx]
        append_prompt = None

        if residual is not None:
            if residual.dim() == 3:
                residual_i = residual[layer_idx].to(device=prompt.device)
            elif residual.dim() == 4:
                residual_i = residual[:, layer_idx].to(device=prompt.device)
            else:
                raise ValueError("instance tokens must have shape [D,N,C] or [B,D,N,C]")

            if mode == "residual":
                if residual_i.dim() == 2:
                    prompt = prompt + residual_i
                else:
                    prompt = prompt.unsqueeze(0) + residual_i
            elif mode == "append":
                append_prompt = residual_i
            else:
                raise ValueError(f"Unsupported instance mode: {mode}")

        prompt = prompt.to(dtype=dtype, device=device)
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)

        if append_prompt is not None:
            append_prompt = append_prompt.to(dtype=dtype, device=device)
            if append_prompt.dim() == 2:
                append_prompt = append_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            prompt = torch.cat([prompt, append_prompt], dim=1)

        return prompt

    def extract_early_patch_tokens(self, image):
        x = self.visual.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = self.visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        return x[:, 1:]

    def forward(self, image, vctx_residual=None):
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
        batch_size = x.shape[0]
        visual_ctx = self._prompt_tokens(
            0, batch_size, x.dtype, x.device, vctx_residual
        )
        x = self._add_prompt_bld(x, visual_ctx)

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        for layer_idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            if 1 < layer_idx <= self.prompt_depth:
                prompt_len = x.shape[0] - base_len
                x = self._strip_prompt_lbd(x, base_len, prompt_len)
                visual_ctx = self._prompt_tokens(
                    layer_idx - 1, batch_size, x.dtype, x.device, vctx_residual
                )
                visual_ctx = visual_ctx.permute(1, 0, 2)
                x = self._add_prompt_lbd(x, visual_ctx)

            x = block(x)

        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x
