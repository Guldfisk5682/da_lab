import torch
import torch.nn as nn


class DomainTextVCTXGenerator(nn.Module):
    """Map CLIP domain-text features to a visual prompt residual."""

    def __init__(
        self,
        domain_names,
        domain_text_features,
        prompt_depth,
        n_vctx,
        vision_dim,
        hidden_dim=512,
        gamma_init=0.0,
        gamma_learnable=True,
    ):
        super().__init__()
        self.domain_names = list(domain_names)
        self.domain_to_idx = {name: idx for idx, name in enumerate(self.domain_names)}
        self.prompt_depth = int(prompt_depth)
        self.n_vctx = int(n_vctx)
        self.vision_dim = int(vision_dim)

        input_dim = int(domain_text_features.shape[-1])
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            hidden_dim = max(input_dim, self.vision_dim)

        self.register_buffer("domain_text_features", domain_text_features.float())
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.prompt_depth * self.n_vctx * self.vision_dim),
        )

        gamma = torch.tensor(float(gamma_init), dtype=torch.float32)
        if gamma_learnable:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

    def _indices(self, domain_names=None):
        if domain_names is None:
            return list(range(len(self.domain_names)))
        if isinstance(domain_names, str):
            domain_names = [domain_names]
        return [self.domain_to_idx[name] for name in domain_names]

    def forward(self, domain_names=None):
        indices = self._indices(domain_names)
        index_tensor = torch.tensor(
            indices, device=self.domain_text_features.device, dtype=torch.long
        )
        domain_features = self.domain_text_features.index_select(0, index_tensor)
        residual = self.net(domain_features)
        residual = residual.view(-1, self.prompt_depth, self.n_vctx, self.vision_dim)
        residual = residual.mean(dim=0)
        return self.gamma.float() * residual


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

    def _prompt_tokens(self, layer_idx, dtype, device, vctx_residual=None):
        prompt = self.vctx[layer_idx]
        if vctx_residual is not None:
            prompt = prompt + vctx_residual[layer_idx].to(device=prompt.device)
        return prompt.to(dtype=dtype, device=device)

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
        visual_ctx = self._prompt_tokens(0, x.dtype, x.device, vctx_residual)
        visual_ctx = visual_ctx.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = self._add_prompt_bld(x, visual_ctx)

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        batch_size = x.shape[1]
        for layer_idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            if 1 < layer_idx <= self.prompt_depth:
                x = self._strip_prompt_lbd(x, base_len)
                visual_ctx = self._prompt_tokens(
                    layer_idx - 1, x.dtype, x.device, vctx_residual
                )
                visual_ctx = visual_ctx.unsqueeze(0).expand(batch_size, -1, -1)
                visual_ctx = visual_ctx.permute(1, 0, 2)
                x = self._add_prompt_lbd(x, visual_ctx)

            x = block(x)

        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x
