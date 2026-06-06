from collections import OrderedDict

import torch
import torch.nn as nn


class CLIPVisualWithHidden(nn.Module):
    """Frozen CLIP ViT visual forward that also returns all layer hidden states."""

    def __init__(self, visual):
        super().__init__()
        if not hasattr(visual, "transformer") or not hasattr(
            visual.transformer, "resblocks"
        ):
            raise TypeError("CLIPVisualWithHidden supports CLIP ViT backbones only")
        self.visual = visual

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @property
    def width(self):
        return self.visual.conv1.out_channels

    @property
    def depth(self):
        return len(self.visual.transformer.resblocks)

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
        hidden_states = []
        for block in self.visual.transformer.resblocks:
            x = block(x)
            hidden_states.append(x.permute(1, 0, 2).detach())

        x = x.permute(1, 0, 2)
        image_features = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            image_features = image_features @ self.visual.proj

        return image_features, hidden_states


class MultiLayerStyleProjector(nn.Module):
    """Project per-layer CLIP hidden-state style statistics to text tokens."""

    def __init__(self, visual_dim, text_dim, depth, hidden_dim=512, eps=1e-6):
        super().__init__()
        self.depth = int(depth)
        self.eps = float(eps)
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("linear1", nn.Linear(visual_dim * 2, hidden_dim)),
                            ("gelu", nn.GELU()),
                            ("linear2", nn.Linear(hidden_dim, text_dim)),
                        ]
                    )
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, hidden_states):
        if len(hidden_states) != self.depth:
            raise ValueError(
                f"Expected {self.depth} hidden states, got {len(hidden_states)}"
            )

        style_tokens = []
        for hidden, projector in zip(hidden_states, self.projectors):
            hidden = hidden.float()
            mu = hidden.mean(dim=1)
            std = hidden.std(dim=1, unbiased=False).clamp_min(self.eps)
            stats = torch.cat([mu, std], dim=-1)
            style_tokens.append(projector(stats))

        return torch.stack(style_tokens, dim=1)


class MultiLayerImageProjector(nn.Module):
    """Project per-layer pooled visual content into text prompt tokens."""

    def __init__(self, visual_dim, text_dim, depth):
        super().__init__()
        self.depth = int(depth)
        self.projectors = nn.ModuleList(
            [nn.Linear(visual_dim, text_dim) for _ in range(self.depth)]
        )

    def forward(self, hidden_states):
        if len(hidden_states) != self.depth:
            raise ValueError(
                f"Expected {self.depth} hidden states, got {len(hidden_states)}"
            )

        image_tokens = []
        for hidden, projector in zip(hidden_states, self.projectors):
            pooled = hidden.float().mean(dim=1)
            image_tokens.append(projector(pooled))

        return torch.stack(image_tokens, dim=1)
