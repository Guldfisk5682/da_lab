import torch
import torch.nn as nn


class VisualEncoderAdapter(nn.Module):
    """Expose shallow-token forward helpers on a CLIP ViT visual encoder."""

    def __init__(self, visual):
        super().__init__()
        if not hasattr(visual, "transformer") or not hasattr(
            visual.transformer, "resblocks"
        ):
            raise TypeError("This helper currently supports CLIP ViT backbones only")

        self.visual = visual
        self.num_layers = len(self.visual.transformer.resblocks)
        self.output_dim = self.visual.output_dim
        self.hidden_dim = self.visual.conv1.out_channels
        self.dtype = self.visual.conv1.weight.dtype

    def patch_embed(self, image):
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
        return x

    def tokens_forward(self, tokens, start_layer=1, end_layer=None):
        if end_layer is None:
            end_layer = self.num_layers

        x = tokens.permute(1, 0, 2)
        for layer_idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            if layer_idx < start_layer:
                continue
            if layer_idx > end_layer:
                break
            x = block(x)
        return x.permute(1, 0, 2)

    def forward_until(self, image, layer_idx):
        tokens = self.patch_embed(image)
        if layer_idx <= 0:
            return tokens
        return self.tokens_forward(tokens, start_layer=1, end_layer=layer_idx)

    def forward_from(self, hidden_tokens, start_layer):
        tokens = self.tokens_forward(hidden_tokens, start_layer=start_layer)
        features = self.visual.ln_post(tokens[:, 0, :])
        if self.visual.proj is not None:
            features = features @ self.visual.proj
        return features

    def forward(self, image):
        hidden = self.patch_embed(image)
        return self.forward_from(hidden, start_layer=1)


def patch_tokens(hidden_tokens):
    return hidden_tokens[:, 1:, :]


def compute_patch_style(patch_hidden, eps=1e-6):
    patch_hidden = patch_hidden.float()
    mu = patch_hidden.mean(dim=1)
    std = patch_hidden.std(dim=1, unbiased=False) + eps
    return torch.cat([mu, std], dim=1)

