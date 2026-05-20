import torch
import torch.nn as nn


def compute_patch_stats(patches, eps=1e-6):
    """Compute per-sample patch-token statistics."""
    mu = patches.mean(dim=1, keepdim=True)
    std = patches.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return mu, std


def softmax_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


class DomainStatsBank(nn.Module):
    """EMA bank for domain-level patch-token statistics."""

    def __init__(self, dim, momentum=0.99, eps=1e-6):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mu", torch.zeros(1, 1, dim))
        self.register_buffer("running_std", torch.ones(1, 1, dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update(self, mu_batch, std_batch):
        mu_domain = mu_batch.mean(dim=0, keepdim=True)
        std_domain = std_batch.mean(dim=0, keepdim=True).clamp_min(self.eps)

        if not bool(self.initialized.item()):
            self.running_mu.copy_(mu_domain)
            self.running_std.copy_(std_domain)
            self.initialized.fill_(True)
            return

        keep = self.momentum
        fresh = 1.0 - keep
        self.running_mu.mul_(keep).add_(mu_domain * fresh)
        self.running_std.mul_(keep).add_(std_domain * fresh)

    def get(self):
        return self.running_mu, self.running_std.clamp_min(self.eps)


class ShallowAdaptation(nn.Module):
    """Learnable affine refinement on top of normalize-restore."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, normalized_tokens, ref_mu, ref_std):
        restored = normalized_tokens * ref_std + ref_mu
        scale = self.scale.to(dtype=restored.dtype)
        bias = self.bias.to(dtype=restored.dtype)
        return restored * (1.0 + scale) + bias


class ShallowGate(nn.Module):
    def __init__(self, dim, init_bias=-2.0):
        super().__init__()
        self.norm_ori = nn.LayerNorm(dim)
        self.norm_adp = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, p_ori, p_adapted):
        target_dtype = self.norm_ori.weight.dtype
        if p_ori.dtype != target_dtype:
            p_ori = p_ori.to(dtype=target_dtype)
        if p_adapted.dtype != target_dtype:
            p_adapted = p_adapted.to(dtype=target_dtype)
        x = torch.cat(
            [self.norm_ori(p_ori), self.norm_adp(p_adapted)],
            dim=-1,
        )
        alpha = torch.sigmoid(self.net(x))
        return alpha


class FinalFeatureGate(nn.Module):
    """Late feature gate for final image features."""

    def __init__(self, dim, hidden_ratio=4, init_bias=-4.0):
        super().__init__()
        hidden_dim = max(1, dim // hidden_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, feat_adapted):
        target_dtype = self.norm.weight.dtype
        if feat_adapted.dtype != target_dtype:
            feat_adapted = feat_adapted.to(dtype=target_dtype)
        x = self.norm(feat_adapted)
        x = self.fc1(x)
        x = self.act(x)
        alpha = torch.sigmoid(self.fc2(x))
        return alpha
