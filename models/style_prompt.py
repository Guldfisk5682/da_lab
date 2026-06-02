from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F


def _sanitize_domain_name(domain_name):
    return domain_name.lower().replace("-", "_")


class StyleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, style_gap):
        return self.net(style_gap)


class DomainStyleMLP(StyleMLP):
    """Shared MLP mapping a target-domain style vector to a prompt bias."""


class TargetStyleQueues(nn.Module):
    """Legacy queue module kept only for backward compatibility."""

    def __init__(self, domain_names, style_dim, queue_size):
        super().__init__()
        self.domain_names = list(domain_names)
        self.style_dim = style_dim
        self.queue_size = queue_size
        self.buffer_names = OrderedDict()

        for domain_name in self.domain_names:
            key = _sanitize_domain_name(domain_name)
            self.buffer_names[domain_name] = key
            self.register_buffer(f"{key}_queue", torch.zeros(queue_size, style_dim))
            self.register_buffer(f"{key}_length", torch.zeros(1, dtype=torch.long))
            self.register_buffer(f"{key}_ptr", torch.zeros(1, dtype=torch.long))

    def _queue_buffer(self, domain_name):
        return getattr(self, f"{self.buffer_names[domain_name]}_queue")

    def _length_buffer(self, domain_name):
        return getattr(self, f"{self.buffer_names[domain_name]}_length")

    def _ptr_buffer(self, domain_name):
        return getattr(self, f"{self.buffer_names[domain_name]}_ptr")

    @torch.no_grad()
    def enqueue(self, domain_name, style_vector):
        style_vector = style_vector.detach().float().view(-1)
        if style_vector.numel() != self.style_dim:
            raise ValueError(
                f"Expected style dim {self.style_dim}, got {style_vector.numel()}"
            )

        queue = self._queue_buffer(domain_name)
        length = self._length_buffer(domain_name)
        ptr = self._ptr_buffer(domain_name)

        queue[int(ptr.item())] = style_vector
        ptr[0] = (ptr + 1) % self.queue_size
        length[0] = min(int(length.item()) + 1, self.queue_size)

    def get_valid(self, domain_name):
        queue = self._queue_buffer(domain_name)
        length = int(self._length_buffer(domain_name).item())
        if length <= 0:
            return queue[:0]
        return queue[:length]

    def lengths(self):
        return OrderedDict(
            (domain_name, int(self._length_buffer(domain_name).item()))
            for domain_name in self.domain_names
        )

    def select(self, source_style):
        source_style = source_style.float()
        selected_styles = OrderedDict()
        selected_indices = OrderedDict()
        similarity_scores = OrderedDict()

        source_norm = F.normalize(source_style, dim=-1)

        for domain_name in self.domain_names:
            valid = self.get_valid(domain_name)
            if valid.numel() == 0:
                zeros = torch.zeros_like(source_style)
                selected_styles[domain_name] = zeros
                selected_indices[domain_name] = torch.full(
                    (source_style.shape[0],),
                    -1,
                    device=source_style.device,
                    dtype=torch.long,
                )
                similarity_scores[domain_name] = torch.zeros(
                    source_style.shape[0], 0, device=source_style.device
                )
                continue

            valid = valid.to(source_style.device)
            queue_norm = F.normalize(valid, dim=-1)
            sims = source_norm @ queue_norm.t()
            idx = sims.argmax(dim=1)
            selected_styles[domain_name] = valid[idx]
            selected_indices[domain_name] = idx
            similarity_scores[domain_name] = sims

        return selected_styles, selected_indices, similarity_scores
