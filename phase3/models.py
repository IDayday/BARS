from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _as_hidden_dims(hidden_dims: Sequence[int] | str | None) -> list[int]:
    if hidden_dims is None:
        return [512, 512, 512]
    if isinstance(hidden_dims, str):
        text = hidden_dims.strip()
        if not text:
            return []
        return [int(x.strip()) for x in text.replace("[", "").replace("]", "").split(",") if x.strip()]
    return [int(x) for x in hidden_dims]


class GCBCMLP(nn.Module):
    """State-based goal-conditioned BC policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] | str | None = None,
        use_remaining_h: bool = True,
        remaining_h_scale: float = 1.0,
        num_edges: int = 0,
        edge_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.use_remaining_h = bool(use_remaining_h)
        self.remaining_h_scale = float(max(1e-6, remaining_h_scale))
        self.edge_embedding_dim = int(edge_embedding_dim)
        self.num_edges = int(num_edges)
        self.edge_embedding = (
            nn.Embedding(self.num_edges, self.edge_embedding_dim)
            if self.num_edges > 0 and self.edge_embedding_dim > 0
            else None
        )

        in_dim = 2 * self.obs_dim
        if self.use_remaining_h:
            in_dim += 1
        if self.edge_embedding is not None:
            in_dim += self.edge_embedding_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for width in _as_hidden_dims(hidden_dims):
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            prev = int(width)
        layers.append(nn.Linear(prev, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        remaining_h: torch.Tensor | None = None,
        edge_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        parts = [obs.float(), goal.float()]
        batch = obs.shape[0]
        if self.use_remaining_h:
            if remaining_h is None:
                remaining = torch.zeros(batch, 1, dtype=obs.dtype, device=obs.device)
            else:
                remaining = remaining_h.to(device=obs.device, dtype=obs.dtype).reshape(batch, 1)
            parts.append(remaining / self.remaining_h_scale)
        if self.edge_embedding is not None:
            if edge_id is None:
                edge_id = torch.zeros(batch, dtype=torch.long, device=obs.device)
            edge = edge_id.to(device=obs.device, dtype=torch.long).clamp(min=0, max=self.num_edges - 1)
            parts.append(self.edge_embedding(edge))
        return self.net(torch.cat(parts, dim=-1))


def action_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target.float()) ** 2)
