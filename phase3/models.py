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
        num_target_sources: int = 0,
        target_source_embedding_dim: int = 0,
        target_source_head_mode: str = "none",
        default_target_source_id: int = 0,
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
        self.num_target_sources = int(num_target_sources)
        self.target_source_embedding_dim = int(target_source_embedding_dim)
        self.target_source_head_mode = str(target_source_head_mode or "none")
        self.default_target_source_id = int(default_target_source_id)
        if self.target_source_head_mode not in {"none", "heads"}:
            raise ValueError("target_source_head_mode must be 'none' or 'heads'")
        self.source_embedding = (
            nn.Embedding(self.num_target_sources, self.target_source_embedding_dim)
            if self.num_target_sources > 0 and self.target_source_embedding_dim > 0
            else None
        )

        in_dim = 2 * self.obs_dim
        if self.use_remaining_h:
            in_dim += 1
        if self.edge_embedding is not None:
            in_dim += self.edge_embedding_dim
        if self.source_embedding is not None:
            in_dim += self.target_source_embedding_dim

        hidden = _as_hidden_dims(hidden_dims)
        if self.target_source_head_mode == "heads":
            if self.num_target_sources <= 0:
                raise ValueError("num_target_sources must be positive when target_source_head_mode='heads'")
            trunk_layers: list[nn.Module] = []
            prev = in_dim
            for width in hidden:
                trunk_layers.append(nn.Linear(prev, int(width)))
                trunk_layers.append(nn.ReLU())
                prev = int(width)
            self.trunk = nn.Sequential(*trunk_layers)
            self.source_heads = nn.ModuleList([nn.Linear(prev, self.action_dim) for _ in range(self.num_target_sources)])
            self.net = None
        else:
            layers: list[nn.Module] = []
            prev = in_dim
            for width in hidden:
                layers.append(nn.Linear(prev, int(width)))
                layers.append(nn.ReLU())
                prev = int(width)
            layers.append(nn.Linear(prev, self.action_dim))
            self.net = nn.Sequential(*layers)
            self.trunk = None
            self.source_heads = None

    def forward(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        remaining_h: torch.Tensor | None = None,
        edge_id: torch.Tensor | None = None,
        target_source_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        parts = [obs.float(), goal.float()]
        batch = obs.shape[0]
        if target_source_id is None:
            source = torch.full(
                (batch,),
                int(self.default_target_source_id),
                dtype=torch.long,
                device=obs.device,
            )
        else:
            source = target_source_id.to(device=obs.device, dtype=torch.long).reshape(batch)
        if self.num_target_sources > 0:
            source = source.clamp(min=0, max=self.num_target_sources - 1)
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
        if self.source_embedding is not None:
            parts.append(self.source_embedding(source))
        features = torch.cat(parts, dim=-1)
        if self.target_source_head_mode == "heads":
            trunk = self.trunk(features) if self.trunk is not None else features
            all_outputs = torch.stack([head(trunk) for head in self.source_heads], dim=1)
            gather_index = source.reshape(batch, 1, 1).expand(batch, 1, self.action_dim)
            return torch.gather(all_outputs, dim=1, index=gather_index).squeeze(1)
        if self.net is None:
            raise RuntimeError("GCBCMLP is missing network layers")
        return self.net(features)


def action_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target.float()) ** 2)
