from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class LowCondActor(nn.Module):
    def __init__(self, obs_dim: int, cond_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        dims = [obs_dim + cond_dim, *hidden_dims, action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.obs_dim = int(obs_dim)
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)

    def forward(self, obs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, cond], dim=-1))


def save_lowcond_actor(path: str | Path, model: LowCondActor, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": model.obs_dim,
            "cond_dim": model.cond_dim,
            "action_dim": model.action_dim,
            "hidden_dims": model.hidden_dims,
            "extra": extra or {},
        },
        path,
    )


def load_lowcond_actor(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[LowCondActor, dict]:
    ckpt = torch.load(path, map_location=map_location)
    model = LowCondActor(
        obs_dim=int(ckpt["obs_dim"]),
        cond_dim=int(ckpt["cond_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dims=tuple(int(x) for x in ckpt.get("hidden_dims", (256, 256, 256))),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, dict(ckpt.get("extra") or {})
