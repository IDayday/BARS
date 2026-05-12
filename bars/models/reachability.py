from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
from .mlp import build_mlp, pair_features
class ReachabilityModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int] = (256,256)):
        super().__init__(); self.latent_dim=latent_dim; self.net=build_mlp(4*latent_dim, 1, hidden_dims)
    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor: return self.net(pair_features(z_u,z_v)).squeeze(-1)
    @torch.no_grad()
    def prob(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor: return torch.sigmoid(self.forward(z_u,z_v))
