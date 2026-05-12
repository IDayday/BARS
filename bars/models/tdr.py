from __future__ import annotations
from typing import Sequence, Tuple
import torch
import torch.nn as nn
from .mlp import build_mlp, pair_features
class TemporalDistanceModel(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 32, hidden_dims: Sequence[int] = (256,256), pair_hidden_dims: Sequence[int] = (256,256)):
        super().__init__(); self.obs_dim=obs_dim; self.latent_dim=latent_dim
        self.encoder = build_mlp(obs_dim, latent_dim, hidden_dims)
        self.pair_head = build_mlp(4*latent_dim, 2, pair_hidden_dims)
    def encode(self, obs: torch.Tensor) -> torch.Tensor: return self.encoder(obs)
    def forward(self, obs_a: torch.Tensor, obs_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        za=self.encode(obs_a); zb=self.encode(obs_b); out=self.pair_head(pair_features(za,zb)); return za,zb,out[:,0],out[:,1]
