from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

def build_mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (256,256), activation: str = 'relu', output_activation: str = 'identity') -> nn.Sequential:
    def act(name):
        name = name.lower()
        return {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh, 'elu': nn.ELU}.get(name, None)
    layers=[]; last=input_dim; a=act(activation)
    for h in hidden_dims:
        layers.append(nn.Linear(last, int(h))); layers.append(a() if a else nn.Identity()); last=int(h)
    layers.append(nn.Linear(last, output_dim)); oa=act(output_activation)
    if oa: layers.append(oa())
    return nn.Sequential(*layers)

def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([a, b, b-a, torch.abs(b-a)], dim=-1)
