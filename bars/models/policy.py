from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from .mlp import build_mlp
class GoalConditionedPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256,256,256), goal_delta: bool = True):
        super().__init__(); self.obs_dim=obs_dim; self.action_dim=action_dim; self.goal_delta=goal_delta
        in_dim = obs_dim*2 + (obs_dim if goal_delta else 0); self.net = build_mlp(in_dim, action_dim, hidden_dims)
    def _features(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, goal, goal-obs], dim=-1) if self.goal_delta else torch.cat([obs, goal], dim=-1)
    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor: return self.net(self._features(obs,goal))
    @torch.no_grad()
    def act(self, obs_np: np.ndarray, goal_np: np.ndarray, obs_normalizer, action_low: Optional[np.ndarray] = None, action_high: Optional[np.ndarray] = None, device: str = 'cuda') -> np.ndarray:
        self.eval(); obs=torch.as_tensor(obs_normalizer.encode(obs_np[None]), dtype=torch.float32, device=device); goal=torch.as_tensor(obs_normalizer.encode(goal_np[None]), dtype=torch.float32, device=device)
        action=self(obs, goal).cpu().numpy()[0]
        if action_low is not None and action_high is not None: action=np.clip(action, action_low, action_high)
        return action.astype(np.float32)
