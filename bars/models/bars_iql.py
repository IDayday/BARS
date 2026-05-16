from __future__ import annotations

"""BARS-native goal-conditioned low-level policy.

This module is deliberately independent from HIQL.  It implements a compact
IQL/AWR-style goal-conditioned low-level learner whose training goal distribution
can be matched to the BARS graph distribution.  The actor API is compatible with
existing BARS rollout code: ``act(obs, goal, obs_normalizer, ...)`` returns a
single numpy action.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .mlp import build_mlp


def gc_features(obs: torch.Tensor, goal: torch.Tensor, goal_delta: bool = True) -> torch.Tensor:
    return torch.cat([obs, goal, goal - obs], dim=-1) if goal_delta else torch.cat([obs, goal], dim=-1)


class DeterministicGCActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256, 256), goal_delta: bool = True):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_delta = bool(goal_delta)
        in_dim = self.obs_dim * (3 if self.goal_delta else 2)
        self.net = build_mlp(in_dim, self.action_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(gc_features(obs, goal, self.goal_delta))


class GCValue(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int] = (256, 256), goal_delta: bool = True):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.goal_delta = bool(goal_delta)
        in_dim = self.obs_dim * (3 if self.goal_delta else 2)
        self.net = build_mlp(in_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(gc_features(obs, goal, self.goal_delta)).squeeze(-1)


class GCQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256), goal_delta: bool = True):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_delta = bool(goal_delta)
        in_dim = self.obs_dim * (3 if self.goal_delta else 2) + self.action_dim
        self.net = build_mlp(in_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([gc_features(obs, goal, self.goal_delta), action], dim=-1)).squeeze(-1)


class BARSIQLPolicy(nn.Module):
    """Goal-conditioned BARS-low policy with value heads for diagnostics.

    The value heads are not used by the existing rollout loop directly, but they
    let policy-conditioned reachability training use the same low-level policy
    signal that will later execute graph edges.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        goal_delta: bool = True,
        value_temperature: float = 5.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_delta = bool(goal_delta)
        self.value_temperature = float(value_temperature)
        self.actor = DeterministicGCActor(obs_dim, action_dim, actor_hidden_dims, goal_delta)
        self.q1 = GCQ(obs_dim, action_dim, value_hidden_dims, goal_delta)
        self.q2 = GCQ(obs_dim, action_dim, value_hidden_dims, goal_delta)
        self.v = GCValue(obs_dim, value_hidden_dims, goal_delta)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.actor(obs, goal)

    def q_min(self, obs: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.minimum(self.q1(obs, goal, action), self.q2(obs, goal, action))

    @torch.no_grad()
    def value_np(self, obs_np: np.ndarray, goal_np: np.ndarray, obs_normalizer, device: str = "cuda") -> float:
        self.eval()
        obs = torch.as_tensor(obs_normalizer.encode(np.asarray(obs_np, dtype=np.float32)[None]), dtype=torch.float32, device=device)
        goal = torch.as_tensor(obs_normalizer.encode(np.asarray(goal_np, dtype=np.float32)[None]), dtype=torch.float32, device=device)
        return float(self.v(obs, goal).detach().cpu().numpy()[0])

    @torch.no_grad()
    def reachability_prob_batch(
        self,
        obs_np: np.ndarray,
        goal_np: np.ndarray,
        obs_normalizer,
        device: str = "cuda",
        batch_size: int = 65536,
        return_tensor: bool = False,
    ):
        self.eval()
        obs_arr = obs_normalizer.encode(np.asarray(obs_np, dtype=np.float32))
        goal_arr = obs_normalizer.encode(np.asarray(goal_np, dtype=np.float32))
        n = int(obs_arr.shape[0])
        if n == 0:
            empty = torch.empty(0, dtype=torch.float32, device=device)
            return empty if return_tensor else empty.cpu().numpy()
        chunks = []
        bs = max(1, int(batch_size))
        temp = max(self.value_temperature, 1e-6)
        for start in range(0, n, bs):
            end = min(n, start + bs)
            obs = torch.as_tensor(obs_arr[start:end], dtype=torch.float32, device=device)
            goal = torch.as_tensor(goal_arr[start:end], dtype=torch.float32, device=device)
            chunks.append(torch.sigmoid(self.v(obs, goal) / temp))
        out = torch.cat(chunks, dim=0)
        return out if return_tensor else out.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def reachability_prob(self, obs_np: np.ndarray, goal_np: np.ndarray, obs_normalizer, device: str = "cuda") -> float:
        # Under sparse -1/0 rewards, larger V roughly means shorter/easier edge.
        val = self.value_np(obs_np, goal_np, obs_normalizer, device=device)
        return float(1.0 / (1.0 + np.exp(-val / max(self.value_temperature, 1e-6))))

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        goal_np: np.ndarray,
        obs_normalizer,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        device: str = "cuda",
    ) -> np.ndarray:
        self.eval()
        obs = torch.as_tensor(obs_normalizer.encode(np.asarray(obs_np, dtype=np.float32)[None]), dtype=torch.float32, device=device)
        goal = torch.as_tensor(obs_normalizer.encode(np.asarray(goal_np, dtype=np.float32)[None]), dtype=torch.float32, device=device)
        action = self.actor(obs, goal).detach().cpu().numpy()[0]
        if action_low is not None and action_high is not None:
            action = np.clip(action, action_low, action_high)
        return action.astype(np.float32)
