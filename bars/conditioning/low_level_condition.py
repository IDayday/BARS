from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol

import numpy as np

from .stats import LowCondStats
from .task_factors import TaskFactorAdapter


class EncoderLike(Protocol):
    def encode(self, obs: np.ndarray) -> np.ndarray:
        ...


ResidualTarget = Literal["task", "local"]


@dataclass
class LowLevelConditionBuilder:
    encoder: EncoderLike
    stats: LowCondStats
    factor_adapter: TaskFactorAdapter
    include_tdr: bool = True
    include_task_factors: bool = True
    explicit_mask: bool = True
    residual_target: ResidualTarget = "task"
    last_info: dict = field(default_factory=dict, init=False)

    @property
    def cond_dim(self) -> int:
        tdr = self.stats.z_dim + 1 if self.include_tdr else 0
        factors = self.stats.padded_factor_dim * (2 if self.explicit_mask else 1) if self.include_task_factors else 0
        return tdr + factors

    def _encode_one(self, obs: np.ndarray) -> np.ndarray:
        batch = np.asarray(obs, dtype=np.float32)[None, :]
        z = np.asarray(self.encoder.encode(batch), dtype=np.float32)
        if z.ndim == 2:
            z = z[0]
        if z.shape != (self.stats.z_dim,):
            raise ValueError(f"encoder returned shape {z.shape}, expected {(self.stats.z_dim,)}")
        return z

    def _tdr_condition(self, obs: np.ndarray, local_target_obs: np.ndarray) -> np.ndarray:
        z_obs = self.stats.normalize_z(self._encode_one(obs))
        z_target = self.stats.normalize_z(self._encode_one(local_target_obs))
        diff = z_target - z_obs
        raw_distance = float(np.linalg.norm(diff))
        if raw_distance <= self.stats.eps:
            direction = np.zeros_like(diff, dtype=np.float32)
        else:
            direction = (diff / raw_distance).astype(np.float32)
        scale = np.asarray([self.stats.distance_scale(raw_distance)], dtype=np.float32)
        return np.concatenate([direction, scale]).astype(np.float32)

    def _factor_condition(
        self,
        obs: np.ndarray,
        local_target_obs: np.ndarray,
        task_goal: np.ndarray | None,
        task_id: int | None,
        goal_info: Mapping | None,
    ) -> np.ndarray:
        if self.residual_target == "local":
            target = local_target_obs
            used_fallback = False
        elif task_goal is None:
            target = local_target_obs
            used_fallback = True
        else:
            target = task_goal
            used_fallback = False
        current = self.factor_adapter.factors(obs, goal_info=goal_info)
        target_factors = self.factor_adapter.target_factors(target, goal_info=goal_info)
        residual = target_factors - current
        residual = self.stats.normalize_residual(residual)
        mask = self.factor_adapter.mask(task_id=task_id, goal_info=goal_info)
        masked = mask.astype(np.float32) * residual
        mask = self.stats.pad_factor(mask)
        masked = self.stats.pad_factor(masked)
        self.last_info["task_goal_fallback_to_local"] = bool(used_fallback)
        if self.explicit_mask:
            return np.concatenate([mask, masked]).astype(np.float32)
        return masked.astype(np.float32)

    def encode(
        self,
        obs: np.ndarray,
        local_target_obs: np.ndarray,
        task_goal: np.ndarray | None = None,
        task_id: int | None = None,
        goal_info: Mapping | None = None,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        self.last_info = {}
        if self.include_tdr:
            parts.append(self._tdr_condition(obs, local_target_obs))
        if self.include_task_factors:
            parts.append(self._factor_condition(obs, local_target_obs, task_goal, task_id, goal_info))
        if not parts:
            cond = np.zeros((0,), dtype=np.float32)
        else:
            cond = np.concatenate(parts).astype(np.float32)
        if cond.shape != (self.cond_dim,):
            raise ValueError(f"condition shape {cond.shape} does not match cond_dim {self.cond_dim}")
        if not np.all(np.isfinite(cond)):
            raise FloatingPointError("low-level condition contains NaN or Inf")
        return cond
