from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class TaskFactorAdapter:
    factor_dim: int

    def factors(self, obs: np.ndarray, goal_info: Mapping | None = None) -> np.ndarray:
        raise NotImplementedError

    def target_factors(self, target: np.ndarray, goal_info: Mapping | None = None) -> np.ndarray:
        return self.factors(target, goal_info=goal_info)

    def mask(self, task_id: int | None = None, goal_info: Mapping | None = None) -> np.ndarray:
        return np.ones((self.factor_dim,), dtype=np.float32)


@dataclass(frozen=True)
class MazeXYFactorAdapter(TaskFactorAdapter):
    xy_indices: tuple[int, int] = (0, 1)
    factor_dim: int = 2

    def factors(self, obs: np.ndarray, goal_info: Mapping | None = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        return obs[list(self.xy_indices)].astype(np.float32)


@dataclass(frozen=True)
class HumanoidMazeXYFactorAdapter(MazeXYFactorAdapter):
    xy_indices: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class ObjectFactorAdapter(TaskFactorAdapter):
    factor_indices: tuple[int, ...]
    task_masks: Mapping[int, Sequence[float]] | None = None

    @property
    def factor_dim(self) -> int:
        return len(self.factor_indices)

    def factors(self, obs: np.ndarray, goal_info: Mapping | None = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        return obs[list(self.factor_indices)].astype(np.float32)

    def mask(self, task_id: int | None = None, goal_info: Mapping | None = None) -> np.ndarray:
        if goal_info is not None and "task_factor_mask" in goal_info:
            mask = np.asarray(goal_info["task_factor_mask"], dtype=np.float32)
        elif task_id is not None and self.task_masks is not None and int(task_id) in self.task_masks:
            mask = np.asarray(self.task_masks[int(task_id)], dtype=np.float32)
        else:
            mask = np.ones((self.factor_dim,), dtype=np.float32)
        if mask.shape != (self.factor_dim,):
            raise ValueError(f"mask shape {mask.shape} does not match factor_dim {self.factor_dim}")
        return mask.astype(np.float32)
