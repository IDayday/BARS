from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cage.config import CAGEConfig
from cage.monitor import DistanceFn, as_path_array


@dataclass(frozen=True)
class RecoverySelection:
    target: np.ndarray
    index: int
    distance: float
    score: float


class RecoverySelector:
    """Chooses a nearby suffix node before asking for global replanning."""

    def __init__(self, config: CAGEConfig, distance_fn: DistanceFn):
        self.config = config
        self.distance_fn = distance_fn

    def select(self, current_state, current_path, current_index: int | None = None) -> RecoverySelection | None:
        path = as_path_array(current_path)
        if path is None:
            return None
        if current_index is None or current_index < 0:
            distances = [self.distance_fn(current_state, node) for node in path]
            current_index = int(np.argmin(distances))

        start_idx = max(0, int(current_index) - 2)
        max_dist = max(float(self.config.max_subgoal_dist), float(self.config.drift_threshold))
        best: RecoverySelection | None = None
        for idx in range(start_idx, len(path)):
            node = path[idx]
            distance = float(self.distance_fn(current_state, node))
            if distance > max_dist:
                continue
            backward_penalty = max(0, int(current_index) - idx)
            score = distance + float(self.config.recovery_suffix_weight) * backward_penalty
            candidate = RecoverySelection(target=np.asarray(node), index=idx, distance=distance, score=score)
            if best is None or candidate.score < best.score:
                best = candidate
        return best
