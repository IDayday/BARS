from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


DistanceFn = Callable[[object, object], float]


@dataclass
class ProgressSnapshot:
    prev_distance: float | None
    current_distance: float
    step_progress: float
    progress_window_value: float
    stalled: bool


def as_path_array(path: Iterable[object] | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(path)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def distance_to_path(current_state, path, distance_fn: DistanceFn) -> float | None:
    path_arr = as_path_array(path)
    if path_arr is None:
        return None
    return min(float(distance_fn(current_state, node)) for node in path_arr)


class ProgressMonitor:
    """Tracks target-distance progress over a rolling window."""

    def __init__(self, distance_fn: DistanceFn, window: int, progress_eps: float):
        self.distance_fn = distance_fn
        self.window = max(1, int(window))
        self.progress_eps = float(progress_eps)
        self.distances: deque[float] = deque(maxlen=self.window + 1)
        self.last_snapshot: ProgressSnapshot | None = None

    def reset(self) -> None:
        self.distances.clear()
        self.last_snapshot = None

    def update(self, prev_state, current_state, target) -> ProgressSnapshot:
        prev_distance = float(self.distance_fn(prev_state, target))
        current_distance = float(self.distance_fn(current_state, target))
        self.distances.append(current_distance)
        if len(self.distances) >= self.window + 1:
            progress_window_value = float(self.distances[0] - self.distances[-1])
        else:
            progress_window_value = 0.0
        stalled = len(self.distances) >= self.window + 1 and progress_window_value < self.progress_eps
        snapshot = ProgressSnapshot(
            prev_distance=prev_distance,
            current_distance=current_distance,
            step_progress=prev_distance - current_distance,
            progress_window_value=progress_window_value,
            stalled=stalled,
        )
        self.last_snapshot = snapshot
        return snapshot

    @property
    def is_stalled(self) -> bool:
        return bool(self.last_snapshot and self.last_snapshot.stalled)

    @property
    def progress_window_value(self) -> float:
        if self.last_snapshot is None:
            return 0.0
        return float(self.last_snapshot.progress_window_value)
