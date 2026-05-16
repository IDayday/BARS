from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .normalization import Normalizer
@dataclass
class TrajectorySlice:
    traj_id: int; start: int; end: int; raw_start: int; raw_end: int
class OfflineDataset:
    def __init__(self, observations: np.ndarray, actions: np.ndarray, next_observations: np.ndarray, traj_id: np.ndarray, timestep: np.ndarray, traj_slices: List[TrajectorySlice], env_name: str = 'unknown'):
        self.observations = observations.astype(np.float32); self.actions = actions.astype(np.float32); self.next_observations = next_observations.astype(np.float32)
        self.traj_id = traj_id.astype(np.int32); self.timestep = timestep.astype(np.int32); self.traj_slices = traj_slices; self.env_name = env_name
        self.obs_normalizer = Normalizer.fit(self.observations); self.action_normalizer = Normalizer.fit(self.actions); self._traj_to_indices: Optional[Dict[int, np.ndarray]] = None; self._valid_slice_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    @property
    def size(self) -> int: return int(self.observations.shape[0])
    @property
    def obs_dim(self) -> int: return int(self.observations.shape[1])
    @property
    def action_dim(self) -> int: return int(self.actions.shape[1])
    @property
    def num_trajectories(self) -> int: return len(self.traj_slices)
    def traj_to_indices(self) -> Dict[int, np.ndarray]:
        if self._traj_to_indices is None:
            self._traj_to_indices = {sl.traj_id: np.arange(sl.start, sl.end, dtype=np.int64) for sl in self.traj_slices if sl.end > sl.start}
        return self._traj_to_indices
    def sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.size, size=batch_size, endpoint=False)
    def sample_future_pairs(self, batch_size: int, horizon: int, rng: np.random.Generator, min_dt: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty
        key = int(min_dt)
        cached = self._valid_slice_cache.get(key)
        if cached is None:
            starts = np.asarray([sl.start for sl in self.traj_slices if sl.end - sl.start > min_dt], dtype=np.int64)
            ends = np.asarray([sl.end for sl in self.traj_slices if sl.end - sl.start > min_dt], dtype=np.int64)
            cached = (starts, ends)
            self._valid_slice_cache[key] = cached
        starts, ends = cached
        if len(starts) == 0: raise RuntimeError('Dataset has no trajectory longer than min_dt.')
        choice = rng.integers(0, len(starts), size=batch_size, endpoint=False)
        sl_start = starts[choice]
        sl_end = ends[choice]
        i_high = np.maximum(sl_start + 1, sl_end - min_dt)
        i_out = rng.integers(sl_start, i_high)
        max_dt = np.minimum(int(horizon), sl_end - i_out - 1)
        dt_high = np.maximum(max_dt + 1, min_dt + 1)
        dt_out = rng.integers(int(min_dt), dt_high).astype(np.int64)
        j_out = i_out + dt_out
        return i_out.astype(np.int64), j_out.astype(np.int64), dt_out
    def get_future_index(self, i: int, dt: int) -> Optional[int]:
        j = i + dt
        return j if j < self.size and self.traj_id[i] == self.traj_id[j] else None
