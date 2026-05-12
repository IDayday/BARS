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
        self.obs_normalizer = Normalizer.fit(self.observations); self.action_normalizer = Normalizer.fit(self.actions); self._traj_to_indices: Optional[Dict[int, np.ndarray]] = None
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
        valid_slices = [sl for sl in self.traj_slices if sl.end - sl.start > min_dt]
        if not valid_slices: raise RuntimeError('Dataset has no trajectory longer than min_dt.')
        i_out = np.empty(batch_size, dtype=np.int64); j_out = np.empty(batch_size, dtype=np.int64); dt_out = np.empty(batch_size, dtype=np.int64)
        for b in range(batch_size):
            sl = valid_slices[int(rng.integers(0, len(valid_slices)))]
            i = int(rng.integers(sl.start, max(sl.start + 1, sl.end - min_dt)))
            max_dt = min(horizon, sl.end - i - 1)
            dt = min_dt if max_dt < min_dt else int(rng.integers(min_dt, max_dt + 1))
            i_out[b] = i; j_out[b] = i + dt; dt_out[b] = dt
        return i_out, j_out, dt_out
    def get_future_index(self, i: int, dt: int) -> Optional[int]:
        j = i + dt
        return j if j < self.size and self.traj_id[i] == self.traj_id[j] else None
