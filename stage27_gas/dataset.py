from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class OfflineDataset:
    """Flat trajectory dataset used by Stage27 graph/planner code.

    Required arrays:
        states:  [N, state_dim]
        traj_ids: [N]
        time_idxs: [N]

    Optional arrays include tdr_emb, tmd_emb, xy, te_scores, actions, rewards,
    terminals, goals, or any project-specific side channel. All arrays must have
    leading dimension N unless scalar metadata.
    """

    states: np.ndarray
    traj_ids: np.ndarray
    time_idxs: np.ndarray
    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        self.traj_ids = np.asarray(self.traj_ids)
        self.time_idxs = np.asarray(self.time_idxs)
        if self.states.ndim != 2:
            raise ValueError(f"states must be [N,D], got {self.states.shape}")
        n = len(self.states)
        if len(self.traj_ids) != n or len(self.time_idxs) != n:
            raise ValueError("states, traj_ids and time_idxs must have the same leading dimension")
        normalized = {}
        for k, v in self.extras.items():
            arr = np.asarray(v)
            if arr.ndim > 0 and arr.shape[0] not in (n,):
                # Allow scalar or metadata-like arrays only when they are not row aligned.
                raise ValueError(f"extras[{k!r}] has incompatible shape {arr.shape}; expected leading {n}")
            normalized[k] = arr
        self.extras = normalized

    @property
    def n(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    def has(self, key: str) -> bool:
        return key == "states" or key in self.extras

    def get(self, key: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        if key == "states":
            return self.states
        if key == "traj_ids":
            return self.traj_ids
        if key == "time_idxs":
            return self.time_idxs
        if key in self.extras:
            return self.extras[key]
        if default is not None:
            return default
        raise KeyError(f"Dataset does not contain array {key!r}; available={self.keys()}")

    def keys(self) -> list[str]:
        return ["states", "traj_ids", "time_idxs", *sorted(self.extras.keys())]

    def embedding(self, preferred: str = "tdr_emb", fallback: str = "states") -> np.ndarray:
        if self.has(preferred):
            return np.asarray(self.get(preferred), dtype=np.float32)
        if self.has(fallback):
            return np.asarray(self.get(fallback), dtype=np.float32)
        return self.states

    def trajectory_slices(self) -> dict[Any, np.ndarray]:
        """Return row indices per trajectory sorted by time index."""
        out: dict[Any, np.ndarray] = {}
        for tid in np.unique(self.traj_ids):
            idx = np.flatnonzero(self.traj_ids == tid)
            idx = idx[np.argsort(self.time_idxs[idx], kind="mergesort")]
            out[tid.item() if hasattr(tid, "item") else tid] = idx
        return out

    def subset(self, indices: Iterable[int]) -> "OfflineDataset":
        idx = np.asarray(list(indices), dtype=np.int64)
        extras = {k: v[idx] for k, v in self.extras.items() if np.asarray(v).ndim > 0 and len(v) == self.n}
        return OfflineDataset(self.states[idx], self.traj_ids[idx], self.time_idxs[idx], extras)


def _infer_traj_ids_from_terminals(n: int, terminals: Optional[np.ndarray], timeouts: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    traj_ids = np.zeros(n, dtype=np.int64)
    time_idxs = np.zeros(n, dtype=np.int64)
    boundaries = np.zeros(n, dtype=bool)
    if terminals is not None:
        boundaries |= np.asarray(terminals).astype(bool)
    if timeouts is not None:
        boundaries |= np.asarray(timeouts).astype(bool)

    current_tid = 0
    current_t = 0
    for i in range(n):
        traj_ids[i] = current_tid
        time_idxs[i] = current_t
        if boundaries[i] and i < n - 1:
            current_tid += 1
            current_t = 0
        else:
            current_t += 1
    return traj_ids, time_idxs


def load_offline_dataset_npz(path: str | Path) -> OfflineDataset:
    """Load a flexible .npz dataset.

    Accepted formats:
    1. Stage27 flat format: states, traj_ids, time_idxs.
    2. D4RL/OGBench-ish format: observations plus terminals/timeouts.

    Recommended optional keys:
        tdr_emb, tmd_emb, xy, te_scores, actions, rewards, goals.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    if "states" in keys:
        states = np.asarray(data["states"], dtype=np.float32)
    elif "observations" in keys:
        states = np.asarray(data["observations"], dtype=np.float32)
    else:
        raise KeyError(f"{path} must contain 'states' or 'observations'; keys={sorted(keys)}")

    n = states.shape[0]
    if "traj_ids" in keys and "time_idxs" in keys:
        traj_ids = np.asarray(data["traj_ids"])
        time_idxs = np.asarray(data["time_idxs"])
    else:
        terminals = data["terminals"] if "terminals" in keys else data["dones"] if "dones" in keys else None
        timeouts = data["timeouts"] if "timeouts" in keys else None
        traj_ids, time_idxs = _infer_traj_ids_from_terminals(n, terminals, timeouts)

    reserved = {"states", "observations", "traj_ids", "time_idxs"}
    extras: Dict[str, np.ndarray] = {}
    for k in data.files:
        if k in reserved:
            continue
        arr = np.asarray(data[k])
        if arr.ndim > 0 and arr.shape[0] == n:
            extras[k] = arr
    return OfflineDataset(states=states, traj_ids=traj_ids, time_idxs=time_idxs, extras=extras)


def save_offline_dataset_npz(dataset: OfflineDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=dataset.states,
        traj_ids=dataset.traj_ids,
        time_idxs=dataset.time_idxs,
        **dataset.extras,
    )
