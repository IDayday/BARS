from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from .dataset import OfflineDataset
from .math_utils import safe_l2


FEATURE_COLUMNS = [
    "d_tdr",
    "d_tmd",
    "d_xy",
    "d_state",
    "te_src",
    "te_dst",
    "density_src",
    "density_dst",
    "is_cross_traj",
    "same_traj_forward",
    "abs_time_delta",
    "metric_disagreement",
    "support_score",
]


@dataclass
class FeatureStats:
    medians: Dict[str, float]
    iqrs: Dict[str, float]

    def transform_dict(self, features: Dict[str, np.ndarray], skip_binary: bool = True) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in features.items():
            arr = np.asarray(v, dtype=np.float32)
            if skip_binary and k in {"is_cross_traj", "same_traj_forward"}:
                out[k] = arr
            elif k in self.medians:
                out[k] = (arr - self.medians[k]) / max(self.iqrs[k], 1e-8)
            else:
                out[k] = arr
        return out

    @staticmethod
    def fit(features: Dict[str, np.ndarray]) -> "FeatureStats":
        medians = {}
        iqrs = {}
        for k, v in features.items():
            if k in {"is_cross_traj", "same_traj_forward"}:
                continue
            arr = np.asarray(v, dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                medians[k] = 0.0
                iqrs[k] = 1.0
                continue
            med = float(np.median(finite))
            iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
            medians[k] = med
            iqrs[k] = max(iqr, 1e-8)
        return FeatureStats(medians=medians, iqrs=iqrs)


def _optional_array(dataset: OfflineDataset, key: str, default_shape: tuple[int, ...], fill: float = 0.0) -> np.ndarray:
    if dataset.has(key):
        return np.asarray(dataset.get(key), dtype=np.float32)
    return np.full(default_shape, fill, dtype=np.float32)


def estimate_support_density(x: np.ndarray, k: int = 10) -> np.ndarray:
    from .math_utils import knn_indices

    k = min(max(2, int(k)), len(x))
    _, dist = knn_indices(x, k)
    mean_dist = np.mean(dist[:, 1:] if dist.shape[1] > 1 else dist, axis=1)
    # Larger means denser / better support.
    return 1.0 / (mean_dist + 1e-6)


class EdgeFeatureExtractor:
    """Compute directed pair features for execution calibration and graph costs."""

    def __init__(
        self,
        dataset: OfflineDataset,
        tdr_key: str = "tdr_emb",
        tmd_key: str = "tmd_emb",
        xy_key: str = "xy",
        density_k: int = 10,
    ) -> None:
        self.dataset = dataset
        self.tdr_key = tdr_key
        self.tmd_key = tmd_key
        self.xy_key = xy_key
        self.tdr = dataset.embedding(tdr_key, "states").astype(np.float32)
        self.tmd = dataset.get(tmd_key, self.tdr).astype(np.float32) if dataset.has(tmd_key) else self.tdr
        self.xy = dataset.get(xy_key, dataset.states[:, : min(2, dataset.state_dim)]).astype(np.float32) if dataset.has(xy_key) else dataset.states[:, : min(2, dataset.state_dim)]
        self.states = dataset.states.astype(np.float32)
        self.te = _optional_array(dataset, "te_scores", (dataset.n,), fill=0.0).reshape(-1)
        self.traj_ids = dataset.traj_ids
        self.time_idxs = dataset.time_idxs
        self.density = estimate_support_density(self.tdr, k=density_k).astype(np.float32)

    def pair_features(self, src_idx: np.ndarray, dst_idx: np.ndarray) -> Dict[str, np.ndarray]:
        src_idx = np.asarray(src_idx, dtype=np.int64)
        dst_idx = np.asarray(dst_idx, dtype=np.int64)
        if src_idx.shape != dst_idx.shape:
            raise ValueError("src_idx and dst_idx must have the same shape")

        d_tdr = safe_l2(self.tdr[src_idx], self.tdr[dst_idx]).astype(np.float32)
        d_tmd = safe_l2(self.tmd[src_idx], self.tmd[dst_idx]).astype(np.float32)
        d_xy = safe_l2(self.xy[src_idx], self.xy[dst_idx]).astype(np.float32)
        d_state = safe_l2(self.states[src_idx], self.states[dst_idx]).astype(np.float32)

        same_traj = self.traj_ids[src_idx] == self.traj_ids[dst_idx]
        dt = self.time_idxs[dst_idx].astype(np.float32) - self.time_idxs[src_idx].astype(np.float32)
        same_traj_forward = (same_traj & (dt > 0)).astype(np.float32)
        abs_time_delta = np.where(same_traj, np.abs(dt), np.nan).astype(np.float32)

        # Normalize disagreement by local average to reduce scale dependence.
        metric_disagreement = np.abs(d_tdr - d_tmd) / (0.5 * (d_tdr + d_tmd) + 1e-6)
        support_score = np.minimum(self.density[src_idx], self.density[dst_idx]).astype(np.float32)

        return {
            "d_tdr": d_tdr,
            "d_tmd": d_tmd,
            "d_xy": d_xy,
            "d_state": d_state,
            "te_src": self.te[src_idx].astype(np.float32),
            "te_dst": self.te[dst_idx].astype(np.float32),
            "density_src": self.density[src_idx].astype(np.float32),
            "density_dst": self.density[dst_idx].astype(np.float32),
            "is_cross_traj": (~same_traj).astype(np.float32),
            "same_traj_forward": same_traj_forward,
            "abs_time_delta": abs_time_delta,
            "metric_disagreement": metric_disagreement.astype(np.float32),
            "support_score": support_score,
        }

    @staticmethod
    def to_matrix(features: Dict[str, np.ndarray], columns: Optional[list[str]] = None) -> np.ndarray:
        columns = columns or FEATURE_COLUMNS
        mats = []
        n = None
        for c in columns:
            if c not in features:
                raise KeyError(f"Missing feature {c!r}; available={sorted(features)}")
            arr = np.asarray(features[c], dtype=np.float32).reshape(-1)
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError(f"Feature {c} has length {len(arr)}, expected {n}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            mats.append(arr)
        return np.stack(mats, axis=1).astype(np.float32)
