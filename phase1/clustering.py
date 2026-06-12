from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs.
    MiniBatchKMeans = None
    StandardScaler = None


def _flatten_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim == 1:
        return observations.reshape(-1, 1)
    return observations.reshape(observations.shape[0], -1)


def _select_state_dims(observations: np.ndarray, state_dims: list[int] | None) -> np.ndarray:
    features = _flatten_observations(observations)
    if state_dims is None:
        return features
    return features[:, np.asarray(state_dims, dtype=np.int64)]


def _grid_bins(n_clusters: int, n_bins_x: int | None, n_bins_y: int | None) -> tuple[int, int]:
    if n_bins_x is not None and n_bins_y is not None:
        return int(n_bins_x), int(n_bins_y)
    if n_bins_x is not None:
        return int(n_bins_x), int(np.ceil(n_clusters / int(n_bins_x)))
    if n_bins_y is not None:
        return int(np.ceil(n_clusters / int(n_bins_y))), int(n_bins_y)
    n_bins_x = max(1, int(round(np.sqrt(n_clusters))))
    n_bins_y = max(1, int(np.ceil(n_clusters / n_bins_x)))
    return n_bins_x, n_bins_y


def fit_state_clusters(
    observations: np.ndarray,
    method: str,
    n_clusters: int,
    seed: int,
    state_dims: list[int] | None = None,
    n_bins_x: int | None = None,
    n_bins_y: int | None = None,
) -> dict[str, Any]:
    """Fit state clusters for support diagnostics."""

    observations = np.asarray(observations)
    if observations.shape[0] == 0:
        raise ValueError("Cannot cluster an empty observation array")

    method = method.lower()
    if method == "kmeans":
        if MiniBatchKMeans is None or StandardScaler is None:
            raise ImportError("cluster_method='kmeans' requires scikit-learn")
        features = _select_state_dims(observations, state_dims)
        actual_clusters = min(int(n_clusters), features.shape[0])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        kmeans = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=seed,
            batch_size=max(1024, actual_clusters * 10),
            n_init=10,
        )
        labels = kmeans.fit_predict(scaled).astype(np.int64)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        metadata = {
            "method": method,
            "requested_n_clusters": int(n_clusters),
            "n_clusters": int(actual_clusters),
            "feature_dim": int(features.shape[1]),
            "state_dims": state_dims,
        }
        return {
            "method": method,
            "labels": labels,
            "cluster_centers": centers,
            "scaler": scaler,
            "estimator": kmeans,
            "state_dims": state_dims,
            "metadata": metadata,
        }

    if method == "grid_xy":
        dims = state_dims if state_dims is not None else [0, 1]
        if len(dims) != 2:
            raise ValueError("grid_xy requires exactly two state dimensions")
        xy = _select_state_dims(observations, dims)
        n_bins_x, n_bins_y = _grid_bins(int(n_clusters), n_bins_x, n_bins_y)
        x_min, y_min = np.min(xy, axis=0)
        x_max, y_max = np.max(xy, axis=0)
        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            y_max = y_min + 1.0
        x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
        y_edges = np.linspace(y_min, y_max, n_bins_y + 1)
        x_idx = np.searchsorted(x_edges, xy[:, 0], side="right") - 1
        y_idx = np.searchsorted(y_edges, xy[:, 1], side="right") - 1
        x_idx = np.clip(x_idx, 0, n_bins_x - 1)
        y_idx = np.clip(y_idx, 0, n_bins_y - 1)
        labels = (x_idx * n_bins_y + y_idx).astype(np.int64)

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        centers = np.array(
            [[x, y] for x in x_centers for y in y_centers],
            dtype=xy.dtype,
        )
        total_clusters = int(n_bins_x * n_bins_y)
        metadata = {
            "method": method,
            "requested_n_clusters": int(n_clusters),
            "n_clusters": total_clusters,
            "n_bins_x": int(n_bins_x),
            "n_bins_y": int(n_bins_y),
            "state_dims": dims,
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)],
        }
        return {
            "method": method,
            "labels": labels,
            "cluster_centers": centers,
            "scaler": None,
            "estimator": None,
            "state_dims": dims,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "metadata": metadata,
        }

    raise ValueError(f"Unsupported cluster method {method!r}; expected 'kmeans' or 'grid_xy'")


def assign_clusters(observations: np.ndarray, cluster_model: dict[str, Any]) -> np.ndarray:
    method = cluster_model["method"]
    if method == "kmeans":
        features = _select_state_dims(observations, cluster_model.get("state_dims"))
        scaled = cluster_model["scaler"].transform(features)
        return cluster_model["estimator"].predict(scaled).astype(np.int64)

    if method == "grid_xy":
        dims = cluster_model.get("state_dims") or [0, 1]
        xy = _select_state_dims(observations, dims)
        x_edges = cluster_model["x_edges"]
        y_edges = cluster_model["y_edges"]
        n_bins_y = int(cluster_model["metadata"]["n_bins_y"])
        x_idx = np.searchsorted(x_edges, xy[:, 0], side="right") - 1
        y_idx = np.searchsorted(y_edges, xy[:, 1], side="right") - 1
        x_idx = np.clip(x_idx, 0, len(x_edges) - 2)
        y_idx = np.clip(y_idx, 0, len(y_edges) - 2)
        return (x_idx * n_bins_y + y_idx).astype(np.int64)

    raise ValueError(f"Unsupported cluster method {method!r}")


def compute_cluster_density(labels: np.ndarray, n_clusters: int | None = None) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        n_clusters = 0 if n_clusters is None else int(n_clusters)
        counts = np.zeros(n_clusters, dtype=np.int64)
    else:
        if n_clusters is None:
            n_clusters = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=int(n_clusters))[: int(n_clusters)]
    total = max(1, int(counts.sum()))
    return pd.DataFrame(
        {
            "cluster": np.arange(int(n_clusters), dtype=np.int64),
            "count": counts.astype(np.int64),
            "density": counts.astype(np.float64) / float(total),
        }
    )
