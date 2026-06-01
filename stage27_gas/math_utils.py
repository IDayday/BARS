from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def safe_l2(a: np.ndarray, b: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.sqrt(np.maximum(np.sum((a - b) ** 2, axis=axis), eps))


def robust_scale(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.nanmedian(x))
    q75 = float(np.nanpercentile(x, 75))
    q25 = float(np.nanpercentile(x, 25))
    iqr = max(q75 - q25, eps)
    return (x - med) / iqr, med, iqr


def minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    return (x - lo) / max(hi - lo, eps)


def pairwise_l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    d2 = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.sqrt(d2)


def knn_indices(x: np.ndarray, k: int, query: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return k nearest neighbors and distances.

    Uses sklearn when available, otherwise a chunked NumPy fallback.
    """
    x = np.asarray(x, dtype=np.float32)
    query = x if query is None else np.asarray(query, dtype=np.float32)
    k = int(min(k, len(x)))
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(x)
        dist, ind = nn.kneighbors(query, return_distance=True)
        return ind.astype(np.int64), dist.astype(np.float32)
    except Exception:
        rows = []
        drows = []
        chunk = 1024
        for start in range(0, len(query), chunk):
            q = query[start : start + chunk]
            d = pairwise_l2(q, x)
            ind = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            row_order = np.arange(len(q))[:, None]
            order = np.argsort(d[row_order, ind], axis=1)
            ind = ind[row_order, order]
            dist = d[row_order, ind]
            rows.append(ind)
            drows.append(dist)
        return np.vstack(rows).astype(np.int64), np.vstack(drows).astype(np.float32)


def farthest_point_sampling(x: np.ndarray, k: int, seed: int = 0, initial_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Greedy farthest-point sampling with NumPy-only O(NK) complexity."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(int(k), n)
    rng = np.random.default_rng(seed)
    selected = []
    min_d2 = np.full(n, np.inf, dtype=np.float32)

    if initial_indices is not None and len(initial_indices) > 0:
        for idx in np.asarray(initial_indices, dtype=np.int64)[:k]:
            if idx not in selected:
                selected.append(int(idx))
                d2 = np.sum((x - x[idx]) ** 2, axis=1)
                min_d2 = np.minimum(min_d2, d2)
    if not selected:
        idx = int(rng.integers(0, n))
        selected.append(idx)
        min_d2 = np.sum((x - x[idx]) ** 2, axis=1)

    while len(selected) < k:
        idx = int(np.argmax(min_d2))
        if idx in selected:
            # Degenerate duplicate embeddings; fill randomly with unseen indices.
            unseen = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=np.int64), assume_unique=False)
            if len(unseen) == 0:
                break
            idx = int(rng.choice(unseen))
        selected.append(idx)
        d2 = np.sum((x - x[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return np.asarray(selected, dtype=np.int64)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return float((centre - margin) / denom), float((centre + margin) / denom)


def bootstrap_mean_ci(values: np.ndarray, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
