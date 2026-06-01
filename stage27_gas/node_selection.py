from __future__ import annotations

from typing import Dict

import numpy as np

from .config import NodeSelectConfig
from .dataset import OfflineDataset
from .math_utils import farthest_point_sampling, knn_indices


def _select_te_high(dataset: OfflineDataset, cfg: NodeSelectConfig) -> np.ndarray:
    if not dataset.has("te_scores"):
        return np.empty(0, dtype=np.int64)
    te = np.asarray(dataset.get("te_scores"), dtype=np.float32)
    te = np.nan_to_num(te, nan=-np.inf)
    if cfg.te_top_k is not None:
        k = min(int(cfg.te_top_k), len(te))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        return np.argpartition(-te, kth=k - 1)[:k].astype(np.int64)
    threshold = float(np.quantile(te[np.isfinite(te)], cfg.te_quantile)) if np.any(np.isfinite(te)) else np.inf
    return np.flatnonzero(te >= threshold).astype(np.int64)


def _select_endpoints(dataset: OfflineDataset, stride: int = 1) -> np.ndarray:
    selected = []
    stride = max(1, int(stride))
    for _, idx in dataset.trajectory_slices().items():
        if len(idx) == 0:
            continue
        selected.append(int(idx[0]))
        selected.append(int(idx[-1]))
        if stride > 1:
            selected.extend(idx[::stride].astype(int).tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _select_bottleneck_like(dataset: OfflineDataset, cfg: NodeSelectConfig, base_selected: np.ndarray) -> np.ndarray:
    """Approximate bottleneck/bridge states without heavy graph libraries.

    We use a light heuristic: build a kNN graph in embedding space, estimate local
    density by mean kNN distance, and pick states with high local-distance but
    central-ish position to avoid choosing only outliers. When networkx exists,
    this can be replaced by true betweenness centrality in the host project.
    """
    if cfg.bottleneck_k <= 0:
        return np.empty(0, dtype=np.int64)

    x = dataset.embedding(cfg.embedding_key, cfg.fallback_embedding_key)
    n = len(x)
    if n <= cfg.bottleneck_k:
        return np.arange(n, dtype=np.int64)

    # Use a manageable support set for O(N log N)-ish kNN.
    knn = min(max(3, cfg.bottleneck_knn), n)
    _, dist = knn_indices(x, knn)
    # Exclude self distance when present.
    local_spacing = np.mean(dist[:, 1:] if dist.shape[1] > 1 else dist, axis=1)

    center = np.mean(x, axis=0, keepdims=True)
    radial = np.sqrt(np.sum((x - center) ** 2, axis=1))
    radial_score = 1.0 - (radial - radial.min()) / max(radial.max() - radial.min(), 1e-8)

    score = local_spacing * (0.25 + 0.75 * radial_score)
    if len(base_selected) > 0:
        # Prefer adding new structural states rather than duplicating existing pools.
        score[np.asarray(base_selected, dtype=np.int64)] *= 0.5
    k = min(cfg.bottleneck_k, n)
    return np.argpartition(-score, kth=k - 1)[:k].astype(np.int64)


def select_stage27_nodes(dataset: OfflineDataset, cfg: NodeSelectConfig) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Select graph nodes as a union of TE, coverage, endpoint, bottleneck pools."""
    pools: Dict[str, np.ndarray] = {}
    pools["te_high"] = _select_te_high(dataset, cfg)

    x = dataset.embedding(cfg.embedding_key, cfg.fallback_embedding_key)
    pools["coverage"] = farthest_point_sampling(x, cfg.coverage_k, seed=cfg.fps_seed, initial_indices=pools["te_high"])

    if cfg.include_segment_endpoints:
        pools["segment_endpoints"] = _select_endpoints(dataset, cfg.endpoint_stride)
    else:
        pools["segment_endpoints"] = np.empty(0, dtype=np.int64)

    base = np.unique(np.concatenate([v for v in pools.values() if len(v) > 0])) if pools else np.empty(0, dtype=np.int64)
    if cfg.include_bottlenecks:
        pools["bottleneck_like"] = _select_bottleneck_like(dataset, cfg, base)
    else:
        pools["bottleneck_like"] = np.empty(0, dtype=np.int64)

    selected = np.unique(np.concatenate([v for v in pools.values() if len(v) > 0])).astype(np.int64)
    if len(selected) > cfg.max_nodes:
        # Keep endpoints first, then TE, then fill by FPS/coverage ordering.
        priority = []
        for name in ["segment_endpoints", "te_high", "bottleneck_like", "coverage"]:
            for idx in pools[name].tolist():
                if idx not in priority:
                    priority.append(int(idx))
                if len(priority) >= cfg.max_nodes:
                    break
            if len(priority) >= cfg.max_nodes:
                break
        selected = np.asarray(priority[: cfg.max_nodes], dtype=np.int64)
    selected = np.unique(selected)

    # Filter pool arrays to selected for cleaner reporting.
    selected_set = set(selected.tolist())
    pools = {name: np.asarray([i for i in arr.tolist() if i in selected_set], dtype=np.int64) for name, arr in pools.items()}
    return selected, pools
