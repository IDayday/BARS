from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs.
    PCA = None
    NearestNeighbors = None
    StandardScaler = None


def _flatten_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim == 1:
        return features.reshape(-1, 1)
    return features.reshape(features.shape[0], -1)


def _select_dims(features: np.ndarray, dims: list[int] | None) -> np.ndarray:
    flat = _flatten_features(features)
    if dims is None:
        return flat
    dim_idx = np.asarray(dims, dtype=np.int64)
    if np.any(dim_idx < 0) or np.any(dim_idx >= flat.shape[1]):
        raise ValueError(f"geometry/state dims {dims} are invalid for feature dim {flat.shape[1]}")
    return flat[:, dim_idx]


def _cluster_centers_from_samples(
    observations_or_features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = _flatten_features(observations_or_features).astype(np.float64, copy=False)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=n_clusters)[:n_clusters]
    present = np.flatnonzero(counts > 0)
    centers = np.zeros((present.size, features.shape[1]), dtype=np.float64)
    for out_idx, cluster in enumerate(present):
        centers[out_idx] = features[labels == cluster].mean(axis=0)
    return present.astype(np.int64), centers


def _knn_indices(features: np.ndarray, n_neighbors: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(features)
        _, neighbor_idx = nn.kneighbors(features, return_distance=True)
        return neighbor_idx

    rows = []
    for start in range(0, features.shape[0], 1024):
        chunk = features[start : start + 1024]
        dists = np.sum((chunk[:, None, :] - features[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(dists, kth=min(n_neighbors - 1, dists.shape[1] - 1), axis=1)[
            :, :n_neighbors
        ]
        order = np.take_along_axis(dists, idx, axis=1).argsort(axis=1, kind="mergesort")
        rows.append(np.take_along_axis(idx, order, axis=1))
    return np.vstack(rows)


def _standardize(features: np.ndarray) -> np.ndarray:
    if StandardScaler is not None:
        return StandardScaler().fit_transform(features)
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return (features - mean) / scale


def _pca_features(features: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if PCA is not None:
        return PCA(n_components=n_components, random_state=seed).fit_transform(features)
    del seed
    _, _, vt = np.linalg.svd(features, full_matrices=False)
    return features @ vt[:n_components].T


def build_knn_edges(
    observations_or_features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    k: int,
    mode: str,
    seed: int = 0,
    samples_per_cluster: int = 16,
) -> set[tuple[int, int]]:
    """Build directed cluster-level candidate edges from geometric kNN."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    mode = mode.lower()
    k = int(k)
    if k <= 0:
        return set()

    if mode == "cluster_center_knn":
        present, centers = _cluster_centers_from_samples(observations_or_features, labels, n_clusters)
        if present.size <= 1:
            return set()
        n_neighbors = min(k + 1, present.size)
        neighbor_idx = _knn_indices(centers, n_neighbors)
        edges: set[tuple[int, int]] = set()
        for row, cluster in enumerate(present):
            added = 0
            for col_idx in neighbor_idx[row]:
                neighbor = int(present[int(col_idx)])
                src = int(cluster)
                if src != neighbor:
                    edges.add((src, neighbor))
                    added += 1
                if added >= k:
                    break
        return edges

    if mode == "sample_knn":
        rng = np.random.default_rng(seed)
        features = _flatten_features(observations_or_features)
        sample_indices: list[np.ndarray] = []
        for cluster in range(n_clusters):
            members = np.flatnonzero(labels == cluster)
            if members.size == 0:
                continue
            size = min(samples_per_cluster, members.size)
            sample_indices.append(rng.choice(members, size=size, replace=False))
        if not sample_indices:
            return set()
        sampled = np.concatenate(sample_indices)
        sample_features = features[sampled]
        sample_labels = labels[sampled]
        if sampled.size <= 1:
            return set()
        neighbor_idx = _knn_indices(sample_features, min(k + 1, sampled.size))
        edges = set()
        for row, src_label in enumerate(sample_labels):
            for col_idx in neighbor_idx[row]:
                dst_label = int(sample_labels[int(col_idx)])
                src = int(src_label)
                if src != dst_label:
                    edges.add((src, dst_label))
        return edges

    raise ValueError(
        f"Unsupported kNN edge mode {mode!r}; expected 'cluster_center_knn' or 'sample_knn'"
    )


def build_grid_adjacent_edges(
    cluster_model: dict[str, Any],
    labels: np.ndarray | None = None,
    occupied_only: bool = True,
) -> set[tuple[int, int]]:
    """Build directed 4-neighbor edges for grid_xy clusters."""

    if cluster_model.get("method") != "grid_xy":
        return set()
    metadata = cluster_model["metadata"]
    n_bins_x = int(metadata["n_bins_x"])
    n_bins_y = int(metadata["n_bins_y"])
    if occupied_only and labels is not None:
        n_clusters = int(metadata["n_clusters"])
        occupied = set(
            int(x)
            for x in np.flatnonzero(
                np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_clusters)[:n_clusters]
                > 0
            )
        )
    else:
        occupied = set(range(n_bins_x * n_bins_y))

    edges: set[tuple[int, int]] = set()
    for x in range(n_bins_x):
        for y in range(n_bins_y):
            src = x * n_bins_y + y
            if src not in occupied:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= n_bins_x or ny < 0 or ny >= n_bins_y:
                    continue
                dst = nx * n_bins_y + ny
                if dst in occupied:
                    edges.add((int(src), int(dst)))
    return edges


def _lookup_support(
    N: sparse.spmatrix | np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)
    if sparse.issparse(N):
        return np.asarray(N.tocsr()[rows, cols]).reshape(-1)
    return np.asarray(N)[rows, cols]


def unsupported_edge_rate(
    candidate_edges: set[tuple[int, int]] | list[tuple[int, int]],
    N: sparse.spmatrix | np.ndarray,
    min_support: int,
) -> dict[str, float | int]:
    edges = list(candidate_edges)
    if not edges:
        return {
            "num_edges": 0,
            "unsupported_edge_rate": 0.0,
            "supported_edge_rate": 0.0,
            "reverse_supported_rate": 0.0,
            "bidirectional_supported_rate": 0.0,
        }
    rows = np.asarray([src for src, _ in edges], dtype=np.int64)
    cols = np.asarray([dst for _, dst in edges], dtype=np.int64)
    forward = _lookup_support(N, rows, cols)
    reverse = _lookup_support(N, cols, rows)
    supported = forward >= min_support
    reverse_supported = reverse >= min_support
    return {
        "num_edges": int(len(edges)),
        "unsupported_edge_rate": float(np.mean(~supported)),
        "supported_edge_rate": float(np.mean(supported)),
        "reverse_supported_rate": float(np.mean(reverse_supported)),
        "bidirectional_supported_rate": float(np.mean(supported & reverse_supported)),
    }


def _random_edges_like(
    labels: np.ndarray,
    n_clusters: int,
    num_edges: int,
    seed: int,
) -> set[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    present = np.flatnonzero(np.bincount(labels, minlength=n_clusters)[:n_clusters] > 0)
    if present.size <= 1 or num_edges <= 0:
        return set()
    max_edges = int(present.size * (present.size - 1))
    target = min(int(num_edges), max_edges)
    edges: set[tuple[int, int]] = set()
    while len(edges) < target:
        src_idx = int(rng.integers(0, present.size))
        dst_idx = int(rng.integers(0, present.size - 1))
        if dst_idx >= src_idx:
            dst_idx += 1
        edges.add((int(present[src_idx]), int(present[dst_idx])))
    return edges


def _support_edges(N: sparse.spmatrix | np.ndarray, min_support: int) -> set[tuple[int, int]]:
    return support_edges(N, min_support=min_support, include_self_loops=False)


def support_edges(
    N: sparse.spmatrix | np.ndarray,
    min_support: int,
    include_self_loops: bool = False,
) -> set[tuple[int, int]]:
    matrix = N.tocoo() if sparse.issparse(N) else sparse.coo_matrix(np.asarray(N))
    mask = matrix.data >= min_support
    if not include_self_loops:
        mask &= matrix.row != matrix.col
    return {
        (int(src), int(dst))
        for src, dst in zip(matrix.row[mask], matrix.col[mask])
    }


def compare_candidate_graphs(
    observations: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    N: sparse.spmatrix | np.ndarray,
    min_support: int,
    k: int,
    cluster_model: dict[str, Any] | None = None,
    geometry_dims: list[int] | None = None,
    geometry_label: str = "geometry_dims_kNN",
    include_grid_adjacent: bool = False,
    include_self_loops: bool = False,
    seed: int = 0,
    pca_dim: int = 16,
) -> tuple[pd.DataFrame, dict[str, set[tuple[int, int]]]]:
    """Compare geometric candidate graphs against empirical H-step support."""

    features = _flatten_features(observations)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    raw_edges = build_knn_edges(features, labels, n_clusters, k, "cluster_center_knn", seed=seed)
    geometry_edges: set[tuple[int, int]] = set()
    if geometry_dims is not None:
        geometry_features = _select_dims(features, geometry_dims)
        geometry_edges = build_knn_edges(
            geometry_features,
            labels,
            n_clusters,
            k,
            "cluster_center_knn",
            seed=seed,
        )

    pca_components = min(int(pca_dim), features.shape[1], max(1, features.shape[0] - 1))
    scaled = _standardize(features)
    if pca_components < features.shape[1]:
        pca_features = _pca_features(scaled, pca_components, seed)
    else:
        pca_features = scaled
    pca_edges = build_knn_edges(pca_features, labels, n_clusters, k, "cluster_center_knn", seed=seed)

    random_edges = _random_edges_like(labels, n_clusters, len(raw_edges), seed=seed)
    empirical_support_edges = support_edges(
        N,
        min_support=min_support,
        include_self_loops=include_self_loops,
    )
    grid_edges = (
        build_grid_adjacent_edges(cluster_model, labels=labels, occupied_only=True)
        if include_grid_adjacent and cluster_model is not None
        else set()
    )

    edge_sets = {
        "raw_state_kNN": raw_edges,
        "PCA_state_kNN": pca_edges,
        "random_edges": random_edges,
        "support_graph": empirical_support_edges,
    }
    if geometry_dims is not None:
        edge_sets[geometry_label] = geometry_edges
    if include_grid_adjacent and cluster_model is not None and cluster_model.get("method") == "grid_xy":
        edge_sets["grid_adjacent_edges"] = grid_edges

    rows: list[dict[str, Any]] = []
    for name, edges in edge_sets.items():
        metrics = unsupported_edge_rate(edges, N, min_support)
        rows.append({"candidate_type": name, **metrics})
    return pd.DataFrame(rows), edge_sets
