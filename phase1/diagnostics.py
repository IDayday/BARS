from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _flatten_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim == 1:
        return features.reshape(-1, 1)
    return features.reshape(features.shape[0], -1)


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
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(centers)
        _, neighbor_idx = nn.kneighbors(centers, return_distance=True)
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
        nn = NearestNeighbors(n_neighbors=min(k + 1, sampled.size))
        nn.fit(sample_features)
        _, neighbor_idx = nn.kneighbors(sample_features, return_distance=True)
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
    matrix = N.tocoo() if sparse.issparse(N) else sparse.coo_matrix(np.asarray(N))
    mask = matrix.data >= min_support
    return {
        (int(src), int(dst))
        for src, dst in zip(matrix.row[mask], matrix.col[mask])
        if int(src) != int(dst)
    }


def compare_candidate_graphs(
    observations: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    N: sparse.spmatrix | np.ndarray,
    min_support: int,
    k: int,
    seed: int = 0,
    pca_dim: int = 16,
) -> tuple[pd.DataFrame, dict[str, set[tuple[int, int]]]]:
    """Compare geometric candidate graphs against empirical H-step support."""

    features = _flatten_features(observations)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    raw_edges = build_knn_edges(features, labels, n_clusters, k, "cluster_center_knn", seed=seed)

    pca_components = min(int(pca_dim), features.shape[1], max(1, features.shape[0] - 1))
    scaled = StandardScaler().fit_transform(features)
    if pca_components < features.shape[1]:
        pca_features = PCA(n_components=pca_components, random_state=seed).fit_transform(scaled)
    else:
        pca_features = scaled
    pca_edges = build_knn_edges(pca_features, labels, n_clusters, k, "cluster_center_knn", seed=seed)

    random_edges = _random_edges_like(labels, n_clusters, len(raw_edges), seed=seed)
    support_edges = _support_edges(N, min_support)

    edge_sets = {
        "raw_state_kNN": raw_edges,
        "PCA_state_kNN": pca_edges,
        "random_edges": random_edges,
        "support_graph": support_edges,
    }

    rows: list[dict[str, Any]] = []
    for name, edges in edge_sets.items():
        metrics = unsupported_edge_rate(edges, N, min_support)
        rows.append({"candidate_type": name, **metrics})
    return pd.DataFrame(rows), edge_sets
