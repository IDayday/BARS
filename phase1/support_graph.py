from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse


EPS = 1e-12


def _record_array(pair_records: dict[str, Any], key: str) -> np.ndarray:
    value = pair_records[key]
    return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)


def _to_csr(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def _lookup(matrix: sparse.spmatrix | np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)
    if sparse.issparse(matrix):
        return np.asarray(matrix.tocsr()[rows, cols]).reshape(-1)
    return np.asarray(matrix)[rows, cols]


def compute_support_counts(
    pair_records: dict[str, Any],
    labels: np.ndarray,
    horizons: list[int],
    n_clusters: int,
) -> dict[int, sparse.csr_matrix]:
    """Compute cumulative H-step directed support counts N_ij^H."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    global_i = _record_array(pair_records, "global_i").astype(np.int64)
    global_j = _record_array(pair_records, "global_j").astype(np.int64)
    h_values = _record_array(pair_records, "h").astype(np.int64)
    horizons = sorted({int(h) for h in horizons if int(h) > 0})

    if global_i.size == 0:
        zero = sparse.csr_matrix((n_clusters, n_clusters), dtype=np.int64)
        return {h: zero.copy() for h in horizons}

    valid = (
        (global_i >= 0)
        & (global_i < labels.shape[0])
        & (global_j >= 0)
        & (global_j < labels.shape[0])
    )
    src = labels[global_i[valid]]
    dst = labels[global_j[valid]]
    h_values = h_values[valid]
    valid_clusters = (src >= 0) & (src < n_clusters) & (dst >= 0) & (dst < n_clusters)
    src = src[valid_clusters]
    dst = dst[valid_clusters]
    h_values = h_values[valid_clusters]

    counts_by_h: dict[int, sparse.csr_matrix] = {}
    for horizon in horizons:
        mask = h_values <= horizon
        if not np.any(mask):
            counts_by_h[horizon] = sparse.csr_matrix((n_clusters, n_clusters), dtype=np.int64)
            continue
        data = np.ones(int(mask.sum()), dtype=np.int64)
        counts_by_h[horizon] = sparse.coo_matrix(
            (data, (src[mask], dst[mask])),
            shape=(n_clusters, n_clusters),
            dtype=np.int64,
        ).tocsr()
    return counts_by_h


def support_asymmetry(N: sparse.spmatrix | np.ndarray) -> float:
    matrix = _to_csr(N)
    diff = matrix - matrix.T
    diff = diff.tocoo()
    numerator = float(np.abs(diff.data).sum())
    denominator = float((matrix + matrix.T).sum()) + EPS
    return numerator / denominator


def one_way_edge_ratio(
    N: sparse.spmatrix | np.ndarray,
    min_support: int,
    include_self_loops: bool = False,
) -> float:
    matrix = _to_csr(N)
    coo = matrix.tocoo()
    supported = coo.data >= min_support
    if not include_self_loops:
        supported &= coo.row != coo.col
    if not np.any(supported):
        return 0.0
    rows = coo.row[supported]
    cols = coo.col[supported]
    reverse = _lookup(matrix, cols, rows)
    return float(np.mean(reverse < min_support))


def supported_edge_stats(
    N: sparse.spmatrix | np.ndarray,
    min_support: int,
) -> dict[str, int]:
    matrix = _to_csr(N)
    coo = matrix.tocoo()
    supported = coo.data >= min_support
    self_edges = supported & (coo.row == coo.col)
    inter_edges = supported & (coo.row != coo.col)
    return {
        "num_self_edges": int(np.count_nonzero(self_edges)),
        "num_inter_edges": int(np.count_nonzero(inter_edges)),
        "num_all_supported_edges": int(np.count_nonzero(supported)),
    }


def build_directed_support_graph(
    N: sparse.spmatrix | np.ndarray,
    min_support: int,
    include_self_loops: bool = False,
) -> nx.DiGraph:
    matrix = _to_csr(N)
    coo = matrix.tocoo()
    graph = nx.DiGraph()
    graph.add_nodes_from(range(matrix.shape[0]))
    graph.graph["n_clusters"] = int(matrix.shape[0])
    graph.graph["min_support"] = int(min_support)
    graph.graph["include_self_loops"] = bool(include_self_loops)
    supported = coo.data >= min_support
    if not include_self_loops:
        supported &= coo.row != coo.col
    rows = coo.row[supported]
    cols = coo.col[supported]
    counts = coo.data[supported]
    reverse_counts = _lookup(matrix, cols, rows)
    for src, dst, count, reverse_count in zip(rows, cols, counts, reverse_counts):
        denom = float(count + reverse_count) + EPS
        graph.add_edge(
            int(src),
            int(dst),
            count=int(count),
            reverse_count=int(reverse_count),
            asymmetry=float((count - reverse_count) / denom),
        )
    return graph


def directed_shortest_path_asymmetry(
    G: nx.DiGraph,
    sample_pairs: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate directed shortest-path asymmetry on sampled node pairs."""

    nodes = np.asarray(list(G.nodes()), dtype=np.int64)
    if nodes.size < 2 or sample_pairs <= 0:
        return {
            "mean_directed_asymmetry": 0.0,
            "one_way_reachable_ratio": 0.0,
            "reachable_pair_ratio": 0.0,
            "directed_reachable_pair_ratio": 0.0,
            "num_sampled_pairs": 0,
        }

    rng = np.random.default_rng(seed)
    total_possible = int(nodes.size * (nodes.size - 1))
    n_samples = min(int(sample_pairs), total_possible)
    src_idx = rng.integers(0, nodes.size, size=n_samples)
    dst_idx = rng.integers(0, nodes.size - 1, size=n_samples)
    dst_idx = dst_idx + (dst_idx >= src_idx)
    srcs = nodes[src_idx]
    dsts = nodes[dst_idx]

    sources = np.unique(np.concatenate([srcs, dsts]))
    lengths: dict[int, dict[int, int]] = {}
    for source in sources:
        lengths[int(source)] = nx.single_source_shortest_path_length(G, int(source))

    either_reachable = 0
    directed_reachable = 0
    one_way = 0
    asymmetries: list[float] = []
    for src, dst in zip(srcs, dsts):
        src_i = int(src)
        dst_i = int(dst)
        dij = lengths[src_i].get(dst_i)
        dji = lengths[dst_i].get(src_i)
        fwd = dij is not None
        rev = dji is not None
        directed_reachable += int(fwd) + int(rev)
        if not (fwd or rev):
            continue
        either_reachable += 1
        if fwd ^ rev:
            one_way += 1
            asymmetries.append(1.0)
        else:
            asymmetries.append(abs(float(dij) - float(dji)) / (float(dij + dji) + EPS))

    sampled = max(1, int(n_samples))
    return {
        "mean_directed_asymmetry": float(np.mean(asymmetries)) if asymmetries else 0.0,
        "one_way_reachable_ratio": float(one_way / sampled),
        "reachable_pair_ratio": float(either_reachable / sampled),
        "directed_reachable_pair_ratio": float(directed_reachable / (2 * sampled)),
        "num_sampled_pairs": int(n_samples),
    }
