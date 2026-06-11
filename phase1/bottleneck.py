from __future__ import annotations

from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd


EPS = 1e-12


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    min_v = float(np.nanmin(values))
    max_v = float(np.nanmax(values))
    if max_v - min_v < EPS:
        return np.zeros_like(values, dtype=np.float64)
    return (values - min_v) / (max_v - min_v)


def crossing_score(
    pair_records: Any,
    labels: np.ndarray,
    n_clusters: int,
    lag: int = 5,
) -> np.ndarray:
    """Count clusters that sit between different clusters in short trajectory windows."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.zeros(int(n_clusters), dtype=np.float64)
    lag = int(lag)
    if lag <= 0:
        return counts

    if isinstance(pair_records, list):
        for episode in pair_records:
            length = int(episode["length"])
            if length <= 2 * lag:
                continue
            start = int(episode["start_index"])
            idx = start + np.arange(length, dtype=np.int64)
            clusters = labels[idx]
            left = clusters[:-2 * lag]
            center = clusters[lag:-lag]
            right = clusters[2 * lag :]
            mask = (left != right) & (center >= 0) & (center < n_clusters)
            np.add.at(counts, center[mask], 1.0)
        return counts

    h_values = np.asarray(pair_records["h"], dtype=np.int64)
    global_i = np.asarray(pair_records["global_i"], dtype=np.int64)
    global_j = np.asarray(pair_records["global_j"], dtype=np.int64)
    mask = h_values == 2 * lag
    if not np.any(mask):
        return counts
    center_idx = global_i[mask] + lag
    valid = (center_idx >= 0) & (center_idx < labels.shape[0])
    left = labels[global_i[mask][valid]]
    center = labels[center_idx[valid]]
    right = labels[global_j[mask][valid]]
    crossing = (left != right) & (center >= 0) & (center < n_clusters)
    np.add.at(counts, center[crossing], 1.0)
    return counts


def _graph_n_clusters(G: nx.DiGraph) -> int:
    if "n_clusters" in G.graph:
        return int(G.graph["n_clusters"])
    if not G.nodes:
        return 0
    return int(max(G.nodes)) + 1


def betweenness_score(
    G: nx.DiGraph,
    sample_k: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    n_clusters = _graph_n_clusters(G)
    scores = np.zeros(n_clusters, dtype=np.float64)
    if G.number_of_nodes() == 0:
        return scores

    if sample_k is None:
        sample_k = min(256, G.number_of_nodes())
    sample_k = min(int(sample_k), G.number_of_nodes())
    if sample_k >= G.number_of_nodes():
        centrality = nx.betweenness_centrality(G, normalized=True)
    else:
        centrality = nx.betweenness_centrality(G, k=sample_k, seed=seed, normalized=True)
    for node, value in centrality.items():
        node_i = int(node)
        if 0 <= node_i < n_clusters:
            scores[node_i] = float(value)
    return scores


def _sample_ordered_pairs(nodes: np.ndarray, sample_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if nodes.size < 2 or sample_pairs <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    total_possible = int(nodes.size * (nodes.size - 1))
    n_samples = min(int(sample_pairs), total_possible)
    src_idx = rng.integers(0, nodes.size, size=n_samples)
    dst_idx = rng.integers(0, nodes.size - 1, size=n_samples)
    dst_idx = dst_idx + (dst_idx >= src_idx)
    return nodes[src_idx], nodes[dst_idx]


def _length_lookup(G: nx.DiGraph, srcs: np.ndarray, dsts: np.ndarray) -> np.ndarray:
    out = np.full(srcs.shape[0], np.inf, dtype=np.float64)
    for source in np.unique(srcs):
        if source not in G:
            continue
        lengths = nx.single_source_shortest_path_length(G, int(source))
        source_mask = srcs == source
        for idx in np.flatnonzero(source_mask):
            dst = int(dsts[idx])
            if dst in lengths:
                out[idx] = float(lengths[dst])
    return out


def removal_impact_score(
    G: nx.DiGraph,
    candidate_nodes: Iterable[int],
    sample_pairs: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Estimate reachable-pair loss after removing each candidate node."""

    n_clusters = _graph_n_clusters(G)
    impacts = np.zeros(n_clusters, dtype=np.float64)
    nodes = np.asarray(list(G.nodes()), dtype=np.int64)
    srcs, dsts = _sample_ordered_pairs(nodes, sample_pairs, seed)
    if srcs.size == 0:
        return impacts

    base_lengths = np.full(srcs.shape[0], np.inf, dtype=np.float64)
    path_indices_by_node: dict[int, list[int]] = {}
    for source in np.unique(srcs):
        if source not in G:
            continue
        paths = nx.single_source_shortest_path(G, int(source))
        source_mask = srcs == source
        for idx in np.flatnonzero(source_mask):
            dst = int(dsts[idx])
            path = paths.get(dst)
            if path is None:
                continue
            base_lengths[idx] = float(len(path) - 1)
            for node in path[1:-1]:
                path_indices_by_node.setdefault(int(node), []).append(int(idx))

    base_reachable = np.isfinite(base_lengths)
    if not np.any(base_reachable):
        return impacts

    for node in candidate_nodes:
        node_i = int(node)
        if node_i not in G or not (0 <= node_i < n_clusters):
            continue
        endpoint_mask = (srcs == node_i) | (dsts == node_i)
        eval_mask = base_reachable & ~endpoint_mask
        denom = int(eval_mask.sum())
        if denom == 0:
            continue
        impacted_idx = np.asarray(path_indices_by_node.get(node_i, []), dtype=np.int64)
        if impacted_idx.size == 0:
            continue
        impacted_idx = impacted_idx[eval_mask[impacted_idx]]
        if impacted_idx.size == 0:
            continue
        G_removed = G.copy()
        G_removed.remove_node(node_i)
        removed_lengths = _length_lookup(G_removed, srcs[impacted_idx], dsts[impacted_idx])
        lost = float(np.sum(~np.isfinite(removed_lengths)) / denom)
        longer = np.zeros_like(removed_lengths, dtype=np.float64)
        still_reachable = np.isfinite(removed_lengths)
        base_eval = base_lengths[impacted_idx]
        longer[still_reachable] = np.maximum(
            0.0,
            removed_lengths[still_reachable] - base_eval[still_reachable],
        ) / (base_eval[still_reachable] + EPS)
        impacts[node_i] = float(lost + 0.1 * np.sum(longer) / denom)
    return impacts


def _density_array(density: pd.DataFrame | np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(density, pd.DataFrame):
        arr = np.zeros(n_clusters, dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.float64)
        for _, row in density.iterrows():
            cluster = int(row["cluster"])
            if 0 <= cluster < n_clusters:
                arr[cluster] = float(row["density"])
                counts[cluster] = float(row.get("count", 0.0))
        return arr, counts
    arr = np.asarray(density, dtype=np.float64).reshape(-1)
    out = np.zeros(n_clusters, dtype=np.float64)
    out[: min(n_clusters, arr.size)] = arr[:n_clusters]
    counts = out.copy()
    return out, counts


def bottleneck_score(
    crossing: np.ndarray,
    betweenness: np.ndarray,
    removal_impact: np.ndarray,
    density: pd.DataFrame | np.ndarray,
    alpha_cross: float = 1.0,
    alpha_btw: float = 1.0,
    alpha_remove: float = 1.0,
    alpha_density: float = 0.3,
) -> pd.DataFrame:
    n_clusters = max(
        np.asarray(crossing).size,
        np.asarray(betweenness).size,
        np.asarray(removal_impact).size,
        len(density) if isinstance(density, pd.DataFrame) else np.asarray(density).size,
    )
    crossing_arr = np.zeros(n_clusters, dtype=np.float64)
    betweenness_arr = np.zeros(n_clusters, dtype=np.float64)
    removal_arr = np.zeros(n_clusters, dtype=np.float64)
    crossing_arr[: np.asarray(crossing).size] = np.asarray(crossing, dtype=np.float64)
    betweenness_arr[: np.asarray(betweenness).size] = np.asarray(betweenness, dtype=np.float64)
    removal_arr[: np.asarray(removal_impact).size] = np.asarray(removal_impact, dtype=np.float64)
    density_arr, count_arr = _density_array(density, n_clusters)

    norm_cross = _normalise(crossing_arr)
    norm_btw = _normalise(betweenness_arr)
    norm_remove = _normalise(removal_arr)
    norm_density = _normalise(density_arr)
    score = (
        alpha_cross * norm_cross
        + alpha_btw * norm_btw
        + alpha_remove * norm_remove
        - alpha_density * norm_density
    )
    return pd.DataFrame(
        {
            "cluster": np.arange(n_clusters, dtype=np.int64),
            "count": count_arr,
            "density": density_arr,
            "crossing": crossing_arr,
            "betweenness": betweenness_arr,
            "removal_impact": removal_arr,
            "normalized_crossing": norm_cross,
            "normalized_betweenness": norm_btw,
            "normalized_removal_impact": norm_remove,
            "normalized_density": norm_density,
            "bottleneck_score": score,
        }
    )


def _top_nodes(values: np.ndarray, fraction: float, at_least: int = 1) -> set[int]:
    n = values.size
    if n == 0:
        return set()
    budget = min(n, max(at_least, int(np.ceil(float(fraction) * n))))
    order = np.argsort(-values, kind="mergesort")
    return {int(x) for x in order[:budget]}


def filtering_risk_analysis(
    bottleneck_scores: pd.DataFrame,
    top_p: float = 0.2,
    high_bottleneck_q: float = 0.1,
    seed: int = 0,
    random_trials: int = 20,
) -> pd.DataFrame:
    """Compare how often simple filters retain high-bottleneck nodes."""

    scores = bottleneck_scores.sort_values("cluster").reset_index(drop=True)
    density = scores["density"].to_numpy(dtype=np.float64)
    bottleneck = scores["bottleneck_score"].to_numpy(dtype=np.float64)
    n_clusters = int(scores.shape[0])
    if n_clusters == 0:
        return pd.DataFrame()

    high_nodes = _top_nodes(bottleneck, high_bottleneck_q)
    budget = min(n_clusters, max(1, int(np.ceil(top_p * n_clusters))))
    density_nodes = _top_nodes(density, top_p)
    bottleneck_nodes = _top_nodes(bottleneck, top_p)

    rng = np.random.default_rng(seed)
    random_rates = []
    all_nodes = np.arange(n_clusters, dtype=np.int64)
    for _ in range(max(1, int(random_trials))):
        random_nodes = set(int(x) for x in rng.choice(all_nodes, size=budget, replace=False))
        random_rates.append(len(high_nodes & random_nodes) / max(1, len(high_nodes)))

    core_budget = budget // 2
    bot_budget = budget - core_budget
    density_order = np.argsort(-density, kind="mergesort")
    bottleneck_order = np.argsort(-bottleneck, kind="mergesort")
    core_plus: set[int] = {int(x) for x in density_order[:core_budget]}
    for node in bottleneck_order:
        core_plus.add(int(node))
        if len(core_plus) >= budget:
            break

    rows = [
        {
            "filter": "density_top_p",
            "budget": int(budget),
            "high_bottleneck_count": int(len(high_nodes)),
            "retention_rate": float(len(high_nodes & density_nodes) / max(1, len(high_nodes))),
        },
        {
            "filter": "random_same_budget",
            "budget": int(budget),
            "high_bottleneck_count": int(len(high_nodes)),
            "retention_rate": float(np.mean(random_rates)),
        },
        {
            "filter": "bottleneck_top_p",
            "budget": int(budget),
            "high_bottleneck_count": int(len(high_nodes)),
            "retention_rate": float(len(high_nodes & bottleneck_nodes) / max(1, len(high_nodes))),
        },
        {
            "filter": "core_plus_bottleneck",
            "budget": int(budget),
            "high_bottleneck_count": int(len(high_nodes)),
            "retention_rate": float(len(high_nodes & core_plus) / max(1, len(high_nodes))),
        },
    ]
    return pd.DataFrame(rows)
