from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from phase1.diagnostics import build_knn_edges


def support_pair_lookup(option_edges: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row.src), int(row.dst)): row._asdict()
        for row in option_edges.itertuples(index=False)
    }


def _edge_cost(src: int, dst: int, support_lookup: dict[tuple[int, int], dict[str, Any]], default_cost: float) -> float:
    meta = support_lookup.get((int(src), int(dst)))
    if meta is None:
        return float(default_cost)
    return float(meta.get("cost", meta.get("median_h", default_cost)))


def build_graph_from_pairs(
    pairs: set[tuple[int, int]] | list[tuple[int, int]],
    support_lookup: dict[tuple[int, int], dict[str, Any]],
    default_cost: float,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for src, dst in pairs:
        supported = (int(src), int(dst)) in support_lookup
        meta = support_lookup.get((int(src), int(dst)), {})
        graph.add_edge(
            int(src),
            int(dst),
            cost=_edge_cost(int(src), int(dst), support_lookup, default_cost),
            supported=supported,
            edge_id=int(meta.get("edge_id", -1)) if supported else -1,
        )
    return graph


def _cluster_centers(observations: np.ndarray, labels: np.ndarray, selected: list[int]) -> dict[int, np.ndarray]:
    features = np.asarray(observations)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    else:
        features = features.reshape(features.shape[0], -1)
    labels = np.asarray(labels, dtype=np.int64)
    centers: dict[int, np.ndarray] = {}
    for cluster in selected:
        idx = np.flatnonzero(labels == int(cluster))
        if idx.size:
            centers[int(cluster)] = features[idx].mean(axis=0)
    return centers


def build_gas_style_threshold_edges(
    observations: np.ndarray,
    labels: np.ndarray,
    selected_nodes: list[int],
    target_num_edges: int,
) -> set[tuple[int, int]]:
    centers = _cluster_centers(observations, labels, selected_nodes)
    nodes = sorted(centers)
    if len(nodes) <= 1 or target_num_edges <= 0:
        return set()
    rows: list[tuple[float, int, int]] = []
    matrix = np.vstack([centers[node] for node in nodes]).astype(np.float64)
    for i, src in enumerate(nodes):
        diff = matrix - matrix[i]
        dist = np.linalg.norm(diff, axis=1)
        for j, dst in enumerate(nodes):
            if src != dst:
                rows.append((float(dist[j]), int(src), int(dst)))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return {(src, dst) for _, src, dst in rows[: min(int(target_num_edges), len(rows))]}


def build_random_edges(
    selected_nodes: list[int],
    edge_budget: int,
    seed: int = 0,
) -> set[tuple[int, int]]:
    nodes = np.asarray(sorted(set(int(x) for x in selected_nodes)), dtype=np.int64)
    if nodes.size <= 1 or edge_budget <= 0:
        return set()
    possible = [(int(src), int(dst)) for src in nodes for dst in nodes if int(src) != int(dst)]
    rng = np.random.default_rng(int(seed))
    take = min(int(edge_budget), len(possible))
    idx = rng.choice(len(possible), size=take, replace=False)
    return {possible[int(i)] for i in idx}


def build_audit_edge_sets(
    option_edges: pd.DataFrame,
    selected_nodes: pd.DataFrame,
    observations: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    knn_k: int = 10,
    seed: int = 0,
) -> dict[str, set[tuple[int, int]]]:
    selected = selected_nodes[selected_nodes["selected"]]["cluster"].astype(int).tolist()
    support_edges = {(int(row.src), int(row.dst)) for row in option_edges.itertuples(index=False)}
    edge_budget = len(support_edges)
    knn_edges = {
        pair
        for pair in build_knn_edges(
            observations,
            labels,
            int(n_clusters),
            int(knn_k),
            mode="cluster_center_knn",
            seed=seed,
        )
        if pair[0] in selected and pair[1] in selected and pair[0] != pair[1]
    }
    return {
        "support_graph": support_edges,
        "kNN_graph": set(sorted(knn_edges)[:edge_budget]),
        "random_graph": build_random_edges(selected, edge_budget=edge_budget, seed=seed),
        "GAS_style_threshold_graph": build_gas_style_threshold_edges(
            observations,
            labels,
            selected,
            target_num_edges=edge_budget,
        ),
    }


def edge_provenance_audit(
    edge_sets: dict[str, set[tuple[int, int]]],
    option_edges: pd.DataFrame,
) -> pd.DataFrame:
    support_lookup = support_pair_lookup(option_edges)
    rows = []
    for graph_type, pairs in edge_sets.items():
        pairs = set((int(src), int(dst)) for src, dst in pairs)
        supported = np.asarray([(src, dst) in support_lookup for src, dst in pairs], dtype=bool)
        reverse = np.asarray([(dst, src) in support_lookup for src, dst in pairs], dtype=bool)
        observed_h = [
            float(support_lookup[(src, dst)]["median_h"])
            for src, dst in pairs
            if (src, dst) in support_lookup
        ]
        rows.append(
            {
                "graph_type": graph_type,
                "graph_note": "GAS_style_approximation, not official GAS graph"
                if graph_type == "GAS_style_threshold_graph"
                else "",
                "num_edges": int(len(pairs)),
                "supported_edge_rate": float(supported.mean()) if supported.size else 0.0,
                "unsupported_edge_rate": float((~supported).mean()) if supported.size else 0.0,
                "reverse_supported_rate": float(reverse.mean()) if reverse.size else 0.0,
                "bidirectional_supported_rate": float((supported & reverse).mean()) if supported.size else 0.0,
                "median_observed_h_for_supported_edges": float(np.median(observed_h)) if observed_h else np.nan,
                "fraction_no_observed_h_leq_H": float((~supported).mean()) if supported.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _parse_queries(path_queries: pd.DataFrame, max_queries: int | None = None, seed: int = 0) -> pd.DataFrame:
    queries = path_queries.copy()
    if "start_cluster" not in queries.columns and "src" in queries.columns:
        queries = queries.rename(columns={"src": "start_cluster", "dst": "goal_cluster"})
    if max_queries is not None and queries.shape[0] > int(max_queries):
        queries = queries.sample(n=int(max_queries), random_state=int(seed)).reset_index(drop=True)
    if "query_id" not in queries.columns:
        queries["query_id"] = np.arange(queries.shape[0], dtype=np.int64)
    return queries


def _path_edges(path: list[int]) -> list[tuple[int, int]]:
    return list(zip(path[:-1], path[1:]))


def path_risk_audit(
    edge_sets: dict[str, set[tuple[int, int]]],
    option_edges: pd.DataFrame,
    path_queries: pd.DataFrame,
    edge_certification: pd.DataFrame | None = None,
    pair_compatibility: pd.DataFrame | None = None,
    max_queries: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    support_lookup = support_pair_lookup(option_edges)
    default_cost = float(pd.to_numeric(option_edges.get("median_h", 1.0), errors="coerce").median())
    if not np.isfinite(default_cost) or default_cost <= 0:
        default_cost = 1.0
    proxy_lookup: dict[tuple[int, int], float] = {}
    if edge_certification is not None and not edge_certification.empty:
        proxy_lookup = {
            (int(row.src), int(row.dst)): float(row.edge_proxy_score)
            for row in edge_certification.itertuples(index=False)
        }
    strict_pairs = set()
    if pair_compatibility is not None and not pair_compatibility.empty:
        strict_pairs = {
            (int(row.edge_id_first), int(row.edge_id_second))
            for row in pair_compatibility.itertuples(index=False)
            if bool(row.strict_compatible)
        }
    bottleneck_thr = float(pd.to_numeric(option_edges["edge_bottleneck_score"], errors="coerce").median())
    queries = _parse_queries(path_queries, max_queries=max_queries, seed=seed)
    rows = []
    for graph_type, pairs in edge_sets.items():
        graph = build_graph_from_pairs(pairs, support_lookup, default_cost=default_cost)
        for query in queries.itertuples(index=False):
            src = int(query.start_cluster)
            dst = int(query.goal_cluster)
            row = {
                "graph_type": graph_type,
                "query_id": int(query.query_id),
                "src": src,
                "dst": dst,
                "reachable": False,
                "path_length": 0,
                "path_cost": np.nan,
                "unsupported_edge_fraction": 0.0,
                "incompatible_edge_fraction": 0.0,
                "bottleneck_edge_fraction": 0.0,
                "mean_edge_proxy_score": 0.0,
                "min_edge_proxy_score": 0.0,
                "proxy_path_success": 0.0,
                "uses_unsupported_shortcut": False,
            }
            if src not in graph or dst not in graph:
                rows.append(row)
                continue
            try:
                path = nx.shortest_path(graph, src, dst, weight="cost")
                cost = nx.shortest_path_length(graph, src, dst, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                rows.append(row)
                continue
            edges = _path_edges([int(x) for x in path])
            supported_flags = [(a, b) in support_lookup for a, b in edges]
            edge_ids = [
                int(support_lookup[(a, b)]["edge_id"]) if (a, b) in support_lookup else -1
                for a, b in edges
            ]
            adjacent_edge_ids = list(zip(edge_ids[:-1], edge_ids[1:]))
            incompatible = [
                (a < 0 or b < 0 or (a, b) not in strict_pairs)
                for a, b in adjacent_edge_ids
            ]
            bottleneck_flags = [
                bool(support_lookup[(a, b)]["edge_bottleneck_score"] >= bottleneck_thr)
                if (a, b) in support_lookup
                else False
                for a, b in edges
            ]
            scores = [float(np.clip(proxy_lookup.get((a, b), 0.0), 0.0, 1.0)) for a, b in edges]
            row.update(
                {
                    "reachable": True,
                    "path_length": int(len(edges)),
                    "path_cost": float(cost),
                    "unsupported_edge_fraction": float(np.mean([not x for x in supported_flags])) if edges else 0.0,
                    "incompatible_edge_fraction": float(np.mean(incompatible)) if incompatible else 0.0,
                    "bottleneck_edge_fraction": float(np.mean(bottleneck_flags)) if bottleneck_flags else 0.0,
                    "mean_edge_proxy_score": float(np.mean(scores)) if scores else 0.0,
                    "min_edge_proxy_score": float(np.min(scores)) if scores else 0.0,
                    "proxy_path_success": float(np.prod(scores)) if scores else 0.0,
                    "uses_unsupported_shortcut": bool(any(not x for x in supported_flags)),
                    "path_nodes": " ".join(str(x) for x in path),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def bottleneck_audit(
    edge_sets: dict[str, set[tuple[int, int]]],
    option_edges: pd.DataFrame,
    bottleneck_df: pd.DataFrame,
    path_queries: pd.DataFrame,
    top_q: float = 0.1,
    max_queries: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    support_lookup = support_pair_lookup(option_edges)
    default_cost = float(pd.to_numeric(option_edges.get("median_h", 1.0), errors="coerce").median())
    queries = _parse_queries(path_queries, max_queries=max_queries, seed=seed)
    rows = []
    top_nodes = (
        bottleneck_df.sort_values("bottleneck_score", ascending=False)
        .head(max(1, int(round(float(top_q) * bottleneck_df.shape[0]))))["cluster"]
        .astype(int)
        .tolist()
    )
    top_set = set(top_nodes)
    bottleneck_thr = float(pd.to_numeric(option_edges["edge_bottleneck_score"], errors="coerce").median())
    for graph_type, pairs in edge_sets.items():
        graph = build_graph_from_pairs(pairs, support_lookup, default_cost=default_cost)
        nodes = set(int(x) for x in graph.nodes())
        retained = len(top_set & nodes) / max(1, len(top_set))
        edge_bottleneck = [
            bool(support_lookup[(a, b)]["edge_bottleneck_score"] >= bottleneck_thr)
            if (a, b) in support_lookup
            else False
            for a, b in pairs
        ]

        def _coverage_and_cost(g: nx.DiGraph) -> tuple[float, float]:
            reachable = 0
            costs = []
            for q in queries.itertuples(index=False):
                try:
                    c = nx.shortest_path_length(g, int(q.start_cluster), int(q.goal_cluster), weight="cost")
                    reachable += 1
                    costs.append(float(c))
                except Exception:
                    continue
            return float(reachable / max(1, queries.shape[0])), float(np.mean(costs)) if costs else np.nan

        before_cov, before_cost = _coverage_and_cost(graph)
        removed = graph.copy()
        removed.remove_nodes_from(top_nodes)
        after_cov, after_cost = _coverage_and_cost(removed)
        rows.append(
            {
                "graph_type": graph_type,
                "high_bottleneck_nodes_retained_rate": float(retained),
                "bottleneck_edge_fraction": float(np.mean(edge_bottleneck)) if edge_bottleneck else 0.0,
                "bottleneck_removal_delta_coverage": float(after_cov - before_cov),
                "bottleneck_removal_delta_cost": float(after_cost - before_cost)
                if np.isfinite(before_cost) and np.isfinite(after_cost)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def audit_summary(edge_audit: pd.DataFrame, path_audit: pd.DataFrame, bottleneck: pd.DataFrame) -> dict[str, Any]:
    path_group = (
        path_audit.groupby("graph_type", sort=True)
        .agg(
            path_coverage=("reachable", "mean"),
            mean_unsupported_edge_fraction=("unsupported_edge_fraction", "mean"),
            shortcut_reliance_rate=("uses_unsupported_shortcut", "mean"),
            mean_proxy_path_success=("proxy_path_success", "mean"),
        )
        .reset_index()
        if not path_audit.empty
        else pd.DataFrame()
    )
    merged = edge_audit.merge(path_group, on="graph_type", how="left").merge(bottleneck, on="graph_type", how="left")

    def _best(col: str, ascending: bool) -> str | None:
        if merged.empty or col not in merged.columns:
            return None
        ranked = merged.sort_values(col, ascending=ascending, kind="mergesort")
        return str(ranked.iloc[0]["graph_type"]) if not ranked.empty else None

    return {
        "which_graph_has_highest_path_coverage": _best("path_coverage", ascending=False),
        "which_graph_has_lowest_unsupported_edge_rate": _best("unsupported_edge_rate", ascending=True),
        "which_graph_relies_most_on_unsupported_shortcuts": _best("shortcut_reliance_rate", ascending=False),
        "support_certified_graph_reduces_path_risk": bool(
            not merged.empty
            and "support_graph" in set(merged["graph_type"])
            and float(
                merged.loc[merged["graph_type"] == "support_graph", "unsupported_edge_rate"].iloc[0]
            )
            <= float(merged["unsupported_edge_rate"].median())
        ),
        "gas_style_or_proximity_graph_overestimates_connectivity": bool(
            not merged.empty
            and any(
                (row.graph_type in {"GAS_style_threshold_graph", "kNN_graph"})
                and bool(row.path_coverage >= merged["path_coverage"].median())
                and bool(row.unsupported_edge_rate > 0.0)
                for row in merged.itertuples(index=False)
            )
        ),
        "graph_metrics": merged.to_dict("records"),
        "note": "GAS_style_threshold_graph is a diagnostic approximation, not an official GAS graph.",
    }
