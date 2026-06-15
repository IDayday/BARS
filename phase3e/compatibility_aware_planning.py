from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any

import numpy as np
import pandas as pd

from phase2.compatibility import compute_edge_compatibility
from phase3e.risk_aware_planning import (
    EPS,
    RiskPlannerConfig,
    _as_bool,
    _as_float,
    _clip01,
    load_edge_table,
    risk_penalized_cost,
)


@dataclass(frozen=True)
class CompatibilityPlannerConfig:
    pair_weight: float = 10.0
    min_pair_coverage: float = 0.05
    missing_pair_coverage: float = 0.0
    high_pair_risk_threshold: float = 0.5
    risk_config: RiskPlannerConfig = RiskPlannerConfig()


def compute_pair_compatibility_from_segments(
    option_edges: pd.DataFrame,
    edge_segments: dict[str, np.ndarray],
    H_intra: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute Phase 2.2 adjacent-edge compatibility from saved segments."""

    summary, pair_df = compute_edge_compatibility(
        option_edges_df=option_edges,
        edge_segments=edge_segments,
        labels=np.empty(0, dtype=np.int64),
        pair_records={},
        H_intra=int(H_intra),
    )
    return summary, pair_df


def make_compatibility_edge_table(
    option_edges: pd.DataFrame,
    certification: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge option edges with calibrated/offline certification for planning."""

    table = load_edge_table(option_edges, certification)
    if "edge_id" not in table.columns:
        raise ValueError("edge table must contain edge_id")
    table = table.copy()
    table["edge_id"] = table["edge_id"].astype(int)
    table["src"] = table["src"].astype(int)
    table["dst"] = table["dst"].astype(int)
    return table


def _parse_queries(path_queries: pd.DataFrame, max_queries: int | None, seed: int) -> pd.DataFrame:
    queries = path_queries.copy()
    if "start_cluster" not in queries.columns and {"src", "dst"} <= set(queries.columns):
        queries = queries.rename(columns={"src": "start_cluster", "dst": "goal_cluster"})
    if "query_id" not in queries.columns:
        queries["query_id"] = np.arange(queries.shape[0], dtype=np.int64)
    if max_queries is not None and queries.shape[0] > int(max_queries):
        queries = queries.sample(n=int(max_queries), random_state=int(seed)).reset_index(drop=True)
    return queries.reset_index(drop=True)


def _edge_cost(row: dict[str, Any], method: str, config: CompatibilityPlannerConfig) -> float:
    if method in {"calibrated_edge_penalized", "calibrated_compat_penalized", "calibrated_compat_threshold"}:
        row_obj = type("Row", (), row)
        return float(risk_penalized_cost(row_obj, config.risk_config))
    return max(EPS, _as_float(row.get("base_cost", row.get("cost", 1.0)), 1.0))


def _method_uses_pair_penalty(method: str) -> bool:
    return method in {"compat_penalized", "calibrated_compat_penalized"}


def _method_uses_pair_threshold(method: str) -> bool:
    return method in {"compat_threshold", "calibrated_compat_threshold"}


def _pair_key(first_edge_id: int, second_edge_id: int) -> tuple[int, int]:
    return int(first_edge_id), int(second_edge_id)


def _pair_lookup(pair_df: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    if pair_df.empty:
        return {}
    return {
        _pair_key(row.edge_id_first, row.edge_id_second): row._asdict()
        for row in pair_df.itertuples(index=False)
    }


def _pair_coverage_lookup(
    pair_map: dict[tuple[int, int], dict[str, Any]],
    config: CompatibilityPlannerConfig,
) -> dict[tuple[int, int], float]:
    return {
        key: _clip01(row.get("termination_bridge_coverage", config.missing_pair_coverage))
        for key, row in pair_map.items()
    }


def _planning_index(edge_table: pd.DataFrame) -> dict[str, Any]:
    rows_by_id = {int(row.edge_id): row._asdict() for row in edge_table.itertuples(index=False)}
    by_src: dict[int, list[int]] = {}
    nodes: set[int] = set()
    for edge_id, row in rows_by_id.items():
        src = int(row["src"])
        dst = int(row["dst"])
        by_src.setdefault(src, []).append(int(edge_id))
        nodes.add(src)
        nodes.add(dst)
    return {"rows_by_id": rows_by_id, "by_src": by_src, "nodes": nodes}


def _method_edge_costs(
    rows_by_id: dict[int, dict[str, Any]],
    method: str,
    config: CompatibilityPlannerConfig,
) -> dict[int, float]:
    return {edge_id: _edge_cost(row, method, config) for edge_id, row in rows_by_id.items()}


def _pair_metrics(
    first_edge_id: int,
    second_edge_id: int,
    pair_map: dict[tuple[int, int], dict[str, Any]],
    config: CompatibilityPlannerConfig,
) -> dict[str, Any]:
    row = pair_map.get(_pair_key(first_edge_id, second_edge_id))
    if row is None:
        coverage = _clip01(config.missing_pair_coverage)
        return {
            "termination_bridge_coverage": coverage,
            "strict_compatible": False,
            "bridge_matches_per_first_segment": 0.0,
            "num_bridge_segments": 0,
            "num_bridge_episodes": 0,
            "missing_pair": True,
        }
    coverage = _clip01(row.get("termination_bridge_coverage", 0.0))
    return {
        "termination_bridge_coverage": coverage,
        "strict_compatible": _as_bool(row.get("strict_compatible", False)),
        "bridge_matches_per_first_segment": max(
            0.0, _as_float(row.get("bridge_matches_per_first_segment", 0.0), 0.0)
        ),
        "num_bridge_segments": int(_as_float(row.get("num_bridge_segments", 0), 0)),
        "num_bridge_episodes": int(_as_float(row.get("num_bridge_episodes", 0), 0)),
        "missing_pair": False,
    }


def _pair_penalty(
    first_edge_id: int,
    second_edge_id: int,
    pair_map: dict[tuple[int, int], dict[str, Any]],
    config: CompatibilityPlannerConfig,
) -> float:
    metrics = _pair_metrics(first_edge_id, second_edge_id, pair_map, config)
    return float(config.pair_weight) * (1.0 - _clip01(metrics["termination_bridge_coverage"]))


def _path_from_prev(
    final_edge_id: int,
    prev: dict[int, int | None],
) -> list[int]:
    out: list[int] = []
    cur: int | None = int(final_edge_id)
    while cur is not None:
        out.append(int(cur))
        cur = prev.get(int(cur))
    return list(reversed(out))


def _line_graph_shortest_path(
    edge_table: pd.DataFrame,
    pair_map: dict[tuple[int, int], dict[str, Any]],
    src: int,
    dst: int,
    method: str,
    config: CompatibilityPlannerConfig,
) -> tuple[list[int] | None, float]:
    index = _planning_index(edge_table)
    pair_coverage = _pair_coverage_lookup(pair_map, config)
    edge_costs = _method_edge_costs(index["rows_by_id"], method, config)
    return _line_graph_shortest_path_index(index, pair_coverage, src, dst, method, config, edge_costs)


def _line_graph_shortest_path_index(
    index: dict[str, Any],
    pair_coverage: dict[tuple[int, int], float],
    src: int,
    dst: int,
    method: str,
    config: CompatibilityPlannerConfig,
    edge_costs: dict[int, float],
) -> tuple[list[int] | None, float]:
    if src == dst:
        return [], 0.0
    rows_by_id: dict[int, dict[str, Any]] = index["rows_by_id"]
    by_src: dict[int, list[int]] = index["by_src"]

    dist: dict[int, float] = {}
    prev: dict[int, int | None] = {}
    heap: list[tuple[float, int]] = []
    for edge_id in by_src.get(int(src), []):
        cost = float(edge_costs[edge_id])
        dist[edge_id] = cost
        prev[edge_id] = None
        heappush(heap, (cost, edge_id))

    best_final: int | None = None
    while heap:
        cur_cost, edge_id = heappop(heap)
        if cur_cost > dist.get(edge_id, np.inf) + EPS:
            continue
        row = rows_by_id[edge_id]
        if int(row["dst"]) == int(dst):
            best_final = edge_id
            break
        for next_edge_id in by_src.get(int(row["dst"]), []):
            coverage = pair_coverage.get(_pair_key(edge_id, next_edge_id), _clip01(config.missing_pair_coverage))
            if _method_uses_pair_threshold(method) and coverage < float(config.min_pair_coverage):
                continue
            step_cost = float(edge_costs[next_edge_id])
            if _method_uses_pair_penalty(method):
                step_cost += float(config.pair_weight) * (1.0 - coverage)
            new_cost = cur_cost + step_cost
            if new_cost + EPS < dist.get(next_edge_id, np.inf):
                dist[next_edge_id] = new_cost
                prev[next_edge_id] = edge_id
                heappush(heap, (new_cost, next_edge_id))

    if best_final is None:
        return None, np.nan
    return _path_from_prev(best_final, prev), float(dist[best_final])


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else np.nan


def _fraction(flags: list[bool]) -> float:
    return float(np.mean(flags)) if flags else np.nan


def _path_nodes(edge_rows: list[dict[str, Any]], src: int) -> list[int]:
    if not edge_rows:
        return [int(src)]
    nodes = [int(edge_rows[0]["src"])]
    nodes.extend(int(row["dst"]) for row in edge_rows)
    return nodes


def _path_metric_row(
    method: str,
    edge_table: pd.DataFrame,
    pair_map: dict[tuple[int, int], dict[str, Any]],
    query_id: int,
    src: int,
    dst: int,
    path_edge_ids: list[int] | None,
    planning_cost: float,
    config: CompatibilityPlannerConfig,
    edge_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "query_id": int(query_id),
        "src": int(src),
        "dst": int(dst),
        "reachable": False,
        "path_length": 0,
        "num_adjacent_edge_pairs": 0,
        "planning_path_cost": np.nan,
        "base_path_cost": np.nan,
        "pair_penalty_cost": 0.0,
        "mean_edge_proxy_score": np.nan,
        "min_edge_proxy_score": np.nan,
        "mean_heldout_support_lcb": np.nan,
        "min_heldout_support_lcb": np.nan,
        "mean_edge_policy_support_score": np.nan,
        "mean_edge_ood_score": np.nan,
        "mean_pair_termination_bridge_coverage": np.nan,
        "min_pair_termination_bridge_coverage": np.nan,
        "mean_pair_bridge_matches_per_first_segment": np.nan,
        "pair_strict_compatible_rate": np.nan,
        "pair_incompatible_fraction": np.nan,
        "low_pair_coverage_fraction": np.nan,
        "missing_pair_fraction": np.nan,
        "uncertified_edge_fraction": np.nan,
        "original_uncertified_edge_fraction": np.nan,
        "high_pair_risk_fraction": np.nan,
        "proxy_path_success": 0.0,
        "path_nodes": "",
        "path_edge_ids": "",
    }
    if path_edge_ids is None:
        return row

    by_id = edge_index["rows_by_id"] if edge_index is not None else {
        int(edge.edge_id): edge._asdict() for edge in edge_table.itertuples(index=False)
    }
    edge_rows = [by_id[int(edge_id)] for edge_id in path_edge_ids]
    pair_metrics = [
        _pair_metrics(a, b, pair_map, config)
        for a, b in zip(path_edge_ids[:-1], path_edge_ids[1:])
    ]
    pair_coverages = [_clip01(m["termination_bridge_coverage"]) for m in pair_metrics]
    pair_match_rates = [max(0.0, _as_float(m["bridge_matches_per_first_segment"], 0.0)) for m in pair_metrics]
    strict_pairs = [_as_bool(m["strict_compatible"]) for m in pair_metrics]
    missing_pairs = [_as_bool(m["missing_pair"]) for m in pair_metrics]
    proxies = [_clip01(rec.get("edge_proxy_score", 0.0)) for rec in edge_rows]
    support_lcbs = [_clip01(rec.get("heldout_support_lcb", 0.0)) for rec in edge_rows]
    policy_scores = [_clip01(rec.get("edge_policy_support_score", 0.0)) for rec in edge_rows]
    ood_scores = [max(0.0, _as_float(rec.get("edge_ood_score", 1.0), 1.0)) for rec in edge_rows]
    certified = [_as_bool(rec.get("certified_offline_binary", False)) for rec in edge_rows]
    original_certified = [
        _as_bool(rec.get("certified_offline_binary_original", rec.get("certified_offline_binary", False)))
        for rec in edge_rows
    ]
    pair_penalty = float(sum(float(config.pair_weight) * (1.0 - v) for v in pair_coverages))
    nodes = _path_nodes(edge_rows, src)

    row.update(
        {
            "reachable": True,
            "path_length": int(len(path_edge_ids)),
            "num_adjacent_edge_pairs": int(max(0, len(path_edge_ids) - 1)),
            "planning_path_cost": float(planning_cost),
            "base_path_cost": float(
                sum(max(EPS, _as_float(rec.get("base_cost", rec.get("cost", 1.0)), 1.0)) for rec in edge_rows)
            ),
            "pair_penalty_cost": pair_penalty,
            "mean_edge_proxy_score": _mean(proxies),
            "min_edge_proxy_score": float(np.min(proxies)) if proxies else np.nan,
            "mean_heldout_support_lcb": _mean(support_lcbs),
            "min_heldout_support_lcb": float(np.min(support_lcbs)) if support_lcbs else np.nan,
            "mean_edge_policy_support_score": _mean(policy_scores),
            "mean_edge_ood_score": _mean(ood_scores),
            "mean_pair_termination_bridge_coverage": _mean(pair_coverages),
            "min_pair_termination_bridge_coverage": float(np.min(pair_coverages)) if pair_coverages else np.nan,
            "mean_pair_bridge_matches_per_first_segment": _mean(pair_match_rates),
            "pair_strict_compatible_rate": _fraction(strict_pairs),
            "pair_incompatible_fraction": _fraction([not x for x in strict_pairs]),
            "low_pair_coverage_fraction": _fraction(
                [v < float(config.min_pair_coverage) for v in pair_coverages]
            ),
            "missing_pair_fraction": _fraction(missing_pairs),
            "uncertified_edge_fraction": _fraction([not x for x in certified]),
            "original_uncertified_edge_fraction": _fraction([not x for x in original_certified]),
            "high_pair_risk_fraction": _fraction(
                [(1.0 - v) > float(config.high_pair_risk_threshold) for v in pair_coverages]
            ),
            "proxy_path_success": float(np.prod(proxies)) if proxies else 0.0,
            "path_nodes": " ".join(str(int(x)) for x in nodes),
            "path_edge_ids": " ".join(str(int(x)) for x in path_edge_ids),
        }
    )
    return row


def evaluate_compatibility_planning_methods(
    edge_table: pd.DataFrame,
    pair_compatibility: pd.DataFrame,
    path_queries: pd.DataFrame,
    methods: list[str] | tuple[str, ...],
    config: CompatibilityPlannerConfig | None = None,
    max_queries: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate support-only planners with transition-dependent compatibility costs."""

    config = config or CompatibilityPlannerConfig()
    pair_map = _pair_lookup(pair_compatibility)
    pair_coverage = _pair_coverage_lookup(pair_map, config)
    edge_index = _planning_index(edge_table)
    queries = _parse_queries(path_queries, max_queries=max_queries, seed=seed)
    rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    valid_methods = {
        "support_shortest_path",
        "calibrated_edge_penalized",
        "compat_penalized",
        "calibrated_compat_penalized",
        "compat_threshold",
        "calibrated_compat_threshold",
    }
    for method in methods:
        if method not in valid_methods:
            raise ValueError(f"Unsupported compatibility planning method: {method}")
        graph_rows.append(
            {
                "method": method,
                "num_graph_nodes": int(len(edge_index["nodes"])),
                "num_graph_edges": int(edge_table.shape[0]),
                "num_pair_compatibility_edges": int(pair_compatibility.shape[0]),
            }
        )
        edge_costs = _method_edge_costs(edge_index["rows_by_id"], method, config)
        for query in queries.itertuples(index=False):
            src = int(query.start_cluster)
            dst = int(query.goal_cluster)
            edge_path, cost = _line_graph_shortest_path_index(
                index=edge_index,
                pair_coverage=pair_coverage,
                src=src,
                dst=dst,
                method=method,
                config=config,
                edge_costs=edge_costs,
            )
            rows.append(
                _path_metric_row(
                    method=method,
                    edge_table=edge_table,
                    pair_map=pair_map,
                    query_id=int(query.query_id),
                    src=src,
                    dst=dst,
                    path_edge_ids=edge_path,
                    planning_cost=cost,
                    config=config,
                    edge_index=edge_index,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(graph_rows)


def summarize_compatibility_planning(
    path_metrics: pd.DataFrame,
    graph_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if path_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, group in path_metrics.groupby("method", sort=False):
        reachable = group[group["reachable"]].copy()
        row = {
            "method": method,
            "num_queries": int(group.shape[0]),
            "num_reachable": int(reachable.shape[0]),
            "path_coverage": float(reachable.shape[0] / max(1, group.shape[0])),
            "mean_path_length": float(reachable["path_length"].mean()) if not reachable.empty else np.nan,
            "mean_planning_path_cost": float(reachable["planning_path_cost"].mean())
            if not reachable.empty
            else np.nan,
            "mean_base_path_cost": float(reachable["base_path_cost"].mean()) if not reachable.empty else np.nan,
            "mean_pair_penalty_cost": float(reachable["pair_penalty_cost"].mean())
            if not reachable.empty
            else np.nan,
            "mean_edge_proxy_score": float(reachable["mean_edge_proxy_score"].mean())
            if not reachable.empty
            else np.nan,
            "mean_min_edge_proxy_score": float(reachable["min_edge_proxy_score"].mean())
            if not reachable.empty
            else np.nan,
            "mean_heldout_support_lcb": float(reachable["mean_heldout_support_lcb"].mean())
            if not reachable.empty
            else np.nan,
            "mean_min_heldout_support_lcb": float(reachable["min_heldout_support_lcb"].mean())
            if not reachable.empty
            else np.nan,
            "mean_edge_policy_support_score": float(reachable["mean_edge_policy_support_score"].mean())
            if not reachable.empty
            else np.nan,
            "mean_edge_ood_score": float(reachable["mean_edge_ood_score"].mean())
            if not reachable.empty
            else np.nan,
            "mean_pair_termination_bridge_coverage": float(
                reachable["mean_pair_termination_bridge_coverage"].mean()
            )
            if not reachable.empty
            else np.nan,
            "mean_min_pair_termination_bridge_coverage": float(
                reachable["min_pair_termination_bridge_coverage"].mean()
            )
            if not reachable.empty
            else np.nan,
            "mean_pair_bridge_matches_per_first_segment": float(
                reachable["mean_pair_bridge_matches_per_first_segment"].mean()
            )
            if not reachable.empty
            else np.nan,
            "mean_pair_strict_compatible_rate": float(reachable["pair_strict_compatible_rate"].mean())
            if not reachable.empty
            else np.nan,
            "mean_pair_incompatible_fraction": float(reachable["pair_incompatible_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_low_pair_coverage_fraction": float(reachable["low_pair_coverage_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_missing_pair_fraction": float(reachable["missing_pair_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_uncertified_edge_fraction": float(reachable["uncertified_edge_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_original_uncertified_edge_fraction": float(
                reachable["original_uncertified_edge_fraction"].mean()
            )
            if not reachable.empty
            else np.nan,
            "mean_high_pair_risk_fraction": float(reachable["high_pair_risk_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_proxy_path_success_reachable": float(reachable["proxy_path_success"].mean())
            if not reachable.empty
            else 0.0,
            "mean_proxy_path_success_over_all": float(group["proxy_path_success"].mean()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if graph_metrics is not None and not graph_metrics.empty:
        out = out.merge(graph_metrics, on="method", how="left")
    return add_compatibility_baseline_deltas(out, path_metrics)


def add_compatibility_baseline_deltas(summary: pd.DataFrame, path_metrics: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "support_shortest_path" not in set(summary["method"]):
        return summary
    out = summary.copy()
    support = path_metrics[path_metrics["method"] == "support_shortest_path"].copy()
    support_reachable = support[support["reachable"]].set_index("query_id")
    baseline_cov = float(out.loc[out["method"] == "support_shortest_path", "path_coverage"].iloc[0])
    for idx, row in out.iterrows():
        method = str(row["method"])
        current = path_metrics[path_metrics["method"] == method]
        current_reachable = current[current["reachable"]].set_index("query_id")
        common = support_reachable.index.intersection(current_reachable.index)
        out.loc[idx, "coverage_delta_vs_support"] = float(row["path_coverage"] - baseline_cov)
        out.loc[idx, "num_common_reachable_with_support"] = int(len(common))
        if len(common) == 0:
            out.loc[idx, "base_cost_delta_common_vs_support"] = np.nan
            out.loc[idx, "min_pair_coverage_delta_common_vs_support"] = np.nan
            out.loc[idx, "pair_incompatible_delta_common_vs_support"] = np.nan
            out.loc[idx, "min_proxy_delta_common_vs_support"] = np.nan
            continue
        out.loc[idx, "base_cost_delta_common_vs_support"] = float(
            current_reachable.loc[common, "base_path_cost"].mean()
            - support_reachable.loc[common, "base_path_cost"].mean()
        )
        out.loc[idx, "min_pair_coverage_delta_common_vs_support"] = float(
            current_reachable.loc[common, "min_pair_termination_bridge_coverage"].mean()
            - support_reachable.loc[common, "min_pair_termination_bridge_coverage"].mean()
        )
        out.loc[idx, "pair_incompatible_delta_common_vs_support"] = float(
            current_reachable.loc[common, "pair_incompatible_fraction"].mean()
            - support_reachable.loc[common, "pair_incompatible_fraction"].mean()
        )
        out.loc[idx, "min_proxy_delta_common_vs_support"] = float(
            current_reachable.loc[common, "min_edge_proxy_score"].mean()
            - support_reachable.loc[common, "min_edge_proxy_score"].mean()
        )
    return out


def compatibility_summary_dict(
    summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> dict[str, Any]:
    if summary.empty:
        return {"methods": [], "note": "Reset-free offline compatibility-aware planning; not rollout success."}
    best_coverage = summary.sort_values("path_coverage", ascending=False).iloc[0]["method"]
    best_pair = summary.sort_values("mean_min_pair_termination_bridge_coverage", ascending=False).iloc[0][
        "method"
    ]
    lowest_incompat = summary.sort_values("mean_pair_incompatible_fraction", ascending=True).iloc[0]["method"]
    return {
        "methods": summary["method"].astype(str).tolist(),
        "best_path_coverage_method": str(best_coverage),
        "best_min_pair_coverage_method": str(best_pair),
        "lowest_pair_incompatible_fraction_method": str(lowest_incompat),
        "pair_compatibility_summary": pair_summary.to_dict("records") if not pair_summary.empty else [],
        "method_metrics": summary.to_dict("records"),
        "note": "Reset-free offline compatibility-aware planning; not rollout success.",
    }
