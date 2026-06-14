from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class RiskPlannerConfig:
    risk_weight: float = 2.0
    ood_weight: float = 1.0
    incompat_weight: float = 1.0
    uncertified_weight: float = 1.0
    min_proxy_score: float = 0.25
    min_heldout_support_lcb: float = 0.01
    high_ood_threshold: float = 0.5
    high_incompat_threshold: float = 0.5


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _clip01(value: Any) -> float:
    return float(np.clip(_as_float(value, 0.0), 0.0, 1.0))


def load_edge_table(option_edges: pd.DataFrame, certification: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge Phase 2 option edges with Phase 3E certification columns."""

    if option_edges.empty:
        return pd.DataFrame()
    base = option_edges.copy()
    if "base_cost" not in base.columns:
        if "cost" in base.columns:
            base["base_cost"] = pd.to_numeric(base["cost"], errors="coerce")
        else:
            base["base_cost"] = pd.to_numeric(base["median_h"], errors="coerce")
    base["base_cost"] = base["base_cost"].fillna(pd.to_numeric(base["median_h"], errors="coerce")).fillna(1.0)

    if certification is not None and not certification.empty:
        cert_cols = [
            "edge_id",
            "heldout_support_lcb",
            "edge_action_mse",
            "edge_policy_support_score",
            "edge_ood_score",
            "outgoing_mean_termination_bridge_coverage",
            "incoming_mean_termination_bridge_coverage",
            "outgoing_incompatible_fraction",
            "incoming_incompatible_fraction",
            "edge_proxy_score",
            "certified_offline_binary",
        ]
        present = [col for col in cert_cols if col in certification.columns]
        base = base.merge(certification[present], on="edge_id", how="left", suffixes=("", "_cert"))

    defaults: dict[str, Any] = {
        "heldout_support_lcb": 0.0,
        "edge_action_mse": np.nan,
        "edge_policy_support_score": 0.0,
        "edge_ood_score": 1.0,
        "outgoing_mean_termination_bridge_coverage": 0.0,
        "incoming_mean_termination_bridge_coverage": 0.0,
        "outgoing_incompatible_fraction": 1.0,
        "incoming_incompatible_fraction": 1.0,
        "edge_proxy_score": 0.0,
        "certified_offline_binary": False,
    }
    for col, default in defaults.items():
        if col not in base.columns:
            base[col] = default
        else:
            base[col] = base[col].fillna(default)

    base["edge_proxy_score_clipped"] = base["edge_proxy_score"].map(_clip01)
    base["heldout_support_lcb_clipped"] = base["heldout_support_lcb"].map(_clip01)
    base["certified_offline_binary"] = base["certified_offline_binary"].map(_as_bool)
    return base


def risk_penalized_cost(row: Any, config: RiskPlannerConfig) -> float:
    base_cost = max(EPS, _as_float(getattr(row, "base_cost", getattr(row, "cost", 1.0)), 1.0))
    proxy = _clip01(getattr(row, "edge_proxy_score", 0.0))
    risk = 1.0 - proxy
    ood = max(0.0, _as_float(getattr(row, "edge_ood_score", 1.0), 1.0))
    incompat = _clip01(getattr(row, "outgoing_incompatible_fraction", 1.0))
    uncertified = 0.0 if _as_bool(getattr(row, "certified_offline_binary", False)) else 1.0
    multiplier = (
        1.0
        + float(config.risk_weight) * risk
        + float(config.ood_weight) * ood
        + float(config.incompat_weight) * incompat
        + float(config.uncertified_weight) * uncertified
    )
    return float(base_cost * max(EPS, multiplier))


def _edge_passes_proxy_threshold(row: Any, config: RiskPlannerConfig) -> bool:
    return bool(
        _clip01(getattr(row, "edge_proxy_score", 0.0)) >= float(config.min_proxy_score)
        and _clip01(getattr(row, "heldout_support_lcb", 0.0)) >= float(config.min_heldout_support_lcb)
    )


def build_planning_graph(
    edge_table: pd.DataFrame,
    method: str,
    config: RiskPlannerConfig | None = None,
) -> nx.DiGraph:
    """Build a support-only planning graph for one risk-aware method."""

    config = config or RiskPlannerConfig()
    method = str(method)
    graph = nx.DiGraph()
    if edge_table.empty:
        graph.graph["method"] = method
        return graph

    for row in edge_table.itertuples(index=False):
        if method == "certified_only" and not _as_bool(getattr(row, "certified_offline_binary", False)):
            continue
        if method == "proxy_threshold" and not _edge_passes_proxy_threshold(row, config):
            continue
        if method == "floor_proxy_penalized" and not _edge_passes_proxy_threshold(row, config):
            continue
        if method not in {
            "support_shortest_path",
            "certified_only",
            "proxy_threshold",
            "proxy_penalized",
            "floor_proxy_penalized",
        }:
            raise ValueError(f"Unsupported risk-aware planning method: {method}")

        planning_cost = (
            risk_penalized_cost(row, config)
            if method in {"proxy_penalized", "floor_proxy_penalized"}
            else max(EPS, _as_float(getattr(row, "base_cost", getattr(row, "cost", 1.0)), 1.0))
        )
        attrs = row._asdict()
        attrs["planning_cost"] = float(planning_cost)
        attrs["base_cost"] = max(EPS, _as_float(attrs.get("base_cost", attrs.get("cost", 1.0)), 1.0))
        graph.add_edge(int(row.src), int(row.dst), **attrs)
    graph.graph["method"] = method
    return graph


def _parse_queries(path_queries: pd.DataFrame, max_queries: int | None = None, seed: int = 0) -> pd.DataFrame:
    queries = path_queries.copy()
    if "start_cluster" not in queries.columns and {"src", "dst"} <= set(queries.columns):
        queries = queries.rename(columns={"src": "start_cluster", "dst": "goal_cluster"})
    if "query_id" not in queries.columns:
        queries["query_id"] = np.arange(queries.shape[0], dtype=np.int64)
    if max_queries is not None and queries.shape[0] > int(max_queries):
        queries = queries.sample(n=int(max_queries), random_state=int(seed)).reset_index(drop=True)
    return queries.reset_index(drop=True)


def _edge_records_for_path(graph: nx.DiGraph, path: list[int]) -> list[dict[str, Any]]:
    return [graph[int(src)][int(dst)] for src, dst in zip(path[:-1], path[1:])]


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else np.nan


def _fraction(flags: list[bool]) -> float:
    return float(np.mean(flags)) if flags else np.nan


def _path_metric_row(
    method: str,
    graph: nx.DiGraph,
    query_id: int,
    src: int,
    dst: int,
    path: list[int] | None,
    config: RiskPlannerConfig,
    bottleneck_threshold: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "query_id": int(query_id),
        "src": int(src),
        "dst": int(dst),
        "reachable": False,
        "path_length": 0,
        "planning_path_cost": np.nan,
        "base_path_cost": np.nan,
        "mean_edge_proxy_score": np.nan,
        "min_edge_proxy_score": np.nan,
        "mean_heldout_support_lcb": np.nan,
        "min_heldout_support_lcb": np.nan,
        "mean_edge_policy_support_score": np.nan,
        "mean_edge_ood_score": np.nan,
        "mean_outgoing_incompatible_fraction": np.nan,
        "mean_outgoing_termination_bridge_coverage": np.nan,
        "uncertified_edge_fraction": np.nan,
        "low_proxy_edge_fraction": np.nan,
        "low_support_lcb_edge_fraction": np.nan,
        "high_ood_edge_fraction": np.nan,
        "high_incompat_edge_fraction": np.nan,
        "bottleneck_edge_fraction": np.nan,
        "proxy_path_success": 0.0,
        "path_nodes": "",
        "path_edge_ids": "",
    }
    if not path:
        return row

    edge_records = _edge_records_for_path(graph, path)
    proxies = [_clip01(rec.get("edge_proxy_score", 0.0)) for rec in edge_records]
    support_lcbs = [_clip01(rec.get("heldout_support_lcb", 0.0)) for rec in edge_records]
    policy_scores = [_clip01(rec.get("edge_policy_support_score", 0.0)) for rec in edge_records]
    ood_scores = [max(0.0, _as_float(rec.get("edge_ood_score", 1.0), 1.0)) for rec in edge_records]
    incompat = [_clip01(rec.get("outgoing_incompatible_fraction", 1.0)) for rec in edge_records]
    bridge = [_clip01(rec.get("outgoing_mean_termination_bridge_coverage", 0.0)) for rec in edge_records]
    certified = [_as_bool(rec.get("certified_offline_binary", False)) for rec in edge_records]
    bottleneck_scores = [_as_float(rec.get("edge_bottleneck_score", np.nan), np.nan) for rec in edge_records]
    bottleneck_thr = float(bottleneck_threshold) if np.isfinite(float(bottleneck_threshold)) else np.inf

    row.update(
        {
            "reachable": True,
            "path_length": int(max(0, len(path) - 1)),
            "planning_path_cost": float(sum(_as_float(rec.get("planning_cost", 1.0), 1.0) for rec in edge_records)),
            "base_path_cost": float(sum(_as_float(rec.get("base_cost", rec.get("cost", 1.0)), 1.0) for rec in edge_records)),
            "mean_edge_proxy_score": _mean(proxies),
            "min_edge_proxy_score": float(np.min(proxies)) if proxies else np.nan,
            "mean_heldout_support_lcb": _mean(support_lcbs),
            "min_heldout_support_lcb": float(np.min(support_lcbs)) if support_lcbs else np.nan,
            "mean_edge_policy_support_score": _mean(policy_scores),
            "mean_edge_ood_score": _mean(ood_scores),
            "mean_outgoing_incompatible_fraction": _mean(incompat),
            "mean_outgoing_termination_bridge_coverage": _mean(bridge),
            "uncertified_edge_fraction": _fraction([not x for x in certified]),
            "low_proxy_edge_fraction": _fraction([v < float(config.min_proxy_score) for v in proxies]),
            "low_support_lcb_edge_fraction": _fraction(
                [v < float(config.min_heldout_support_lcb) for v in support_lcbs]
            ),
            "high_ood_edge_fraction": _fraction([v > float(config.high_ood_threshold) for v in ood_scores]),
            "high_incompat_edge_fraction": _fraction(
                [v > float(config.high_incompat_threshold) for v in incompat]
            ),
            "bottleneck_edge_fraction": _fraction(
                [np.isfinite(v) and v >= bottleneck_thr for v in bottleneck_scores]
            ),
            "proxy_path_success": float(np.prod(proxies)) if proxies else 0.0,
            "path_nodes": " ".join(str(int(x)) for x in path),
            "path_edge_ids": " ".join(str(int(rec.get("edge_id", -1))) for rec in edge_records),
        }
    )
    return row


def evaluate_planning_methods(
    edge_table: pd.DataFrame,
    path_queries: pd.DataFrame,
    methods: list[str] | tuple[str, ...],
    config: RiskPlannerConfig | None = None,
    max_queries: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate support-only planning methods on offline cluster queries."""

    config = config or RiskPlannerConfig()
    queries = _parse_queries(path_queries, max_queries=max_queries, seed=seed)
    bottleneck_values = pd.to_numeric(edge_table.get("edge_bottleneck_score", np.nan), errors="coerce")
    bottleneck_threshold = float(bottleneck_values.median()) if bottleneck_values.notna().any() else np.inf
    rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for method in methods:
        graph = build_planning_graph(edge_table, method=method, config=config)
        graph_rows.append(
            {
                "method": method,
                "num_graph_nodes": int(graph.number_of_nodes()),
                "num_graph_edges": int(graph.number_of_edges()),
            }
        )
        for query in queries.itertuples(index=False):
            src = int(query.start_cluster)
            dst = int(query.goal_cluster)
            path: list[int] | None = None
            if src in graph and dst in graph:
                try:
                    path = [int(x) for x in nx.shortest_path(graph, src, dst, weight="planning_cost")]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path = None
            rows.append(
                _path_metric_row(
                    method,
                    graph,
                    int(query.query_id),
                    src,
                    dst,
                    path,
                    config,
                    bottleneck_threshold=bottleneck_threshold,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(graph_rows)


def summarize_planning_results(path_metrics: pd.DataFrame, graph_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if path_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, group in path_metrics.groupby("method", sort=False):
        reachable = group[group["reachable"]].copy()
        row: dict[str, Any] = {
            "method": method,
            "num_queries": int(group.shape[0]),
            "num_reachable": int(reachable.shape[0]),
            "path_coverage": float(reachable.shape[0] / max(1, group.shape[0])),
            "mean_path_length": float(reachable["path_length"].mean()) if not reachable.empty else np.nan,
            "mean_planning_path_cost": float(reachable["planning_path_cost"].mean()) if not reachable.empty else np.nan,
            "mean_base_path_cost": float(reachable["base_path_cost"].mean()) if not reachable.empty else np.nan,
            "mean_edge_proxy_score": float(reachable["mean_edge_proxy_score"].mean()) if not reachable.empty else np.nan,
            "mean_min_edge_proxy_score": float(reachable["min_edge_proxy_score"].mean()) if not reachable.empty else np.nan,
            "mean_heldout_support_lcb": float(reachable["mean_heldout_support_lcb"].mean()) if not reachable.empty else np.nan,
            "mean_min_heldout_support_lcb": float(reachable["min_heldout_support_lcb"].mean()) if not reachable.empty else np.nan,
            "mean_edge_policy_support_score": float(reachable["mean_edge_policy_support_score"].mean())
            if not reachable.empty
            else np.nan,
            "mean_edge_ood_score": float(reachable["mean_edge_ood_score"].mean()) if not reachable.empty else np.nan,
            "mean_outgoing_incompatible_fraction": float(
                reachable["mean_outgoing_incompatible_fraction"].mean()
            )
            if not reachable.empty
            else np.nan,
            "mean_outgoing_termination_bridge_coverage": float(
                reachable["mean_outgoing_termination_bridge_coverage"].mean()
            )
            if not reachable.empty
            else np.nan,
            "mean_uncertified_edge_fraction": float(reachable["uncertified_edge_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_low_proxy_edge_fraction": float(reachable["low_proxy_edge_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_low_support_lcb_edge_fraction": float(reachable["low_support_lcb_edge_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_high_ood_edge_fraction": float(reachable["high_ood_edge_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_high_incompat_edge_fraction": float(reachable["high_incompat_edge_fraction"].mean())
            if not reachable.empty
            else np.nan,
            "mean_bottleneck_edge_fraction": float(reachable["bottleneck_edge_fraction"].mean())
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
    return add_support_baseline_deltas(out, path_metrics)


def add_support_baseline_deltas(summary: pd.DataFrame, path_metrics: pd.DataFrame) -> pd.DataFrame:
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
            out.loc[idx, "min_proxy_delta_common_vs_support"] = np.nan
            out.loc[idx, "uncertified_fraction_delta_common_vs_support"] = np.nan
            continue
        out.loc[idx, "base_cost_delta_common_vs_support"] = float(
            current_reachable.loc[common, "base_path_cost"].mean()
            - support_reachable.loc[common, "base_path_cost"].mean()
        )
        out.loc[idx, "min_proxy_delta_common_vs_support"] = float(
            current_reachable.loc[common, "min_edge_proxy_score"].mean()
            - support_reachable.loc[common, "min_edge_proxy_score"].mean()
        )
        out.loc[idx, "uncertified_fraction_delta_common_vs_support"] = float(
            current_reachable.loc[common, "uncertified_edge_fraction"].mean()
            - support_reachable.loc[common, "uncertified_edge_fraction"].mean()
        )
    return out


def planning_summary_dict(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"methods": [], "note": "Reset-free offline planning; not rollout success."}
    best_coverage = summary.sort_values("path_coverage", ascending=False).iloc[0]["method"]
    best_proxy = summary.sort_values("mean_min_edge_proxy_score", ascending=False).iloc[0]["method"]
    best_uncertified = summary.sort_values("mean_uncertified_edge_fraction", ascending=True).iloc[0]["method"]
    return {
        "methods": summary["method"].astype(str).tolist(),
        "best_path_coverage_method": str(best_coverage),
        "best_mean_min_edge_proxy_method": str(best_proxy),
        "lowest_uncertified_edge_fraction_method": str(best_uncertified),
        "note": "Reset-free offline planning; not rollout success.",
        "method_metrics": summary.to_dict("records"),
    }
