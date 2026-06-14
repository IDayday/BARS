from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from phase3e.compatibility_aware_planning import (
    CompatibilityPlannerConfig,
    evaluate_compatibility_planning_methods,
    make_compatibility_edge_table,
    summarize_compatibility_planning,
)
from phase3e.edge_risk_calibration import (
    EdgeRiskCalibrationConfig,
    calibrate_edge_risk,
    make_planner_certification,
    score_diagnostics,
)


EPS = 1e-12


@dataclass(frozen=True)
class RepairCertificationConfig:
    support_floor: float = 0.02
    support_scale: float = 0.85
    endpoint_fallback_quantile: float = 0.25
    policy_transfer_scale: float = 0.85
    compatibility_floor: float = 0.02
    certification_source: str = "repair_transfer_proxy"


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(np.clip(out, 0.0, 1.0))


def _series_clip01(values: Any, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(default).clip(0.0, 1.0).astype(float)


def _rank01(values: Any) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if series.notna().sum() <= 1:
        return pd.Series(np.ones(series.shape[0], dtype=np.float64), index=series.index)
    return series.rank(method="average", pct=True).fillna(0.0).clip(0.0, 1.0).astype(float)


def _bool_series(values: Any) -> pd.Series:
    series = pd.Series(values)
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.map(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}).fillna(False).astype(bool)


def _pair_context(pair_compatibility: pd.DataFrame) -> pd.DataFrame:
    if pair_compatibility.empty:
        return pd.DataFrame(
            columns=[
                "edge_id",
                "outgoing_mean_termination_bridge_coverage",
                "incoming_mean_termination_bridge_coverage",
                "outgoing_incompatible_fraction",
                "incoming_incompatible_fraction",
                "outgoing_num_pair_contexts",
                "incoming_num_pair_contexts",
            ]
        )
    pair_df = pair_compatibility.copy()
    pair_df["termination_bridge_coverage"] = _series_clip01(
        pair_df.get("termination_bridge_coverage", 0.0)
    )
    pair_df["incompatible"] = ~_bool_series(pair_df.get("strict_compatible", False))
    outgoing = (
        pair_df.groupby("edge_id_first", sort=False)
        .agg(
            outgoing_mean_termination_bridge_coverage=("termination_bridge_coverage", "mean"),
            outgoing_incompatible_fraction=("incompatible", "mean"),
            outgoing_num_pair_contexts=("termination_bridge_coverage", "size"),
        )
        .rename_axis("edge_id")
        .reset_index()
    )
    incoming = (
        pair_df.groupby("edge_id_second", sort=False)
        .agg(
            incoming_mean_termination_bridge_coverage=("termination_bridge_coverage", "mean"),
            incoming_incompatible_fraction=("incompatible", "mean"),
            incoming_num_pair_contexts=("termination_bridge_coverage", "size"),
        )
        .rename_axis("edge_id")
        .reset_index()
    )
    return outgoing.merge(incoming, on="edge_id", how="outer")


def _endpoint_policy_transfer(base_certification: pd.DataFrame, edge: pd.Series, config: RepairCertificationConfig) -> float:
    if base_certification.empty or "edge_policy_support_score" not in base_certification.columns:
        return float(config.support_floor)
    base = base_certification.copy()
    base["edge_policy_support_score"] = _series_clip01(base["edge_policy_support_score"])
    src = int(edge["src"])
    dst = int(edge["dst"])
    same_endpoint = base[
        (base.get("src", -1).astype(int) == src)
        | (base.get("dst", -1).astype(int) == dst)
        | (base.get("dst", -1).astype(int) == src)
        | (base.get("src", -1).astype(int) == dst)
    ]
    if not same_endpoint.empty:
        value = float(same_endpoint["edge_policy_support_score"].median())
    else:
        q = float(np.clip(config.endpoint_fallback_quantile, 0.0, 1.0))
        value = float(base["edge_policy_support_score"].quantile(q))
    return _clip01(value * float(config.policy_transfer_scale), default=config.support_floor)


def _repair_support_scores(augmented_edges: pd.DataFrame, repair_mask: pd.Series, config: RepairCertificationConfig) -> pd.DataFrame:
    repair = augmented_edges[repair_mask].copy()
    if repair.empty:
        return pd.DataFrame(index=repair.index)
    all_support_rank = _rank01(np.log1p(pd.to_numeric(augmented_edges.get("num_segments", 0), errors="coerce").fillna(0.0)))
    all_start_rank = _rank01(np.log1p(pd.to_numeric(augmented_edges.get("num_unique_starts", 0), errors="coerce").fillna(0.0)))
    all_episode_rank = _rank01(np.log1p(pd.to_numeric(augmented_edges.get("num_episodes", 0), errors="coerce").fillna(0.0)))
    out = pd.DataFrame(index=repair.index)
    support_rank = all_support_rank.loc[repair.index]
    diversity_rank = ((all_start_rank.loc[repair.index] + all_episode_rank.loc[repair.index]) / 2.0).clip(0.0, 1.0)
    out["repair_support_rank"] = support_rank
    out["repair_diversity_rank"] = diversity_rank
    out["repair_support_proxy_lcb"] = (
        float(config.support_scale)
        * np.sqrt(np.maximum(EPS, support_rank.to_numpy()) * np.maximum(EPS, diversity_rank.to_numpy()))
    )
    out["repair_support_proxy_lcb"] = out["repair_support_proxy_lcb"].clip(
        lower=float(config.support_floor),
        upper=1.0,
    )
    endpoint_in_base = (
        repair["src"].isin(augmented_edges.loc[~repair_mask, "src"]).astype(float)
        + repair["src"].isin(augmented_edges.loc[~repair_mask, "dst"]).astype(float)
        + repair["dst"].isin(augmented_edges.loc[~repair_mask, "src"]).astype(float)
        + repair["dst"].isin(augmented_edges.loc[~repair_mask, "dst"]).astype(float)
    )
    out["repair_endpoint_familiarity"] = (endpoint_in_base / 2.0).clip(0.0, 1.0)
    out["repair_behavior_reliability"] = np.cbrt(
        np.maximum(EPS, out["repair_support_rank"].to_numpy())
        * np.maximum(EPS, out["repair_diversity_rank"].to_numpy())
        * np.maximum(EPS, out["repair_endpoint_familiarity"].to_numpy())
    )
    out["repair_behavior_reliability"] = out["repair_behavior_reliability"].clip(
        lower=float(config.support_floor),
        upper=1.0,
    )
    return out


def build_repair_edge_certification(
    augmented_edges: pd.DataFrame,
    base_certification: pd.DataFrame,
    pair_compatibility: pd.DataFrame,
    repair_config: RepairCertificationConfig | None = None,
    calibration_config: EdgeRiskCalibrationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build planner certification for base and repair edges.

    Base edges retain Phase 4C certification. Repair edges receive conservative
    transfer-proxy scores from support scale, endpoint-neighbor policy scores,
    behavior support, and augmented compatibility context.
    """

    repair_config = repair_config or RepairCertificationConfig()
    calibration_config = calibration_config or EdgeRiskCalibrationConfig()
    edges = augmented_edges.copy()
    if "is_repair_edge" not in edges.columns:
        edges["is_repair_edge"] = False
    repair_mask = _bool_series(edges["is_repair_edge"])
    pair_ctx = _pair_context(pair_compatibility)

    base = base_certification.copy()
    if "edge_id" not in base.columns:
        raise ValueError("base_certification must contain edge_id")
    base["edge_id"] = base["edge_id"].astype(int)
    base["certification_source"] = base.get("certification_source", "phase4c_base")
    base["is_repair_edge"] = False

    repair_edges = edges[repair_mask].copy()
    support_scores = _repair_support_scores(edges, repair_mask, repair_config)
    repair_rows: list[dict[str, Any]] = []
    pair_ctx_by_edge = pair_ctx.set_index("edge_id") if not pair_ctx.empty else pd.DataFrame()
    for idx, edge in repair_edges.iterrows():
        edge_id = int(edge["edge_id"])
        ctx = pair_ctx_by_edge.loc[edge_id].to_dict() if edge_id in pair_ctx_by_edge.index else {}
        support = support_scores.loc[idx] if idx in support_scores.index else pd.Series(dtype=float)
        outgoing_bridge = _clip01(ctx.get("outgoing_mean_termination_bridge_coverage", repair_config.compatibility_floor))
        incoming_bridge = _clip01(ctx.get("incoming_mean_termination_bridge_coverage", repair_config.compatibility_floor))
        outgoing_incompat = _clip01(ctx.get("outgoing_incompatible_fraction", 1.0), default=1.0)
        incoming_incompat = _clip01(ctx.get("incoming_incompatible_fraction", 1.0), default=1.0)
        policy_score = _endpoint_policy_transfer(base, edge, repair_config)
        behavior_reliability = _clip01(support.get("repair_behavior_reliability", repair_config.support_floor))
        row = {
            "edge_id": edge_id,
            "bank_edge_id": int(edge["bank_edge_id"]) if "bank_edge_id" in edge and pd.notna(edge["bank_edge_id"]) else np.nan,
            "src": int(edge["src"]),
            "dst": int(edge["dst"]),
            "heldout_support_lcb": _clip01(support.get("repair_support_proxy_lcb", repair_config.support_floor)),
            "repair_support_proxy_lcb": _clip01(support.get("repair_support_proxy_lcb", repair_config.support_floor)),
            "edge_action_mse": np.nan,
            "edge_action_mse_ucb": np.nan,
            "edge_policy_support_score": policy_score,
            "obs_goal_knn_distance": np.nan,
            "action_residual_score": np.nan,
            "low_support_flag": bool(_clip01(support.get("repair_support_proxy_lcb", 0.0)) < 0.1),
            "edge_ood_score": 1.0 - behavior_reliability,
            "outgoing_mean_termination_bridge_coverage": outgoing_bridge,
            "incoming_mean_termination_bridge_coverage": incoming_bridge,
            "outgoing_incompatible_fraction": outgoing_incompat,
            "incoming_incompatible_fraction": incoming_incompat,
            "edge_proxy_score": _clip01(edge.get("edge_proxy_score", 0.0)),
            "certified_offline_binary": False,
            "certified_offline_binary_original": False,
            "edge_bottleneck_score": _clip01(edge.get("edge_bottleneck_score", 0.0)),
            "num_unique_starts": int(edge.get("num_unique_starts", 0)),
            "num_unique_episodes": int(edge.get("num_unique_episodes", edge.get("num_episodes", 0))),
            "median_h": float(edge.get("median_h", np.nan)),
            "num_segments": int(edge.get("num_segments", 0)),
            "num_episodes": int(edge.get("num_episodes", 0)),
            "heldout_support_binary": np.nan,
            "heldout_support_rate": np.nan,
            "num_policy_eval_samples": 0,
            "repair_score": float(edge.get("repair_score", np.nan)),
            "repair_reason": edge.get("repair_reason", ""),
            "repair_support_rank": _clip01(support.get("repair_support_rank", 0.0)),
            "repair_diversity_rank": _clip01(support.get("repair_diversity_rank", 0.0)),
            "repair_endpoint_familiarity": _clip01(support.get("repair_endpoint_familiarity", 0.0)),
            "certification_source": repair_config.certification_source,
            "is_repair_edge": True,
        }
        repair_rows.append(row)

    repair_cert = pd.DataFrame(repair_rows)
    combined = pd.concat([base, repair_cert], ignore_index=True, sort=False)
    combined["edge_id"] = combined["edge_id"].astype(int)
    if "edge_proxy_score_original" not in combined.columns and "edge_proxy_score" in combined.columns:
        combined["edge_proxy_score_original"] = combined["edge_proxy_score"]
    if "certified_offline_binary_original" not in combined.columns:
        combined["certified_offline_binary_original"] = combined.get("certified_offline_binary", False)
    calibrated = calibrate_edge_risk(combined, calibration_config)
    planner = make_planner_certification(calibrated)
    repair_subset = calibrated[calibrated.get("is_repair_edge", False).astype(bool)].copy()
    diagnostics = score_diagnostics(repair_subset)
    if "heldout_support_binary" not in repair_subset.columns or repair_subset["heldout_support_binary"].isna().all():
        diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if "heldout_support" not in key and "brier" not in key
        }
        diagnostics["heldout_label_available_for_repair_edges"] = False
    diagnostics.update(
        {
            "num_base_certification_edges": int(base.shape[0]),
            "num_repair_certification_edges": int(repair_cert.shape[0]),
            "note": "Repair-edge certification is a conservative offline transfer proxy, not rollout success.",
        }
    )
    return calibrated, planner, diagnostics


def add_repair_path_metrics(
    path_metrics: pd.DataFrame,
    edge_certification: pd.DataFrame,
) -> pd.DataFrame:
    if path_metrics.empty:
        return path_metrics
    cert = edge_certification.set_index("edge_id")
    out = path_metrics.copy()
    rows: list[dict[str, Any]] = []
    for row in out.itertuples(index=False):
        edge_ids = [int(x) for x in str(row.path_edge_ids).split() if x.strip()]
        if not edge_ids:
            rows.append(
                {
                    "num_repair_edges": 0,
                    "repair_edge_fraction": np.nan if not bool(row.reachable) else 0.0,
                    "repair_certified_fraction": np.nan,
                    "mean_repair_edge_proxy_score": np.nan,
                    "min_repair_edge_proxy_score": np.nan,
                }
            )
            continue
        present = cert.loc[[eid for eid in edge_ids if eid in cert.index]]
        repair = present[present.get("is_repair_edge", False).astype(bool)] if not present.empty else pd.DataFrame()
        rows.append(
            {
                "num_repair_edges": int(repair.shape[0]),
                "repair_edge_fraction": float(repair.shape[0] / max(1, len(edge_ids))),
                "repair_certified_fraction": float(
                    repair.get("certified_offline_binary", pd.Series(dtype=bool)).astype(bool).mean()
                )
                if not repair.empty
                else np.nan,
                "mean_repair_edge_proxy_score": float(
                    pd.to_numeric(repair.get("edge_proxy_score", np.nan), errors="coerce").mean()
                )
                if not repair.empty
                else np.nan,
                "min_repair_edge_proxy_score": float(
                    pd.to_numeric(repair.get("edge_proxy_score", np.nan), errors="coerce").min()
                )
                if not repair.empty
                else np.nan,
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize_repair_certified_planning(
    path_metrics: pd.DataFrame,
    graph_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    summary = summarize_compatibility_planning(path_metrics, graph_metrics)
    if summary.empty:
        return summary
    out = summary.copy()
    for method in out["method"].dropna().unique():
        reachable = path_metrics[(path_metrics["method"] == method) & (path_metrics["reachable"])]
        idx = out.index[out["method"] == method]
        if len(idx) == 0:
            continue
        idx0 = idx[0]
        out.loc[idx0, "mean_repair_edge_fraction"] = (
            float(reachable["repair_edge_fraction"].mean()) if not reachable.empty else np.nan
        )
        out.loc[idx0, "mean_num_repair_edges"] = (
            float(reachable["num_repair_edges"].mean()) if not reachable.empty else np.nan
        )
        out.loc[idx0, "mean_repair_certified_fraction"] = (
            float(reachable["repair_certified_fraction"].mean()) if not reachable.empty else np.nan
        )
        out.loc[idx0, "mean_repair_edge_proxy_score"] = (
            float(reachable["mean_repair_edge_proxy_score"].mean()) if not reachable.empty else np.nan
        )
        out.loc[idx0, "mean_min_repair_edge_proxy_score"] = (
            float(reachable["min_repair_edge_proxy_score"].mean()) if not reachable.empty else np.nan
        )
    return out


def evaluate_repair_certified_planning(
    augmented_edges: pd.DataFrame,
    pair_compatibility: pd.DataFrame,
    path_queries: pd.DataFrame,
    planner_certification: pd.DataFrame,
    planner_config: CompatibilityPlannerConfig,
    methods: list[str],
    max_queries: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_table = make_compatibility_edge_table(augmented_edges, planner_certification)
    path_metrics, graph_metrics = evaluate_compatibility_planning_methods(
        edge_table=edge_table,
        pair_compatibility=pair_compatibility,
        path_queries=path_queries,
        methods=methods,
        config=planner_config,
        max_queries=max_queries,
        seed=seed,
    )
    path_metrics = add_repair_path_metrics(path_metrics, planner_certification)
    summary = summarize_repair_certified_planning(path_metrics, graph_metrics)
    return path_metrics, summary
