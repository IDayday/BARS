from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from phase3e.compatibility_aware_planning import (
    CompatibilityPlannerConfig,
    compute_pair_compatibility_from_segments,
    evaluate_compatibility_planning_methods,
    make_compatibility_edge_table,
    summarize_compatibility_planning,
)


@dataclass(frozen=True)
class GraphRepairConfig:
    max_repair_edges: int = 500
    min_repair_support: int = 3
    min_repair_episodes: int = 2
    min_pair_coverage: float = 0.05
    bad_junction_weight: float = 3.0
    bad_endpoint_weight: float = 1.0
    support_weight: float = 0.25
    diversity_weight: float = 0.25
    short_horizon_weight: float = 0.25


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _required_edge_cols(df: pd.DataFrame, name: str) -> None:
    missing = {"edge_id", "src", "dst"}.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def bad_node_scores(pair_compatibility: pd.DataFrame, min_pair_coverage: float) -> pd.Series:
    """Score clusters by how often they sit on low-compatibility adjacent pairs."""

    if pair_compatibility.empty:
        return pd.Series(dtype=float)
    scores: dict[int, float] = {}
    for row in pair_compatibility.itertuples(index=False):
        coverage = _as_float(getattr(row, "termination_bridge_coverage", 0.0), 0.0)
        deficit = max(0.0, float(min_pair_coverage) - coverage)
        if deficit <= 0.0:
            continue
        junction = int(getattr(row, "junction"))
        scores[junction] = scores.get(junction, 0.0) + deficit
        if hasattr(row, "first_src"):
            first_src = int(getattr(row, "first_src"))
            scores[first_src] = scores.get(first_src, 0.0) + 0.25 * deficit
        if hasattr(row, "second_dst"):
            second_dst = int(getattr(row, "second_dst"))
            scores[second_dst] = scores.get(second_dst, 0.0) + 0.25 * deficit
    if not scores:
        return pd.Series(dtype=float)
    out = pd.Series(scores, dtype=float)
    max_score = float(out.max())
    return out / max(max_score, 1e-12)


def select_repair_edges(
    base_edges: pd.DataFrame,
    bank_edges: pd.DataFrame,
    pair_compatibility: pd.DataFrame,
    config: GraphRepairConfig,
) -> pd.DataFrame:
    """Select support-certified bank edges that target low-compatibility junctions."""

    _required_edge_cols(base_edges, "base_edges")
    _required_edge_cols(bank_edges, "bank_edges")
    if int(config.max_repair_edges) <= 0 or bank_edges.empty:
        return pd.DataFrame(columns=list(bank_edges.columns) + ["repair_score", "repair_reason"])

    base_pairs = {(int(row.src), int(row.dst)) for row in base_edges.itertuples(index=False)}
    base_nodes = set(base_edges["src"].astype(int)).union(set(base_edges["dst"].astype(int)))
    bad_scores = bad_node_scores(pair_compatibility, config.min_pair_coverage)
    candidates = bank_edges.copy()
    candidates["src"] = candidates["src"].astype(int)
    candidates["dst"] = candidates["dst"].astype(int)
    candidates["edge_id"] = candidates["edge_id"].astype(int)
    candidates = candidates[
        ~candidates.apply(lambda r: (int(r["src"]), int(r["dst"])) in base_pairs, axis=1)
    ].copy()
    if "num_segments" in candidates.columns:
        candidates = candidates[candidates["num_segments"] >= int(config.min_repair_support)]
    if "num_episodes" in candidates.columns:
        candidates = candidates[candidates["num_episodes"] >= int(config.min_repair_episodes)]
    if candidates.empty:
        return pd.DataFrame(columns=list(bank_edges.columns) + ["repair_score", "repair_reason"])

    src_bad = candidates["src"].map(lambda x: float(bad_scores.get(int(x), 0.0)))
    dst_bad = candidates["dst"].map(lambda x: float(bad_scores.get(int(x), 0.0)))
    touches_base = candidates["src"].isin(base_nodes).astype(float) + candidates["dst"].isin(base_nodes).astype(float)
    support = np.log1p(pd.to_numeric(candidates.get("num_segments", 0.0), errors="coerce").fillna(0.0))
    starts = np.log1p(pd.to_numeric(candidates.get("num_unique_starts", 0.0), errors="coerce").fillna(0.0))
    episodes = np.log1p(pd.to_numeric(candidates.get("num_episodes", 0.0), errors="coerce").fillna(0.0))
    median_h = pd.to_numeric(candidates.get("median_h", 1.0), errors="coerce").fillna(1.0)
    max_support = max(float(support.max()), 1e-12)
    max_diversity = max(float((starts + episodes).max()), 1e-12)
    horizon_score = 1.0 / median_h.clip(lower=1.0)
    candidates["repair_score"] = (
        float(config.bad_junction_weight) * (src_bad + dst_bad)
        + float(config.bad_endpoint_weight) * touches_base
        + float(config.support_weight) * (support / max_support)
        + float(config.diversity_weight) * ((starts + episodes) / max_diversity)
        + float(config.short_horizon_weight) * horizon_score
    )
    candidates["repair_reason"] = np.where(
        (src_bad + dst_bad) > 0.0,
        "low_compatibility_junction",
        "support_bank_connector",
    )
    return candidates.sort_values(
        ["repair_score", "num_segments" if "num_segments" in candidates.columns else "edge_id"],
        ascending=[False, False],
        kind="mergesort",
    ).head(int(config.max_repair_edges)).reset_index(drop=True)


def _segment_frame(edge_segments: dict[str, np.ndarray]) -> pd.DataFrame:
    required = ["edge_id", "ep_id", "global_i", "global_j"]
    missing = [key for key in required if key not in edge_segments]
    if missing:
        raise ValueError(f"edge_segments missing required arrays: {missing}")
    data = {key: np.asarray(edge_segments[key]) for key in edge_segments}
    return pd.DataFrame(data)


def build_augmented_graph_inputs(
    base_edges: pd.DataFrame,
    base_segments: dict[str, np.ndarray],
    repair_edges: pd.DataFrame,
    bank_segments: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Append selected support-bank repair edges and reassign edge ids."""

    _required_edge_cols(base_edges, "base_edges")
    _required_edge_cols(repair_edges, "repair_edges")
    base = base_edges.copy()
    base["edge_id"] = base["edge_id"].astype(int)
    if repair_edges.empty:
        return base, {key: np.asarray(value) for key, value in base_segments.items()}, pd.DataFrame()

    start_edge_id = int(base["edge_id"].max()) + 1 if not base.empty else 0
    repair = repair_edges.copy().reset_index(drop=True)
    repair["bank_edge_id"] = repair["edge_id"].astype(int)
    repair["edge_id"] = np.arange(start_edge_id, start_edge_id + repair.shape[0], dtype=np.int64)
    repair["is_repair_edge"] = True
    if "repair_score" not in repair.columns:
        repair["repair_score"] = np.nan
    if "repair_reason" not in repair.columns:
        repair["repair_reason"] = "manual_repair_edge"
    base["is_repair_edge"] = False
    for col in repair.columns:
        if col not in base.columns:
            base[col] = np.nan
    for col in base.columns:
        if col not in repair.columns:
            repair[col] = np.nan
    augmented_edges = pd.concat([base[repair.columns], repair], ignore_index=True)

    base_seg_df = _segment_frame(base_segments)
    bank_seg_df = _segment_frame(bank_segments)
    id_map = dict(zip(repair["bank_edge_id"].astype(int), repair["edge_id"].astype(int)))
    repair_seg_df = bank_seg_df[bank_seg_df["edge_id"].astype(int).isin(id_map)].copy()
    repair_seg_df["edge_id"] = repair_seg_df["edge_id"].astype(int).map(id_map).astype(np.int64)
    all_cols = sorted(set(base_seg_df.columns).union(set(repair_seg_df.columns)))
    for col in all_cols:
        if col not in base_seg_df.columns:
            base_seg_df[col] = 0
        if col not in repair_seg_df.columns:
            repair_seg_df[col] = 0
    seg_df = pd.concat([base_seg_df[all_cols], repair_seg_df[all_cols]], ignore_index=True)
    edge_segments = {col: seg_df[col].to_numpy() for col in all_cols}
    repair_map = repair[["edge_id", "bank_edge_id", "src", "dst", "repair_score", "repair_reason"]].copy()
    return augmented_edges, edge_segments, repair_map


def run_repair_evaluation(
    base_edges: pd.DataFrame,
    base_segments: dict[str, np.ndarray],
    bank_edges: pd.DataFrame,
    bank_segments: dict[str, np.ndarray],
    path_queries: pd.DataFrame,
    certification: pd.DataFrame | None,
    repair_config: GraphRepairConfig,
    planner_config: CompatibilityPlannerConfig,
    methods: list[str],
    H_intra: int,
    max_queries: int | None = None,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Select repair edges, recompute compatibility, and compare base vs repaired graphs."""

    base_pair_summary, base_pair_df = compute_pair_compatibility_from_segments(
        base_edges,
        base_segments,
        H_intra=H_intra,
    )
    repair_edges = select_repair_edges(base_edges, bank_edges, base_pair_df, repair_config)
    augmented_edges, augmented_segments, repair_map = build_augmented_graph_inputs(
        base_edges,
        base_segments,
        repair_edges,
        bank_segments,
    )
    augmented_pair_summary, augmented_pair_df = compute_pair_compatibility_from_segments(
        augmented_edges,
        augmented_segments,
        H_intra=H_intra,
    )

    base_edge_table = make_compatibility_edge_table(base_edges, certification)
    repaired_edge_table = make_compatibility_edge_table(augmented_edges, certification)
    base_paths, base_graph = evaluate_compatibility_planning_methods(
        base_edge_table,
        base_pair_df,
        path_queries,
        methods=methods,
        config=planner_config,
        max_queries=max_queries,
        seed=seed,
    )
    repaired_paths, repaired_graph = evaluate_compatibility_planning_methods(
        repaired_edge_table,
        augmented_pair_df,
        path_queries,
        methods=methods,
        config=planner_config,
        max_queries=max_queries,
        seed=seed,
    )
    base_paths["graph_variant"] = "base"
    repaired_paths["graph_variant"] = "repaired"
    base_graph["graph_variant"] = "base"
    repaired_graph["graph_variant"] = "repaired"

    path_metrics = pd.concat([base_paths, repaired_paths], ignore_index=True)
    base_summary = summarize_compatibility_planning(base_paths, base_graph)
    repaired_summary = summarize_compatibility_planning(repaired_paths, repaired_graph)
    base_summary["graph_variant"] = "base"
    repaired_summary["graph_variant"] = "repaired"
    summary = pd.concat([base_summary, repaired_summary], ignore_index=True)
    summary = add_repair_deltas(summary)

    repair_summary = summarize_repair_edges(repair_edges, repair_map)
    base_pair_summary["graph_variant"] = "base"
    augmented_pair_summary["graph_variant"] = "repaired"
    pair_summary = pd.concat([base_pair_summary, augmented_pair_summary], ignore_index=True)

    return {
        "repair_edges": repair_edges,
        "repair_edge_map": repair_map,
        "augmented_edges": augmented_edges,
        "base_pair_compatibility": base_pair_df,
        "augmented_pair_compatibility": augmented_pair_df,
        "pair_summary": pair_summary,
        "path_metrics": path_metrics,
        "summary": summary,
        "repair_summary": repair_summary,
    }


def summarize_repair_edges(repair_edges: pd.DataFrame, repair_map: pd.DataFrame) -> pd.DataFrame:
    if repair_edges.empty:
        return pd.DataFrame(
            [
                {
                    "num_repair_edges": 0,
                    "num_repair_nodes": 0,
                    "mean_repair_score": np.nan,
                    "median_repair_support": np.nan,
                    "mean_repair_median_h": np.nan,
                }
            ]
        )
    nodes = set(repair_edges["src"].astype(int)).union(set(repair_edges["dst"].astype(int)))
    return pd.DataFrame(
        [
            {
                "num_repair_edges": int(repair_edges.shape[0]),
                "num_repair_nodes": int(len(nodes)),
                "num_mapped_repair_edges": int(repair_map.shape[0]),
                "mean_repair_score": float(repair_edges["repair_score"].mean()),
                "median_repair_support": float(
                    pd.to_numeric(repair_edges.get("num_segments", np.nan), errors="coerce").median()
                ),
                "mean_repair_median_h": float(
                    pd.to_numeric(repair_edges.get("median_h", np.nan), errors="coerce").mean()
                ),
            }
        ]
    )


def add_repair_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    for method in out["method"].dropna().unique():
        mask_base = (out["method"] == method) & (out["graph_variant"] == "base")
        mask_repaired = (out["method"] == method) & (out["graph_variant"] == "repaired")
        if not mask_base.any() or not mask_repaired.any():
            continue
        base = out[mask_base].iloc[0]
        for idx in out[mask_repaired].index:
            out.loc[idx, "coverage_delta_vs_base_graph"] = float(
                out.loc[idx, "path_coverage"] - base["path_coverage"]
            )
            out.loc[idx, "min_pair_coverage_delta_vs_base_graph"] = float(
                out.loc[idx, "mean_min_pair_termination_bridge_coverage"]
                - base["mean_min_pair_termination_bridge_coverage"]
            )
            out.loc[idx, "pair_incompatible_delta_vs_base_graph"] = float(
                out.loc[idx, "mean_pair_incompatible_fraction"] - base["mean_pair_incompatible_fraction"]
            )
            out.loc[idx, "base_cost_delta_vs_base_graph"] = float(
                out.loc[idx, "mean_base_path_cost"] - base["mean_base_path_cost"]
            )
    return out
