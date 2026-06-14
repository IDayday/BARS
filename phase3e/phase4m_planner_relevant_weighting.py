from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase3.train_gcbc import edge_loss_weight_values
from phase3e.phase4l_repair_group_diagnostics import add_edge_groups, parse_path_edge_ids


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _rank01(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.0, index=values.index, dtype=float)
    ranked = values.rank(method="average", pct=True)
    return ranked.fillna(0.0).clip(0.0, 1.0)


def planner_edge_usage_counts(
    path_metrics: pd.DataFrame,
    planner_method: str = "calibrated_compat_threshold",
    graph_variant: str | None = "repaired",
) -> pd.DataFrame:
    """Count how often each edge appears on reachable planner paths."""

    if path_metrics.empty or "path_edge_ids" not in path_metrics.columns:
        return pd.DataFrame(columns=["edge_id", "planner_usage_count"])
    df = path_metrics.copy()
    if "method" in df.columns:
        df = df[df["method"] == str(planner_method)]
    if graph_variant is not None and "graph_variant" in df.columns:
        df = df[df["graph_variant"] == str(graph_variant)]
    if "reachable" in df.columns:
        df = df[df["reachable"].map(_as_bool)]
    counts: Counter[int] = Counter()
    for value in df["path_edge_ids"].tolist():
        counts.update(parse_path_edge_ids(value))
    if not counts:
        return pd.DataFrame(columns=["edge_id", "planner_usage_count"])
    return pd.DataFrame(
        [{"edge_id": int(edge_id), "planner_usage_count": int(count)} for edge_id, count in sorted(counts.items())]
    )


def repair_hardness_scores(option_edges: pd.DataFrame) -> pd.Series:
    """Continuous hard-edge score from low support, high bottleneck, and long horizon."""

    edges = option_edges.copy()
    if edges.empty:
        return pd.Series(dtype=float)
    support_col = "num_unique_starts" if "num_unique_starts" in edges.columns else "num_segments"
    support_rank = _rank01(pd.to_numeric(edges.get(support_col, 1.0), errors="coerce").fillna(1.0))
    low_support = 1.0 - support_rank
    bottleneck = _rank01(pd.to_numeric(edges.get("edge_bottleneck_score", 0.0), errors="coerce").fillna(0.0))
    horizon = _rank01(pd.to_numeric(edges.get("median_h", 0.0), errors="coerce").fillna(0.0))
    score = (low_support + bottleneck + horizon) / 3.0
    score.index = edges.index
    return score.clip(0.0, 1.0)


def build_planner_relevant_loss_weights(
    option_edges: pd.DataFrame,
    path_metrics: pd.DataFrame | None = None,
    planner_method: str = "calibrated_compat_threshold",
    graph_variant: str | None = "repaired",
    base_loss_weight_mode: str = "support_bottleneck",
    base_loss_weight_strength: float = 0.3,
    base_loss_weight_min: float = 0.7,
    base_loss_weight_max: float = 1.8,
    planner_relevance_strength: float = 0.5,
    hard_repair_strength: float = 0.25,
    repair_only_planner_relevance: bool = True,
    min_weight: float = 0.7,
    max_weight: float = 2.2,
) -> pd.DataFrame:
    """Build clipped external loss weights for planner-relevant repair edges.

    The table is meant to be consumed by ``train_gcbc(..., loss_weight_mode="external")``.
    It keeps all examples support-certified; the only change is the supervised
    loss weight applied to each edge.
    """

    if option_edges.empty:
        return pd.DataFrame(columns=["edge_id", "loss_weight"])
    edges = option_edges.copy().reset_index(drop=True)
    edges["edge_id"] = edges["edge_id"].astype(int)
    if "is_repair_edge" not in edges.columns:
        edges["is_repair_edge"] = False
    repair_mask = edges["is_repair_edge"].map(_as_bool).to_numpy(dtype=bool)

    base = edge_loss_weight_values(
        edges,
        mode=base_loss_weight_mode,
        strength=float(base_loss_weight_strength),
        min_weight=float(base_loss_weight_min),
        max_weight=float(base_loss_weight_max),
    ).rename(columns={"loss_weight": "base_loss_weight"})
    out = edges[["edge_id", "is_repair_edge"]].merge(base, on="edge_id", how="left")
    out["base_loss_weight"] = pd.to_numeric(out["base_loss_weight"], errors="coerce").fillna(1.0)

    usage_input = pd.DataFrame() if path_metrics is None else path_metrics
    usage = planner_edge_usage_counts(usage_input, planner_method, graph_variant)
    out = out.merge(usage, on="edge_id", how="left")
    out["planner_usage_count"] = pd.to_numeric(out.get("planner_usage_count", 0), errors="coerce").fillna(0.0)
    if float(out["planner_usage_count"].max()) > 0.0:
        usage_score = np.log1p(out["planner_usage_count"].to_numpy(dtype=float))
        usage_score = usage_score / max(float(usage_score.max()), 1e-12)
    else:
        usage_score = np.zeros(out.shape[0], dtype=float)
    if repair_only_planner_relevance:
        usage_score = usage_score * repair_mask.astype(float)
    out["planner_relevance_score"] = usage_score

    hard_score = repair_hardness_scores(edges).reindex(edges.index).fillna(0.0).to_numpy(dtype=float)
    out["hard_repair_score"] = hard_score * repair_mask.astype(float)
    out["external_multiplier"] = (
        1.0
        + float(planner_relevance_strength) * out["planner_relevance_score"].to_numpy(dtype=float)
        + float(hard_repair_strength) * out["hard_repair_score"].to_numpy(dtype=float)
    )
    raw = out["base_loss_weight"].to_numpy(dtype=float) * out["external_multiplier"].to_numpy(dtype=float)
    if raw.size and np.isfinite(raw).all() and float(raw.mean()) > 0.0:
        raw = raw / float(raw.mean())
    lo = min(float(min_weight), float(max_weight))
    hi = max(float(min_weight), float(max_weight))
    out["loss_weight"] = np.clip(raw, lo, hi)
    out["loss_weight_reason"] = np.where(
        out["planner_relevance_score"] > 0.0,
        "planner_used_repair",
        np.where(out["hard_repair_score"] > 0.0, "hard_repair", "base_support_bottleneck"),
    )
    return out[
        [
            "edge_id",
            "loss_weight",
            "base_loss_weight",
            "external_multiplier",
            "planner_usage_count",
            "planner_relevance_score",
            "hard_repair_score",
            "is_repair_edge",
            "loss_weight_reason",
        ]
    ]


def summarize_loss_weights(option_edges: pd.DataFrame, loss_weights: pd.DataFrame) -> pd.DataFrame:
    if option_edges.empty or loss_weights.empty:
        return pd.DataFrame()
    merged = option_edges.merge(loss_weights, on="edge_id", how="left")
    merged["loss_weight"] = pd.to_numeric(merged["loss_weight"], errors="coerce").fillna(1.0)
    merged["planner_usage_count"] = pd.to_numeric(merged.get("planner_usage_count", 0), errors="coerce").fillna(0)
    merged["used_by_planner"] = merged["planner_usage_count"] > 0
    grouped = add_edge_groups(merged)
    rows: list[dict[str, Any]] = []
    for group_type in [
        "is_repair_edge",
        "support_group",
        "bottleneck_group",
        "horizon_group",
        "compatibility_group",
        "planner_usage_group",
        "loss_weight_reason",
    ]:
        if group_type not in grouped.columns:
            continue
        for group_value, group in grouped.groupby(group_type, dropna=False, sort=True):
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": str(group_value),
                    "num_edges": int(group.shape[0]),
                    "mean_loss_weight": float(group["loss_weight"].mean()),
                    "median_loss_weight": float(group["loss_weight"].median()),
                    "max_loss_weight": float(group["loss_weight"].max()),
                    "mean_planner_usage_count": float(
                        pd.to_numeric(group.get("planner_usage_count", 0), errors="coerce").fillna(0.0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_direct_repair_scores(
    direct_scores: pd.DataFrame,
    augmented_edges: pd.DataFrame,
    path_metrics: pd.DataFrame,
    planner_method: str = "calibrated_compat_threshold",
    graph_variant: str = "repaired",
) -> pd.DataFrame:
    if direct_scores.empty:
        return pd.DataFrame()
    usage = planner_edge_usage_counts(path_metrics, planner_method, graph_variant)
    metadata_cols = [
        col
        for col in [
            "edge_id",
            "src",
            "dst",
            "is_repair_edge",
            "num_unique_starts",
            "num_unique_episodes",
            "edge_bottleneck_score",
            "median_h",
            "num_segments",
            "outgoing_mean_termination_bridge_coverage",
            "incoming_mean_termination_bridge_coverage",
            "repair_reason",
        ]
        if col in augmented_edges.columns
    ]
    table = direct_scores.merge(augmented_edges[metadata_cols], on="edge_id", how="left", suffixes=("", "_edge"))
    table = table.merge(usage, on="edge_id", how="left")
    table["planner_usage_count"] = pd.to_numeric(table.get("planner_usage_count", 0), errors="coerce").fillna(0)
    table["used_by_planner"] = table["planner_usage_count"] > 0
    return add_edge_groups(table)


def grouped_direct_repair_metrics(edge_scores: pd.DataFrame) -> pd.DataFrame:
    if edge_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group_type in [
        "all_repair_edges",
        "planner_usage_group",
        "support_group",
        "bottleneck_group",
        "horizon_group",
        "compatibility_group",
        "repair_reason",
    ]:
        if group_type == "all_repair_edges":
            groups = [("all_repair_edges", edge_scores)]
        elif group_type in edge_scores.columns:
            groups = list(edge_scores.groupby(group_type, dropna=False, sort=True))
        else:
            continue
        for group_value, group in groups:
            mse = pd.to_numeric(group.get("edge_action_mse", np.nan), errors="coerce")
            samples = pd.to_numeric(group.get("num_policy_eval_samples", 1), errors="coerce").fillna(1.0)
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": str(group_value),
                    "num_edges": int(group["edge_id"].nunique()) if "edge_id" in group.columns else int(group.shape[0]),
                    "mean_edge_action_mse": float(mse.mean()) if mse.notna().any() else np.nan,
                    "median_edge_action_mse": float(mse.median()) if mse.notna().any() else np.nan,
                    "sample_weighted_edge_action_mse": float(np.average(mse.fillna(0.0), weights=samples))
                    if float(samples.sum()) > 0.0
                    else np.nan,
                    "mean_direct_policy_support_score": float(
                        pd.to_numeric(group.get("direct_edge_policy_support_score", np.nan), errors="coerce").mean()
                    ),
                    "mean_planner_usage_count": float(
                        pd.to_numeric(group.get("planner_usage_count", 0), errors="coerce").fillna(0.0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def aggregate_phase4m_training_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in [
            "final_val_action_mse",
            "best_val_action_mse",
            "bottleneck_edge_val_mse",
            "non_bottleneck_edge_val_mse",
            "high_support_edge_val_mse",
            "low_support_edge_val_mse",
            "short_horizon_edge_val_mse",
            "long_horizon_edge_val_mse",
            "direct_repair_edge_mse",
            "planner_used_repair_edge_mse",
            "not_planner_used_repair_edge_mse",
            "direct_repair_policy_support_score",
        ]
        if col in rows.columns
    ]
    group_cols = ["dataset", "phase2_run", "method"]
    grouped = rows.groupby(group_cols, sort=True)
    mean = grouped[metric_cols].mean().reset_index()
    std = grouped[metric_cols].std(ddof=0).reset_index()
    first_cols = [
        col
        for col in ["sampling_mode", "loss_weight_mode", "loss_weight_strength", "loss_weight_min", "loss_weight_max"]
        if col in rows.columns
    ]
    first = grouped[first_cols].first().reset_index() if first_cols else mean[group_cols]
    out = mean.merge(first, on=group_cols, how="left")
    for col in metric_cols:
        out[f"{col}_std"] = std[col]
    out["num_seeds"] = grouped.size().to_numpy(dtype=np.int64)
    return out


def compare_phase4m_to_baseline(aggregate: pd.DataFrame, baseline_method: str) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in [
            "final_val_action_mse",
            "direct_repair_edge_mse",
            "planner_used_repair_edge_mse",
            "not_planner_used_repair_edge_mse",
            "direct_repair_policy_support_score",
        ]
        if col in aggregate.columns
    ]
    rows: list[dict[str, Any]] = []
    for _, group in aggregate.groupby(["dataset", "phase2_run"], sort=True):
        baseline = group[group["method"] == baseline_method]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for row in group.to_dict("records"):
            entry = {
                "dataset": row.get("dataset"),
                "phase2_run": row.get("phase2_run"),
                "method": row.get("method"),
                "baseline_method": baseline_method,
                "num_seeds": row.get("num_seeds"),
            }
            for col in metric_cols:
                value = float(row.get(col, np.nan))
                base_value = float(base.get(col, np.nan))
                entry[col] = value
                entry[f"{col}_baseline"] = base_value
                entry[f"{col}_delta_vs_baseline"] = value - base_value
                entry[f"{col}_ratio_vs_baseline"] = (
                    value / base_value if np.isfinite(base_value) and base_value > 0 else np.nan
                )
            rows.append(entry)
    return pd.DataFrame(rows)


def write_phase4m_outputs(
    output_dir: str | Path,
    payload: dict[str, Any],
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    weight_summary: pd.DataFrame,
    direct_group_metrics: pd.DataFrame,
) -> None:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "phase4m_per_seed_metrics.csv", index=False)
    aggregate.to_csv(out / "phase4m_training_aggregate.csv", index=False)
    comparisons.to_csv(out / "phase4m_vs_baseline.csv", index=False)
    weight_summary.to_csv(out / "phase4m_weight_summary.csv", index=False)
    direct_group_metrics.to_csv(out / "phase4m_direct_repair_group_metrics.csv", index=False)
    with (out / "phase4m_planner_relevant_weighting_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (out / "phase4m_planner_relevant_weighting_summary.md").write_text(
        render_phase4m_markdown(payload), encoding="utf-8"
    )


def render_phase4m_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 4M Planner-Relevant Repair Loss Weighting",
        "",
        "This phase is an offline supervised GCBC loss-weighting experiment.",
        "It does not use environment rollout and does not claim edge execution",
        "success.",
        "",
        "## Question",
        "",
        "Can repair-edge loss weighting target edges that are both hard and",
        "actually used by the compatibility-aware repaired planner?",
        "",
        "## Baseline Comparisons",
        "",
    ]
    comparisons = pd.DataFrame(payload.get("comparisons", []))
    if comparisons.empty:
        lines.append("No comparison rows were generated.")
    else:
        cols = [
            "method",
            "final_val_action_mse",
            "direct_repair_edge_mse",
            "planner_used_repair_edge_mse",
            "final_val_action_mse_ratio_vs_baseline",
            "planner_used_repair_edge_mse_ratio_vs_baseline",
        ]
        present = [col for col in cols if col in comparisons.columns]
        lines.append("| " + " | ".join(present) + " |")
        lines.append("| " + " | ".join("---" for _ in present) + " |")
        for row in comparisons[present].to_dict("records"):
            values = []
            for col in present:
                value = row.get(col)
                values.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Related Work Checked", ""])
    for item in payload.get("related_work_reviewed", []):
        lines.append(f"- [{item.get('name')}]({item.get('url')}): {item.get('role')}")
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- The training examples remain support-certified offline segments.",
            "- Planner relevance is a weighting signal, not a new unsupported edge source.",
            "- Direct repair-edge action MSE is a reset-free supervised proxy, not rollout success.",
            "- The useful comparison is against the same augmented graph with ordinary",
            "  support+bottleneck loss weighting.",
            "",
        ]
    )
    return "\n".join(lines)
