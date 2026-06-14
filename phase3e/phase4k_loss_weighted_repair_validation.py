from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIRECT_DIAGNOSTIC_KEYS = [
    "mean_direct_edge_action_mse",
    "median_direct_edge_action_mse",
    "mean_direct_policy_support_score",
    "median_direct_policy_support_score",
    "direct_certified_rate",
    "transfer_certified_rate",
    "mean_policy_score_delta_direct_minus_transfer",
    "spearman_transfer_vs_direct_policy_score",
    "mean_direct_reliability",
    "mean_transfer_reliability",
]

TRAINING_KEYS = [
    "final_val_action_mse",
    "best_val_action_mse",
    "final_bottleneck_edge_val_mse",
    "final_low_support_edge_val_mse",
    "final_long_horizon_edge_val_mse",
]

PLANNER_KEYS = [
    "path_coverage",
    "mean_min_edge_proxy_score",
    "mean_uncertified_edge_fraction",
    "mean_pair_incompatible_fraction",
    "mean_repair_edge_fraction",
    "mean_repair_certified_fraction",
    "mean_base_path_cost",
]


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


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def flatten_direct_repair_summary(
    summary: dict[str, Any],
    model_record: dict[str, Any],
    training_summary: dict[str, Any] | None = None,
    planner_method: str = "calibrated_compat_threshold",
) -> dict[str, Any]:
    """Flatten one Phase 4G direct-repair summary into one Phase 4K row."""

    training_summary = training_summary or {}
    diagnostics = summary.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    method_metrics = pd.DataFrame(summary.get("method_metrics", []))
    planner_row: dict[str, Any] = {}
    if not method_metrics.empty and "method" in method_metrics.columns:
        selected = method_metrics[method_metrics["method"] == planner_method]
        if selected.empty:
            selected = method_metrics.head(1)
        planner_row = selected.iloc[0].to_dict()
    row = {
        "dataset": model_record.get("dataset"),
        "phase2_run": model_record.get("phase2_run"),
        "method": model_record.get("method"),
        "seed": model_record.get("seed"),
        "sampling_mode": model_record.get("sampling_mode"),
        "loss_weight_mode": model_record.get("loss_weight_mode"),
        "loss_weight_strength": model_record.get("loss_weight_strength"),
        "loss_weight_min": model_record.get("loss_weight_min"),
        "loss_weight_max": model_record.get("loss_weight_max"),
        "model_path": model_record.get("model_path"),
        "training_run_dir": model_record.get("training_run_dir"),
        "direct_summary_path": model_record.get("direct_summary_path"),
        "planner_method": planner_row.get("method", planner_method),
    }
    for key in DIRECT_DIAGNOSTIC_KEYS:
        row[key] = _as_float_or_nan(diagnostics.get(key))
    for key in TRAINING_KEYS:
        row[key] = _as_float_or_nan(training_summary.get(key))
    for key in PLANNER_KEYS:
        row[key] = _as_float_or_nan(planner_row.get(key))
    return row


def aggregate_phase4k_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in DIRECT_DIAGNOSTIC_KEYS + TRAINING_KEYS + PLANNER_KEYS
        if col in rows.columns
    ]
    group_cols = ["dataset", "phase2_run", "method"]
    grouped = rows.groupby(group_cols, sort=True)
    mean = grouped[metric_cols].mean().reset_index()
    std = grouped[metric_cols].std(ddof=0).reset_index()
    first_cols = [
        col
        for col in [
            "sampling_mode",
            "loss_weight_mode",
            "loss_weight_strength",
            "loss_weight_min",
            "loss_weight_max",
            "planner_method",
        ]
        if col in rows.columns
    ]
    first = grouped[first_cols].first().reset_index() if first_cols else mean[group_cols]
    out = mean.merge(first, on=group_cols, how="left")
    for col in metric_cols:
        out[f"{col}_std"] = std[col]
    out["num_seeds"] = grouped.size().to_numpy(dtype=np.int64)
    return out


def compare_phase4k_to_baseline(
    aggregate: pd.DataFrame,
    baseline_method: str = "uniform_transition_none",
) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in [
            "final_val_action_mse",
            "mean_direct_edge_action_mse",
            "mean_direct_policy_support_score",
            "direct_certified_rate",
            "mean_direct_reliability",
            "mean_min_edge_proxy_score",
            "mean_uncertified_edge_fraction",
            "mean_repair_certified_fraction",
            "mean_base_path_cost",
        ]
        if col in aggregate.columns
    ]
    rows: list[dict[str, Any]] = []
    for _, group in aggregate.groupby(["dataset", "phase2_run"], sort=True):
        baseline = group[group["method"] == baseline_method]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for row in group.itertuples(index=False):
            entry = {
                "dataset": getattr(row, "dataset"),
                "phase2_run": getattr(row, "phase2_run"),
                "method": getattr(row, "method"),
                "baseline_method": baseline_method,
                "num_seeds": int(getattr(row, "num_seeds", 0)),
                "sampling_mode": getattr(row, "sampling_mode", None),
                "loss_weight_mode": getattr(row, "loss_weight_mode", None),
                "loss_weight_strength": getattr(row, "loss_weight_strength", None),
            }
            for col in metric_cols:
                value = _as_float_or_nan(getattr(row, col))
                base_value = _as_float_or_nan(base[col])
                entry[col] = value
                entry[f"{col}_baseline"] = base_value
                entry[f"{col}_delta_vs_baseline"] = value - base_value
                entry[f"{col}_ratio_vs_baseline"] = (
                    value / base_value if np.isfinite(base_value) and base_value > 0 else np.nan
                )
            rows.append(entry)
    return pd.DataFrame(rows)


def recommend_phase4k_methods(
    comparisons: pd.DataFrame,
    max_overall_regret: float = 0.05,
    require_direct_mse_improvement: bool = True,
) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, group in comparisons.groupby(["dataset", "phase2_run"], sort=True):
        eligible = group.copy()
        if "final_val_action_mse_ratio_vs_baseline" in eligible.columns:
            eligible = eligible[
                eligible["final_val_action_mse_ratio_vs_baseline"].fillna(np.inf)
                <= 1.0 + float(max_overall_regret)
            ]
        if require_direct_mse_improvement and "mean_direct_edge_action_mse_ratio_vs_baseline" in eligible.columns:
            improved = eligible[
                eligible["mean_direct_edge_action_mse_ratio_vs_baseline"].fillna(np.inf) <= 1.0
            ]
            if not improved.empty:
                eligible = improved
        if eligible.empty:
            eligible = group.copy()
        sort_cols = [
            col
            for col in [
                "mean_direct_edge_action_mse",
                "mean_uncertified_edge_fraction",
                "final_val_action_mse",
                "method",
            ]
            if col in eligible.columns
        ]
        best = eligible.sort_values(sort_cols, kind="mergesort").iloc[0].to_dict()
        best["selection_rule"] = (
            "lowest direct repair-edge MSE among methods within overall-MSE regret "
            f"<={1.0 + float(max_overall_regret):.3f}x baseline; prefer direct-MSE improvement when available"
        )
        rows.append(best)
    return pd.DataFrame(rows)


def build_phase4k_payload(
    config: dict[str, Any],
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "phase": "Phase 4K",
        "title": "Loss-Weighted GCBC Direct Repair-Edge Validation",
        "config": config,
        "per_seed_metrics": per_seed.to_dict("records"),
        "aggregate_metrics": aggregate.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
        "recommendations": recommendations.to_dict("records"),
        "related_work_reviewed": [
            {
                "name": "Goal-Conditioned Supervised Learning",
                "url": "https://arxiv.org/abs/1912.06088",
                "role": "Goal-conditioned supervised policy framing and claim boundary.",
            },
            {
                "name": "RvS: What is Essential for Offline RL via Supervised Learning?",
                "url": "https://arxiv.org/abs/2112.10751",
                "role": "Supervised offline RL framing; offline MSE is not rollout success.",
            },
            {
                "name": "Class-Balanced Loss Based on Effective Number of Samples",
                "url": "https://arxiv.org/abs/1901.05555",
                "role": "Loss reweighting motivation for under-represented groups.",
            },
            {
                "name": "Focal Loss for Dense Object Detection",
                "url": "https://arxiv.org/abs/1708.02002",
                "role": "Loss-side focusing under imbalance without changing data provenance.",
            },
            {
                "name": "Goal-Conditioned Supervised Learning reference implementation",
                "url": "https://github.com/dibyaghosh/gcsl",
                "role": "Open-source reference for simple GCBC-style supervised policy training.",
            },
        ],
        "note": "Reset-free offline direct repair-edge supervised evidence only; not closed-loop execution.",
    }


def write_phase4k_outputs(
    output_dir: str | Path,
    payload: dict[str, Any],
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "phase4k_loss_weighted_repair_per_seed.csv", index=False)
    aggregate.to_csv(out / "phase4k_loss_weighted_repair_aggregate.csv", index=False)
    comparisons.to_csv(out / "phase4k_loss_weighted_repair_vs_baseline.csv", index=False)
    recommendations.to_csv(out / "phase4k_loss_weighted_repair_recommendations.csv", index=False)
    with (out / "phase4k_loss_weighted_repair_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (out / "phase4k_loss_weighted_repair_summary.md").write_text(render_phase4k_markdown(payload), encoding="utf-8")


def render_phase4k_markdown(payload: dict[str, Any]) -> str:
    recommendations = payload.get("recommendations", [])
    comparisons = pd.DataFrame(payload.get("comparisons", []))
    lines = [
        "# Phase 4K Loss-Weighted GCBC Direct Repair-Edge Validation",
        "",
        "Phase 4K reuses the Phase 4J GCBC checkpoints and evaluates them on",
        "the same Scene repair-bank segments used by Phase 4G/4H. This checks",
        "whether loss-weighted training improves direct repair-edge evidence,",
        "not only ordinary edge validation MSE. No environment rollout is used.",
        "",
        "## Recommendations",
        "",
    ]
    if not recommendations:
        lines.append("No recommendation rows were generated.")
    else:
        for row in recommendations:
            lines.extend(
                [
                    f"- `{row.get('phase2_run')}`: `{row.get('method')}`",
                    f"  - direct repair MSE: `{row.get('mean_direct_edge_action_mse')}`",
                    f"  - direct repair MSE ratio vs baseline: `{row.get('mean_direct_edge_action_mse_ratio_vs_baseline')}`",
                    f"  - final validation MSE ratio vs baseline: `{row.get('final_val_action_mse_ratio_vs_baseline')}`",
                    f"  - direct certified rate: `{row.get('direct_certified_rate')}`",
                ]
            )
    lines.extend(["", "## Baseline Comparisons", ""])
    if comparisons.empty:
        lines.append("No comparison rows were generated.")
    else:
        cols = [
            "method",
            "final_val_action_mse_ratio_vs_baseline",
            "mean_direct_edge_action_mse_ratio_vs_baseline",
            "direct_certified_rate",
            "mean_uncertified_edge_fraction",
            "mean_min_edge_proxy_score",
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
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- A useful method should improve direct repair-edge MSE without a large",
            "  ordinary validation-MSE regression.",
            "- Planner coverage is mostly graph-limited here; model changes mainly",
            "  affect repair-edge policy support, certification, and path risk.",
            "- Direct repair-edge action MSE is still offline supervised evidence and",
            "  does not prove option execution or online task success.",
            "",
            "Related work reviewed: GCSL, RvS, Class-Balanced Loss, Focal Loss,",
            "and the GCSL reference implementation.",
            "",
        ]
    )
    return "\n".join(lines)


def _as_float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")
