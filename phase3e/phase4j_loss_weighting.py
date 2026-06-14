from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RARE_GROUP_COLUMNS = [
    "bottleneck_edge_val_mse",
    "low_support_edge_val_mse",
    "long_horizon_edge_val_mse",
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


def add_loss_weight_tradeoff_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    present = [col for col in RARE_GROUP_COLUMNS if col in out.columns]
    out["rare_edge_mean_mse"] = out[present].mean(axis=1) if present else np.nan
    out["overall_to_rare_gap"] = out["rare_edge_mean_mse"] - out["final_val_action_mse"]
    return out


def aggregate_loss_weighting_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = add_loss_weight_tradeoff_metrics(rows)
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
            "rare_edge_mean_mse",
            "overall_to_rare_gap",
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


def compare_loss_weighting_to_baseline(
    summary: pd.DataFrame,
    baseline_method: str = "uniform_transition_none",
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in [
            "final_val_action_mse",
            "rare_edge_mean_mse",
            "bottleneck_edge_val_mse",
            "low_support_edge_val_mse",
            "long_horizon_edge_val_mse",
        ]
        if col in summary.columns
    ]
    rows: list[dict[str, Any]] = []
    for _, group in summary.groupby(["dataset", "phase2_run"], sort=True):
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
                value = float(getattr(row, col))
                base_value = float(base[col])
                entry[col] = value
                entry[f"{col}_baseline"] = base_value
                entry[f"{col}_delta_vs_baseline"] = value - base_value
                entry[f"{col}_ratio_vs_baseline"] = value / base_value if base_value > 0 else np.nan
            rows.append(entry)
    return pd.DataFrame(rows)


def recommend_loss_weighting_methods(
    comparisons: pd.DataFrame,
    max_overall_regret: float = 0.05,
) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, group in comparisons.groupby(["dataset", "phase2_run"], sort=True):
        eligible = group[
            group["final_val_action_mse_ratio_vs_baseline"].fillna(np.inf) <= 1.0 + float(max_overall_regret)
        ].copy()
        if eligible.empty:
            eligible = group.copy()
        eligible = eligible.sort_values(["rare_edge_mean_mse", "final_val_action_mse", "method"], kind="mergesort")
        best = eligible.iloc[0].to_dict()
        best["selection_rule"] = (
            f"lowest rare_edge_mean_mse with final_val_action_mse <= "
            f"{1.0 + float(max_overall_regret):.3f}x baseline when available"
        )
        rows.append(best)
    return pd.DataFrame(rows)


def build_phase4j_payload(
    config: dict[str, Any],
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "phase": "Phase 4J",
        "title": "Mixed/Loss-Weighted GCBC Training Study",
        "config": config,
        "aggregate_metrics": aggregate.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
        "recommendations": recommendations.to_dict("records"),
        "related_work_reviewed": [
            {
                "name": "Goal-Conditioned Supervised Learning",
                "url": "https://arxiv.org/abs/1912.06088",
                "role": "Goal-conditioned supervised policy training framing.",
            },
            {
                "name": "RvS: What is Essential for Offline RL via Supervised Learning?",
                "url": "https://arxiv.org/abs/2112.10751",
                "role": "Supervised offline RL framing and claim-boundary reminder.",
            },
            {
                "name": "Class-Balanced Loss Based on Effective Number of Samples",
                "url": "https://arxiv.org/abs/1901.05555",
                "role": "Long-tail loss reweighting motivation.",
            },
            {
                "name": "Focal Loss for Dense Object Detection",
                "url": "https://arxiv.org/abs/1708.02002",
                "role": "Loss-side focusing without changing the data provenance boundary.",
            },
        ],
        "note": "Offline supervised loss-weighting study only; not rollout success.",
    }


def write_phase4j_outputs(
    output_dir: str | Path,
    payload: dict[str, Any],
    rows: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "phase4j_loss_weighting_per_seed.csv", index=False)
    aggregate.to_csv(out / "phase4j_loss_weighting_aggregate.csv", index=False)
    comparisons.to_csv(out / "phase4j_loss_weighting_vs_baseline.csv", index=False)
    recommendations.to_csv(out / "phase4j_loss_weighting_recommendations.csv", index=False)
    with (out / "phase4j_loss_weighting_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (out / "phase4j_loss_weighting_summary.md").write_text(render_phase4j_markdown(payload), encoding="utf-8")


def render_phase4j_markdown(payload: dict[str, Any]) -> str:
    recommendations = payload.get("recommendations", [])
    comparisons = pd.DataFrame(payload.get("comparisons", []))
    lines = [
        "# Phase 4J Mixed/Loss-Weighted GCBC Training Study",
        "",
        "This phase keeps transition sampling broad and applies controlled",
        "per-edge loss weights to rare, low-support, or bottleneck edges. It is",
        "offline supervised action fitting only, not rollout success.",
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
                    f"  - final MSE: `{row.get('final_val_action_mse')}`",
                    f"  - rare-edge mean MSE: `{row.get('rare_edge_mean_mse')}`",
                    f"  - final MSE ratio vs baseline: `{row.get('final_val_action_mse_ratio_vs_baseline')}`",
                    f"  - rare-edge ratio vs baseline: `{row.get('rare_edge_mean_mse_ratio_vs_baseline')}`",
                ]
            )
    lines.extend(["", "## Baseline Comparisons", ""])
    if comparisons.empty:
        lines.append("No comparison rows were generated.")
    else:
        cols = [
            "method",
            "final_val_action_mse",
            "rare_edge_mean_mse",
            "final_val_action_mse_ratio_vs_baseline",
            "rare_edge_mean_mse_ratio_vs_baseline",
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
            "- A useful loss-weighted method should improve rare-edge mean MSE without",
            "  a large final validation MSE regression.",
            "- This study does not change graph provenance: all examples still come from",
            "  Phase 2 support-certified edge segments.",
            "- These metrics are offline proxies and do not prove option execution.",
            "",
        ]
    )
    return "\n".join(lines)
