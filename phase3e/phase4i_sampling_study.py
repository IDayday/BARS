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


def load_sampling_rows(path: str | Path) -> pd.DataFrame:
    rows = pd.read_csv(Path(path).expanduser())
    required = {"phase2_run", "sampling_mode", "seed", "final_val_action_mse"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"sampling metrics missing required columns: {sorted(missing)}")
    return rows


def add_tradeoff_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    present = [col for col in RARE_GROUP_COLUMNS if col in out.columns]
    if present:
        out["rare_edge_mean_mse"] = out[present].mean(axis=1)
    else:
        out["rare_edge_mean_mse"] = np.nan
    out["overall_to_rare_gap"] = out["rare_edge_mean_mse"] - out["final_val_action_mse"]
    return out


def aggregate_sampling_tradeoffs(rows: pd.DataFrame) -> pd.DataFrame:
    rows = add_tradeoff_metrics(rows)
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
    grouped = rows.groupby(["dataset", "phase2_run", "sampling_mode"], sort=True)
    mean = grouped[metric_cols].mean().reset_index()
    std = grouped[metric_cols].std(ddof=0).reset_index()
    out = mean.copy()
    for col in metric_cols:
        out[f"{col}_std"] = std[col]
    out["num_seeds"] = grouped.size().to_numpy(dtype=np.int64)
    return out


def compare_to_baseline(
    summary: pd.DataFrame,
    baseline_sampling_mode: str = "uniform_transition",
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "phase2_run"]
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
    for _, group in summary.groupby(group_cols, sort=True):
        baseline = group[group["sampling_mode"] == baseline_sampling_mode]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for row in group.itertuples(index=False):
            entry = {
                "dataset": getattr(row, "dataset"),
                "phase2_run": getattr(row, "phase2_run"),
                "sampling_mode": getattr(row, "sampling_mode"),
                "baseline_sampling_mode": baseline_sampling_mode,
                "num_seeds": int(getattr(row, "num_seeds", 0)),
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


def recommend_sampling_modes(
    comparisons: pd.DataFrame,
    max_overall_regret: float = 0.05,
) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, group in comparisons.groupby(["dataset", "phase2_run"], sort=True):
        candidate = group[
            group["final_val_action_mse_ratio_vs_baseline"].fillna(np.inf) <= 1.0 + float(max_overall_regret)
        ].copy()
        if candidate.empty:
            candidate = group.copy()
        candidate = candidate.sort_values(
            ["rare_edge_mean_mse", "final_val_action_mse", "sampling_mode"],
            kind="mergesort",
        )
        best = candidate.iloc[0].to_dict()
        best["selection_rule"] = (
            f"lowest rare_edge_mean_mse with final_val_action_mse <= "
            f"{1.0 + float(max_overall_regret):.3f}x baseline when available"
        )
        rows.append(best)
    return pd.DataFrame(rows)


def build_phase4i_summary_payload(
    config: dict[str, Any],
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "phase": "Phase 4I",
        "title": "Stronger GCBC Sampling Study",
        "config": config,
        "aggregate_metrics": aggregate.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
        "recommendations": recommendations.to_dict("records"),
        "related_work_reviewed": [
            {
                "name": "Learning to Reach Goals via Iterated Supervised Learning / GCSL",
                "url": "https://arxiv.org/abs/1912.06088",
                "role": "Goal-conditioned supervised learning baseline.",
            },
            {
                "name": "RvS: Offline RL via Supervised Learning",
                "url": "https://arxiv.org/abs/2112.10751",
                "role": "Supervised offline RL framing; action MSE remains a proxy metric.",
            },
            {
                "name": "Class-Balanced Loss Based on Effective Number of Samples",
                "url": "https://arxiv.org/abs/1901.05555",
                "role": "Motivation for long-tail support-aware reweighting.",
            },
            {
                "name": "Focal Loss for Dense Object Detection",
                "url": "https://arxiv.org/abs/1708.02002",
                "role": "Motivation for focusing optimization on under-served hard groups.",
            },
        ],
        "note": "Offline supervised sampling study only; not rollout success.",
    }


def write_phase4i_outputs(
    output_dir: str | Path,
    payload: dict[str, Any],
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(out / "phase4i_sampling_aggregate.csv", index=False)
    comparisons.to_csv(out / "phase4i_sampling_vs_baseline.csv", index=False)
    recommendations.to_csv(out / "phase4i_sampling_recommendations.csv", index=False)
    with (out / "phase4i_sampling_study_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (out / "phase4i_sampling_study_summary.md").write_text(render_phase4i_markdown(payload), encoding="utf-8")


def render_phase4i_markdown(payload: dict[str, Any]) -> str:
    recommendations = payload.get("recommendations", [])
    comparisons = pd.DataFrame(payload.get("comparisons", []))
    lines = [
        "# Phase 4I Stronger GCBC Sampling Study",
        "",
        "This phase evaluates whether edge-balanced sampling improves offline",
        "supervised GCBC fitting on rare, low-support, bottleneck, and",
        "long-horizon option edges. It is not rollout success.",
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
                    f"- `{row.get('phase2_run')}`: `{row.get('sampling_mode')}`",
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
            "phase2_run",
            "sampling_mode",
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
            "- Lower final validation MSE means better average held-out action fitting.",
            "- Lower rare-edge mean MSE means better fitting on bottleneck, low-support,",
            "  and long-horizon edge groups.",
            "- A sampling mode is useful only if rare-edge gains do not create a large",
            "  overall validation regression.",
            "- These metrics are offline proxies and do not prove option execution.",
            "",
        ]
    )
    return "\n".join(lines)
