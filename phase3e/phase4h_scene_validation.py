from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIAGNOSTIC_KEYS = [
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

METHOD_METRIC_KEYS = [
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


def load_training_summary(run_dir: str | Path) -> dict[str, Any]:
    """Load compact GCBC training metrics from a Phase 3 train output dir."""

    run_dir = Path(run_dir).expanduser()
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    summary_path = run_dir / "train_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        final = summary.get("final", {})
        if isinstance(final, dict):
            out.update({f"final_{key}": value for key, value in final.items()})
        if "num_train_segments" in summary:
            out["num_train_segments"] = summary["num_train_segments"]
    val_path = run_dir / "val_metrics.csv"
    if val_path.exists():
        val = pd.read_csv(val_path)
        if not val.empty:
            last = val.iloc[-1].to_dict()
            out["last_step"] = int(last.get("step", 0))
            if "val_action_mse" in val.columns:
                out["final_val_action_mse"] = float(last["val_action_mse"])
                out["best_val_action_mse"] = float(pd.to_numeric(val["val_action_mse"], errors="coerce").min())
            for key in [
                "bottleneck_edge_val_mse",
                "non_bottleneck_edge_val_mse",
                "high_support_edge_val_mse",
                "low_support_edge_val_mse",
                "short_horizon_edge_val_mse",
                "long_horizon_edge_val_mse",
            ]:
                if key in val.columns:
                    out[f"final_{key}"] = float(last[key])
    return out


def diagnostics_delta_frame(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    baseline_diag = baseline_summary.get("diagnostics", {})
    candidate_diag = candidate_summary.get("diagnostics", {})
    if not isinstance(baseline_diag, dict):
        baseline_diag = {}
    if not isinstance(candidate_diag, dict):
        candidate_diag = {}
    rows: list[dict[str, Any]] = []
    for key in keys or DIAGNOSTIC_KEYS:
        b = _as_float_or_none(baseline_diag.get(key))
        c = _as_float_or_none(candidate_diag.get(key))
        if b is None and c is None:
            continue
        rows.append(
            {
                "metric": key,
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "baseline_value": b,
                "candidate_value": c,
                "delta_candidate_minus_baseline": None if b is None or c is None else float(c - b),
            }
        )
    return pd.DataFrame(rows)


def method_delta_frame(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    baseline = pd.DataFrame(baseline_summary.get("method_metrics", []))
    candidate = pd.DataFrame(candidate_summary.get("method_metrics", []))
    if baseline.empty or candidate.empty or "method" not in baseline.columns or "method" not in candidate.columns:
        return pd.DataFrame()
    merged = baseline.merge(candidate, on="method", how="outer", suffixes=("_baseline", "_candidate"))
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        method = getattr(row, "method")
        for key in keys or METHOD_METRIC_KEYS:
            b = _as_float_or_none(getattr(row, f"{key}_baseline", None))
            c = _as_float_or_none(getattr(row, f"{key}_candidate", None))
            if b is None and c is None:
                continue
            rows.append(
                {
                    "method": method,
                    "metric": key,
                    "baseline_label": baseline_label,
                    "candidate_label": candidate_label,
                    "baseline_value": b,
                    "candidate_value": c,
                    "delta_candidate_minus_baseline": None if b is None or c is None else float(c - b),
                }
            )
    return pd.DataFrame(rows)


def build_phase4h_summary(
    config: dict[str, Any],
    training_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    diagnostic_deltas: pd.DataFrame,
    method_deltas: pd.DataFrame,
) -> dict[str, Any]:
    candidate_diag = candidate_summary.get("diagnostics", {})
    baseline_diag = baseline_summary.get("diagnostics", {})
    if not isinstance(candidate_diag, dict):
        candidate_diag = {}
    if not isinstance(baseline_diag, dict):
        baseline_diag = {}
    return {
        "phase": "Phase 4H",
        "title": "Stronger Scene GCBC Direct Repair Validation",
        "config": config,
        "training_summary": training_summary,
        "baseline_diagnostics": baseline_diag,
        "candidate_diagnostics": candidate_diag,
        "diagnostic_deltas": diagnostic_deltas.to_dict("records"),
        "method_deltas": method_deltas.to_dict("records"),
        "related_work_reviewed": [
            {
                "name": "Learning to Reach Goals via Iterated Supervised Learning / GCSL",
                "url": "https://arxiv.org/abs/1912.06088",
                "role": "Goal-conditioned supervised learning baseline and hindsight goal relabeling motivation.",
            },
            {
                "name": "RvS: What is Essential for Offline RL via Supervised Learning?",
                "url": "https://arxiv.org/abs/2112.10751",
                "role": "Offline RL via supervised learning motivation; does not make action MSE an execution metric.",
            },
            {
                "name": "Goal-Conditioned Supervised Learning reference implementation",
                "url": "https://github.com/dibyaghosh/gcsl",
                "role": "Open-source reference for simple goal-conditioned supervised policy training.",
            },
        ],
        "note": "This is reset-free offline supervised evidence. It is not rollout success.",
    }


def write_phase4h_outputs(
    output_dir: str | Path,
    payload: dict[str, Any],
    diagnostic_deltas: pd.DataFrame,
    method_deltas: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_deltas.to_csv(output_dir / "phase4h_diagnostic_delta.csv", index=False)
    method_deltas.to_csv(output_dir / "phase4h_method_delta.csv", index=False)
    with (output_dir / "phase4h_scene_gcbc_repair_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (output_dir / "phase4h_scene_gcbc_repair_validation_summary.md").write_text(
        render_phase4h_markdown(payload),
        encoding="utf-8",
    )


def render_phase4h_markdown(payload: dict[str, Any]) -> str:
    training = payload.get("training_summary", {})
    candidate = payload.get("candidate_diagnostics", {})
    baseline = payload.get("baseline_diagnostics", {})
    method_rows = pd.DataFrame(payload.get("method_deltas", []))
    lines = [
        "# Phase 4H Stronger Scene GCBC Direct Repair Validation",
        "",
        "Phase 4H checks whether the Scene Phase 4G direct repair-edge policy",
        "evidence survives a GCBC model trained longer than the prior 200-step",
        "smoke run. The metric remains offline supervised action fitting and is",
        "not rollout success or closed-loop option execution.",
        "",
        "## Training",
        "",
    ]
    for key in ["run_dir", "last_step", "final_val_action_mse", "best_val_action_mse", "num_train_segments"]:
        if key in training:
            lines.append(f"- `{key}`: `{training[key]}`")
    lines.extend(["", "## Direct Repair Diagnostics", ""])
    for key in DIAGNOSTIC_KEYS:
        if key in candidate or key in baseline:
            lines.append(f"- `{key}`: baseline `{baseline.get(key)}`, candidate `{candidate.get(key)}`")
    lines.extend(["", "## Planner Deltas", ""])
    if method_rows.empty:
        lines.append("No planner deltas were available.")
    else:
        selected = method_rows[
            method_rows["metric"].isin(
                [
                    "path_coverage",
                    "mean_uncertified_edge_fraction",
                    "mean_pair_incompatible_fraction",
                    "mean_min_edge_proxy_score",
                ]
            )
        ]
        for row in selected.itertuples(index=False):
            lines.append(
                f"- `{row.method}` `{row.metric}`: baseline `{row.baseline_value}`, "
                f"candidate `{row.candidate_value}`, delta `{row.delta_candidate_minus_baseline}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The comparison isolates model-training strength while reusing the same",
            "support-bank repair edges, Phase 4F calibration, and Phase 4G planning",
            "queries. Improvements in direct repair-edge MSE or certification rate",
            "make the offline proxy more credible, but still do not establish rollout",
            "success or arbitrary-reset executability.",
            "",
            "Related work reviewed: GCSL, RvS, and the GCSL reference implementation.",
            "",
        ]
    )
    return "\n".join(lines)


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
