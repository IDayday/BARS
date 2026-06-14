from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUP_TYPES = [
    "support_group",
    "bottleneck_group",
    "horizon_group",
    "compatibility_group",
    "planner_usage_group",
    "repair_reason",
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


def parse_path_edge_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    out: list[int] = []
    for token in text.replace(",", " ").split():
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def path_usage_counts(path_metrics: pd.DataFrame, planner_method: str) -> pd.DataFrame:
    if path_metrics.empty or "path_edge_ids" not in path_metrics.columns:
        return pd.DataFrame(columns=["edge_id", "path_usage_count"])
    df = path_metrics.copy()
    if "method" in df.columns:
        df = df[df["method"] == planner_method]
    if "reachable" in df.columns:
        df = df[df["reachable"].map(_as_bool)]
    counts: Counter[int] = Counter()
    for value in df["path_edge_ids"].tolist():
        counts.update(parse_path_edge_ids(value))
    if not counts:
        return pd.DataFrame(columns=["edge_id", "path_usage_count"])
    return pd.DataFrame(
        [{"edge_id": int(edge_id), "path_usage_count": int(count)} for edge_id, count in sorted(counts.items())]
    )


def load_run_edge_table(summary_path: str | Path, planner_method: str) -> pd.DataFrame:
    summary_path = Path(summary_path).expanduser()
    run_dir = summary_path.parent
    scores_path = run_dir / "direct_repair_edge_policy_scores.csv"
    cert_path = run_dir / "direct_repair_edge_certification.csv"
    paths_path = run_dir / "direct_repair_policy_planning_paths.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing direct scores: {scores_path}")
    scores = pd.read_csv(scores_path)
    if cert_path.exists():
        cert = pd.read_csv(cert_path)
        cert = cert[cert.get("is_repair_edge", False).map(_as_bool)].copy()
        metadata_cols = [
            col
            for col in [
                "edge_id",
                "edge_bottleneck_score",
                "num_unique_starts",
                "num_unique_episodes",
                "median_h",
                "num_segments",
                "num_episodes",
                "outgoing_mean_termination_bridge_coverage",
                "incoming_mean_termination_bridge_coverage",
                "calibrated_certified_binary",
                "calibrated_edge_reliability_score",
                "edge_proxy_score",
                "repair_reason",
            ]
            if col in cert.columns
        ]
        table = scores.merge(cert[metadata_cols], on="edge_id", how="left", suffixes=("", "_cert"))
    else:
        table = scores.copy()
    if paths_path.exists():
        usage = path_usage_counts(pd.read_csv(paths_path), planner_method=planner_method)
        table = table.merge(usage, on="edge_id", how="left")
    else:
        table["path_usage_count"] = 0
    table["path_usage_count"] = pd.to_numeric(table.get("path_usage_count", 0), errors="coerce").fillna(0).astype(int)
    table["used_by_planner"] = table["path_usage_count"] > 0
    return add_edge_groups(table)


def add_edge_groups(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    out["num_unique_starts"] = pd.to_numeric(out.get("num_unique_starts", np.nan), errors="coerce")
    out["edge_bottleneck_score"] = pd.to_numeric(out.get("edge_bottleneck_score", np.nan), errors="coerce")
    out["median_h"] = pd.to_numeric(out.get("median_h", np.nan), errors="coerce")
    compat_cols = [
        col
        for col in ["outgoing_mean_termination_bridge_coverage", "incoming_mean_termination_bridge_coverage"]
        if col in out.columns
    ]
    if compat_cols:
        out["mean_endpoint_compatibility"] = out[compat_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        out["mean_endpoint_compatibility"] = np.nan
    out["support_group"] = _low_high_group(out["num_unique_starts"], low_name="low_support", high_name="high_support")
    out["bottleneck_group"] = _low_high_group(
        out["edge_bottleneck_score"], low_name="low_bottleneck", high_name="high_bottleneck"
    )
    out["horizon_group"] = _low_high_group(out["median_h"], low_name="short_horizon", high_name="long_horizon")
    out["compatibility_group"] = _low_high_group(
        out["mean_endpoint_compatibility"], low_name="low_compatibility", high_name="high_compatibility"
    )
    out["planner_usage_group"] = np.where(out.get("used_by_planner", False).map(_as_bool), "planner_used", "not_planner_used")
    if "repair_reason" not in out.columns:
        out["repair_reason"] = "unknown"
    out["repair_reason"] = out["repair_reason"].fillna("unknown").astype(str)
    return out


def build_edge_delta_table(
    phase4k_per_seed: pd.DataFrame,
    baseline_method: str = "uniform_transition_none",
    planner_method: str = "calibrated_compat_threshold",
) -> pd.DataFrame:
    if phase4k_per_seed.empty:
        return pd.DataFrame()
    run_tables: dict[tuple[str, int], pd.DataFrame] = {}
    for row in phase4k_per_seed.to_dict("records"):
        method = str(row["method"])
        seed = int(row["seed"])
        table = load_run_edge_table(row["direct_summary_path"], planner_method=planner_method)
        table["dataset"] = row.get("dataset")
        table["phase2_run"] = row.get("phase2_run")
        table["method"] = method
        table["seed"] = seed
        run_tables[(method, seed)] = table

    rows: list[pd.DataFrame] = []
    for (method, seed), candidate in run_tables.items():
        if method == baseline_method:
            continue
        baseline = run_tables.get((baseline_method, seed))
        if baseline is None:
            continue
        merged = baseline.merge(
            candidate,
            on="edge_id",
            how="inner",
            suffixes=("_baseline", "_candidate"),
        )
        if merged.empty:
            continue
        out = pd.DataFrame(
            {
                "dataset": merged.get("dataset_candidate", merged.get("dataset_baseline")),
                "phase2_run": merged.get("phase2_run_candidate", merged.get("phase2_run_baseline")),
                "method": method,
                "baseline_method": baseline_method,
                "seed": seed,
                "edge_id": merged["edge_id"].astype(int),
                "bank_edge_id": merged.get("bank_edge_id_candidate", merged.get("bank_edge_id_baseline")),
                "src": merged.get("src_candidate", merged.get("src_baseline")),
                "dst": merged.get("dst_candidate", merged.get("dst_baseline")),
                "baseline_edge_action_mse": pd.to_numeric(merged["edge_action_mse_baseline"], errors="coerce"),
                "candidate_edge_action_mse": pd.to_numeric(merged["edge_action_mse_candidate"], errors="coerce"),
                "baseline_policy_support_score": pd.to_numeric(
                    merged["direct_edge_policy_support_score_baseline"], errors="coerce"
                ),
                "candidate_policy_support_score": pd.to_numeric(
                    merged["direct_edge_policy_support_score_candidate"], errors="coerce"
                ),
                "baseline_path_usage_count": pd.to_numeric(
                    merged.get("path_usage_count_baseline", 0), errors="coerce"
                ).fillna(0),
                "candidate_path_usage_count": pd.to_numeric(
                    merged.get("path_usage_count_candidate", 0), errors="coerce"
                ).fillna(0),
                "num_policy_eval_samples": pd.to_numeric(
                    merged.get("num_policy_eval_samples_candidate", merged.get("num_policy_eval_samples_baseline", 1)),
                    errors="coerce",
                ).fillna(1),
            }
        )
        out["edge_action_mse_delta"] = out["candidate_edge_action_mse"] - out["baseline_edge_action_mse"]
        out["edge_action_mse_ratio"] = out["candidate_edge_action_mse"] / out["baseline_edge_action_mse"].replace(0, np.nan)
        out["policy_support_delta"] = out["candidate_policy_support_score"] - out["baseline_policy_support_score"]
        out["improved"] = out["edge_action_mse_delta"] < 0
        for col in [
            "support_group",
            "bottleneck_group",
            "horizon_group",
            "compatibility_group",
            "planner_usage_group",
            "repair_reason",
            "num_unique_starts",
            "edge_bottleneck_score",
            "median_h",
            "num_segments",
            "num_unique_episodes",
            "mean_endpoint_compatibility",
        ]:
            candidate_col = f"{col}_candidate"
            baseline_col = f"{col}_baseline"
            if candidate_col in merged.columns:
                out[col] = merged[candidate_col]
            elif baseline_col in merged.columns:
                out[col] = merged[baseline_col]
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_group_deltas(edge_deltas: pd.DataFrame) -> pd.DataFrame:
    if edge_deltas.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group_type in GROUP_TYPES:
        if group_type not in edge_deltas.columns:
            continue
        for keys, group in edge_deltas.groupby(["dataset", "phase2_run", "method", group_type], dropna=False, sort=True):
            dataset, phase2_run, method, group_value = keys
            rows.append(_summarize_delta_group(dataset, phase2_run, method, group_type, group_value, group))
    return pd.DataFrame(rows)


def summarize_method_deltas(edge_deltas: pd.DataFrame) -> pd.DataFrame:
    if edge_deltas.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in edge_deltas.groupby(["dataset", "phase2_run", "method"], sort=True):
        dataset, phase2_run, method = keys
        rows.append(_summarize_delta_group(dataset, phase2_run, method, "all", "all_repair_edges", group))
    return pd.DataFrame(rows)


def rank_group_findings(group_summary: pd.DataFrame) -> pd.DataFrame:
    if group_summary.empty:
        return pd.DataFrame()
    out = group_summary.copy()
    out["benefit_score"] = -pd.to_numeric(out["mean_edge_action_mse_delta"], errors="coerce").fillna(0)
    out["planner_relevance_score"] = pd.to_numeric(out["planner_usage_rate"], errors="coerce").fillna(0)
    out["finding_score"] = out["benefit_score"] * (1.0 + out["planner_relevance_score"])
    return out.sort_values(["finding_score", "num_edges", "method"], ascending=[False, False, True], kind="mergesort")


def build_phase4l_payload(
    config: dict[str, Any],
    method_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    ranked_findings: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "phase": "Phase 4L",
        "title": "Direct Repair-Edge Group Diagnostics",
        "config": config,
        "method_summary": method_summary.to_dict("records"),
        "group_summary": group_summary.to_dict("records"),
        "top_group_findings": ranked_findings.head(int(config.get("num_top_findings", 12))).to_dict("records"),
        "related_work_reviewed": [
            {
                "name": "Goal-Conditioned Supervised Learning",
                "url": "https://arxiv.org/abs/1912.06088",
                "role": "Supervised goal-conditioned policy framing.",
            },
            {
                "name": "RvS: What is Essential for Offline RL via Supervised Learning?",
                "url": "https://arxiv.org/abs/2112.10751",
                "role": "Supervised offline RL framing and claim-boundary reminder.",
            },
            {
                "name": "Class-Balanced Loss Based on Effective Number of Samples",
                "url": "https://arxiv.org/abs/1901.05555",
                "role": "Loss reweighting under long-tailed sample imbalance.",
            },
            {
                "name": "Focal Loss for Dense Object Detection",
                "url": "https://arxiv.org/abs/1708.02002",
                "role": "Loss-side focusing without changing data provenance.",
            },
        ],
        "note": "Offline direct repair-edge supervised diagnostics only; not rollout success.",
    }


def write_phase4l_outputs(
    output_dir: str | Path,
    payload: dict[str, Any],
    edge_deltas: pd.DataFrame,
    method_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    ranked_findings: pd.DataFrame,
) -> None:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    edge_deltas.to_csv(out / "phase4l_repair_edge_deltas.csv", index=False)
    method_summary.to_csv(out / "phase4l_method_delta_summary.csv", index=False)
    group_summary.to_csv(out / "phase4l_group_delta_summary.csv", index=False)
    ranked_findings.to_csv(out / "phase4l_ranked_group_findings.csv", index=False)
    with (out / "phase4l_repair_group_diagnostics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (out / "phase4l_repair_group_diagnostics_summary.md").write_text(render_phase4l_markdown(payload), encoding="utf-8")


def render_phase4l_markdown(payload: dict[str, Any]) -> str:
    method_summary = pd.DataFrame(payload.get("method_summary", []))
    findings = pd.DataFrame(payload.get("top_group_findings", []))
    lines = [
        "# Phase 4L Direct Repair-Edge Group Diagnostics",
        "",
        "Phase 4L compares each loss-weighted checkpoint against the matched",
        "uniform-transition baseline on the same repair edges and seed. The",
        "metric is direct repair-edge supervised action MSE, not rollout success.",
        "",
        "## Method Deltas",
        "",
    ]
    if method_summary.empty:
        lines.append("No method summary rows were generated.")
    else:
        cols = [
            "method",
            "num_edges",
            "mean_edge_action_mse_delta",
            "mean_edge_action_mse_ratio",
            "fraction_edges_improved",
            "planner_usage_rate",
        ]
        present = [col for col in cols if col in method_summary.columns]
        lines.extend(_markdown_table(method_summary[present]))
    lines.extend(["", "## Top Group Findings", ""])
    if findings.empty:
        lines.append("No grouped findings were generated.")
    else:
        cols = [
            "method",
            "group_type",
            "group_value",
            "num_edges",
            "mean_edge_action_mse_delta",
            "mean_edge_action_mse_ratio",
            "fraction_edges_improved",
            "planner_usage_rate",
        ]
        present = [col for col in cols if col in findings.columns]
        lines.extend(_markdown_table(findings[present]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Negative MSE deltas mean the candidate checkpoint fits the repair edge",
            "better than the matched uniform-transition baseline. Planner-used groups",
            "are more relevant to the current repaired planner than unused groups,",
            "but this remains an offline supervised proxy.",
            "",
        ]
    )
    return "\n".join(lines)


def _summarize_delta_group(
    dataset: Any,
    phase2_run: Any,
    method: Any,
    group_type: str,
    group_value: Any,
    group: pd.DataFrame,
) -> dict[str, Any]:
    baseline = pd.to_numeric(group["baseline_edge_action_mse"], errors="coerce")
    candidate = pd.to_numeric(group["candidate_edge_action_mse"], errors="coerce")
    delta = candidate - baseline
    ratio = candidate / baseline.replace(0, np.nan)
    usage = pd.to_numeric(group.get("candidate_path_usage_count", 0), errors="coerce").fillna(0)
    samples = pd.to_numeric(group.get("num_policy_eval_samples", 1), errors="coerce").fillna(1)
    return {
        "dataset": dataset,
        "phase2_run": phase2_run,
        "method": method,
        "group_type": group_type,
        "group_value": group_value,
        "num_edges": int(group["edge_id"].nunique()),
        "num_seed_edge_rows": int(group.shape[0]),
        "num_planner_used_edges": int(group.loc[usage > 0, "edge_id"].nunique()),
        "planner_usage_rate": float((usage > 0).mean()) if len(usage) else np.nan,
        "baseline_mean_edge_action_mse": float(baseline.mean()),
        "candidate_mean_edge_action_mse": float(candidate.mean()),
        "mean_edge_action_mse_delta": float(delta.mean()),
        "median_edge_action_mse_delta": float(delta.median()),
        "mean_edge_action_mse_ratio": float(ratio.mean()),
        "fraction_edges_improved": float((delta < 0).mean()),
        "sample_weighted_baseline_mse": _weighted_mean(baseline, samples),
        "sample_weighted_candidate_mse": _weighted_mean(candidate, samples),
        "sample_weighted_mse_delta": _weighted_mean(candidate, samples) - _weighted_mean(baseline, samples),
        "mean_policy_support_delta": float(pd.to_numeric(group["policy_support_delta"], errors="coerce").mean()),
        "mean_candidate_path_usage_count": float(usage.mean()) if len(usage) else np.nan,
    }


def _low_high_group(values: pd.Series, low_name: str, high_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series("unknown", index=values.index)
    median = numeric.median()
    return pd.Series(np.where(numeric <= median, low_name, high_name), index=values.index).where(numeric.notna(), "unknown")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows."]
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for row in df.to_dict("records"):
        values = []
        for col in df.columns:
            value = row.get(col)
            if isinstance(value, float) and np.isfinite(value):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines
