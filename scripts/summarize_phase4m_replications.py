#!/usr/bin/env python
"""Summarize Phase 4M planner-relevant loss-weighting replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _run_name(path: Path) -> str:
    try:
        return str(path.parent.relative_to(path.parents[2]))
    except ValueError:
        return path.parent.name


def collect_phase4m_replications(result_root: str | Path, method: str) -> pd.DataFrame:
    root = Path(result_root).expanduser()
    rows: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/*/phase4m_vs_baseline.csv")):
        df = pd.read_csv(path)
        if "method" not in df.columns:
            continue
        df = df[df["method"] == str(method)].copy()
        if df.empty:
            continue
        df.insert(0, "run_dir", str(path.parent))
        df.insert(1, "run_name", _run_name(path))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    metric_pairs = {
        "final_val_action_mse": "final_val_action_mse_ratio_vs_baseline",
        "direct_repair_edge_mse": "direct_repair_edge_mse_ratio_vs_baseline",
        "planner_used_repair_edge_mse": "planner_used_repair_edge_mse_ratio_vs_baseline",
        "not_planner_used_repair_edge_mse": "not_planner_used_repair_edge_mse_ratio_vs_baseline",
        "direct_repair_policy_support_score": "direct_repair_policy_support_score_ratio_vs_baseline",
    }
    for base_col, ratio_col in metric_pairs.items():
        if ratio_col in out.columns:
            out[f"{base_col}_improved"] = pd.to_numeric(out[ratio_col], errors="coerce") < 1.0
    if "direct_repair_policy_support_score_ratio_vs_baseline" in out.columns:
        out["direct_repair_policy_support_score_improved"] = (
            pd.to_numeric(out["direct_repair_policy_support_score_ratio_vs_baseline"], errors="coerce") > 1.0
        )
    return out


def summarize_replication_table(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {"num_runs": 0}
    planner_ratio = pd.to_numeric(table.get("planner_used_repair_edge_mse_ratio_vs_baseline"), errors="coerce")
    direct_ratio = pd.to_numeric(table.get("direct_repair_edge_mse_ratio_vs_baseline"), errors="coerce")
    final_ratio = pd.to_numeric(table.get("final_val_action_mse_ratio_vs_baseline"), errors="coerce")
    return {
        "num_runs": int(table.shape[0]),
        "datasets": sorted(table["dataset"].astype(str).unique().tolist()) if "dataset" in table.columns else [],
        "mean_final_val_mse_ratio": float(final_ratio.mean()),
        "mean_direct_repair_mse_ratio": float(direct_ratio.mean()),
        "mean_planner_used_repair_mse_ratio": float(planner_ratio.mean()),
        "num_runs_planner_used_improved": int((planner_ratio < 1.0).sum()),
        "num_runs_direct_repair_improved": int((direct_ratio < 1.0).sum()),
        "num_runs_final_val_improved": int((final_ratio < 1.0).sum()),
        "note": "Offline supervised replication summary; not rollout success.",
    }


def render_markdown(payload: dict[str, Any], table: pd.DataFrame) -> str:
    lines = [
        "# Phase 4M Planner-Relevant Weighting Replication Summary",
        "",
        "This aggregates reset-free offline supervised Phase 4M runs. It does",
        "not report environment rollout success.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Runs", ""])
    if table.empty:
        lines.append("No runs found.")
    else:
        cols = [
            "dataset",
            "phase2_run",
            "final_val_action_mse_ratio_vs_baseline",
            "direct_repair_edge_mse_ratio_vs_baseline",
            "planner_used_repair_edge_mse_ratio_vs_baseline",
            "direct_repair_policy_support_score_ratio_vs_baseline",
        ]
        present = [col for col in cols if col in table.columns]
        lines.append("| " + " | ".join(present) + " |")
        lines.append("| " + " | ".join("---" for _ in present) + " |")
        for row in table[present].to_dict("records"):
            values = []
            for col in present:
                value = row.get(col)
                values.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A ratio below 1.0 is better for MSE metrics. A ratio above 1.0 is",
            "better for direct repair policy support score. These are supervised",
            "offline proxies and must not be interpreted as closed-loop execution.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", default="results/phase4m")
    parser.add_argument("--method", default="planner_relevant_repair_s04")
    parser.add_argument("--output_dir", default="results/phase4m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = collect_phase4m_replications(args.result_root, args.method)
    payload = summarize_replication_table(table)
    table.to_csv(output_dir / "phase4m_replication_summary.csv", index=False)
    with (output_dir / "phase4m_replication_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
    (output_dir / "phase4m_replication_summary.md").write_text(render_markdown(payload, table), encoding="utf-8")
    print(f"Wrote Phase 4M replication summary to {output_dir}")
    if not table.empty:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
