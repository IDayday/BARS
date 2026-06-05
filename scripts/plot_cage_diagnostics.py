#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PLOTS = [
    ("success_rate_mean", "success_rate_by_variant_per_env.png", "Success Rate"),
    ("drift_count_mean", "drift_count_by_variant_per_env.png", "Drift Count"),
    ("target_switch_count_mean", "target_switch_by_variant_per_env.png", "Target Switch Count"),
    ("recovery_success_rate_mean", "recovery_success_rate_by_variant_per_env.png", "Recovery Success Rate"),
    ("global_replan_request_count_mean", "global_replan_by_variant_per_env.png", "Global Replan Request Count"),
    ("max_consecutive_replan_burst_mean", "replan_burst_by_variant_per_env.png", "Max Consecutive Replan Burst"),
    ("churn_guard_trigger_count_mean", "churn_guard_trigger_by_variant_per_env.png", "Churn Guard Trigger Count"),
    ("segment_target_reach_rate_mean", "segment_target_reach_rate_by_variant_per_env.png", "Segment Target Reach Rate"),
    ("final_goal_stall_count_mean", "final_goal_stall_by_variant_per_env.png", "Final Goal Stall Count"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot focused CAGE diagnostic summary metrics.")
    parser.add_argument("--input", default="", help="Aggregated summary CSV or JSON path.")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--input_root", default="")
    parser.add_argument("--output_dir", default="")
    return parser


def load_grouped(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    if args.input:
        summary_path = Path(args.input)
        if summary_path.suffix.lower() == ".json":
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = data.get("grouped", [])
        else:
            with summary_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "plots"
        return rows, output_dir
    if args.summary_json:
        summary_path = Path(args.summary_json)
    elif args.input_root:
        summary_path = Path(args.input_root) / "tables" / "focused_summary.json"
    else:
        raise SystemExit("provide --summary_json or --input_root")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "plots"
    return data.get("grouped", []), output_dir


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, out_path: Path) -> bool:
    values = [numeric(row.get(metric)) for row in rows]
    if not any(value is not None for value in values):
        return False
    labels = [f"{row.get('env_name')}\\n{row.get('variant')}" for row in rows]
    heights = [value if value is not None else 0.0 for value in values]
    width = max(8.0, min(24.0, 0.45 * len(labels)))
    plt.figure(figsize=(width, 5.0))
    plt.bar(range(len(labels)), heights)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel(title)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return True


def main() -> int:
    args = build_parser().parse_args()
    rows, output_dir = load_grouped(args)
    written: list[str] = []
    for metric, filename, title in PLOTS:
        out_path = output_dir / filename
        if plot_metric(rows, metric, title, out_path):
            written.append(str(out_path))
    print(json.dumps({"plots": written}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
