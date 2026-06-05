#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRICS = [
    "target_switch_count",
    "stall_count",
    "drift_count",
    "recovery_attempt_count",
    "mean_segment_progress",
    "segment_target_reach_rate",
]


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def load_episode_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("record_type", "episode") == "episode":
                    rows.append(record)
    return rows


def mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [numeric(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [numeric(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return mean(values)


def format_value(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def print_summary(rows: list[dict[str, Any]]) -> None:
    print(f"episodes\t{len(rows)}")
    print(f"success_rate\t{format_value(rate(rows, 'success'))}")
    print(f"no_path_rate\t{format_value(rate(rows, 'no_path'))}")

    recovery_attempts = sum(numeric(row.get("recovery_attempt_count")) or 0.0 for row in rows)
    recovery_successes = sum(numeric(row.get("recovery_success_count")) or 0.0 for row in rows)
    recovery_success_rate = recovery_successes / recovery_attempts if recovery_attempts else None
    print(f"recovery_success_rate\t{format_value(recovery_success_rate)}")

    final_rows = [row for row in rows if row.get("final_goal_on_step") is not None]
    print(f"final_goal_phase_success\t{format_value(rate(final_rows, 'success'))}")

    print("metric\tmean\tstd")
    for key in METRICS:
        avg, sd = mean_std(rows, key)
        print(f"{key}\t{format_value(avg)}\t{format_value(sd)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize CAGE JSONL trace files.")
    parser.add_argument("trace_files", nargs="+", type=Path, help="One or more CAGE JSONL trace files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_episode_rows(args.trace_files)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
