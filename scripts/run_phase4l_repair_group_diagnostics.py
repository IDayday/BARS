#!/usr/bin/env python
"""Run Phase 4L grouped diagnostics for Phase 4K repair-edge evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4l_repair_group_diagnostics import (  # noqa: E402
    build_edge_delta_table,
    build_phase4l_payload,
    rank_group_findings,
    summarize_group_deltas,
    summarize_method_deltas,
    write_phase4l_outputs,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = _load_yaml(config_path)
    per_seed_csv = Path(str(config["phase4k_per_seed_csv"])).expanduser()
    output_dir = Path(str(config.get("output_dir", "results/phase4l"))).expanduser()
    if not per_seed_csv.exists():
        raise FileNotFoundError(f"Missing Phase 4K per-seed file: {per_seed_csv}")
    per_seed = pd.read_csv(per_seed_csv)
    edge_deltas = build_edge_delta_table(
        per_seed,
        baseline_method=str(config.get("baseline_method", "uniform_transition_none")),
        planner_method=str(config.get("planner_method", "calibrated_compat_threshold")),
    )
    method_summary = summarize_method_deltas(edge_deltas)
    group_summary = summarize_group_deltas(edge_deltas)
    ranked_findings = rank_group_findings(group_summary)
    payload = build_phase4l_payload(
        config={**config, "config_path": str(config_path)},
        method_summary=method_summary,
        group_summary=group_summary,
        ranked_findings=ranked_findings,
    )
    write_phase4l_outputs(output_dir, payload, edge_deltas, method_summary, group_summary, ranked_findings)
    print(f"[phase4l] wrote grouped diagnostics under {output_dir}")
    if not method_summary.empty:
        print(method_summary.to_string(index=False))
    if not ranked_findings.empty:
        print(ranked_findings.head(int(config.get("num_top_findings", 12))).to_string(index=False))


if __name__ == "__main__":
    main()
