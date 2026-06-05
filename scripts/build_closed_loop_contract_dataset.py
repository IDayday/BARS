#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import iter_jsonl, summarize_numeric, write_jsonl

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.closed_loop_contracts import ContractThresholds, contract_dataset_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labeled closed-loop contract dataset from CLP0 probe outputs.")
    parser.add_argument("--probe_files", nargs="+", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--contractive_progress_threshold", type=float, default=0.2)
    parser.add_argument("--action_norm_high", type=float, default=0.95)
    return parser.parse_args()


def load_probe_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for pattern in paths:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matches:
            if path.exists():
                rows.extend(row for row in iter_jsonl(path) if row.get("record_type") == "closed_loop_probe")
    return rows


def main() -> None:
    args = parse_args()
    thresholds = ContractThresholds(contractive_progress=args.contractive_progress_threshold, action_norm_high=args.action_norm_high)
    rows = [contract_dataset_row(row, thresholds) for row in load_probe_rows(args.probe_files)]
    write_jsonl(args.out_jsonl, rows)
    write_summary(args.out_summary, rows)
    print({"out_jsonl": args.out_jsonl, "out_summary": args.out_summary, "rows": len(rows)})


def write_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP0 Contract Dataset Summary\n\n")
        fh.write(f"- `num_rows`: {len(rows)}\n")
        if rows:
            for key in ["label_hit", "label_contractive", "label_negative_progress", "label_unstable", "label_good_contract"]:
                fh.write(f"- `{key}_rate`: {float(np.mean([bool(r.get(key)) for r in rows])):.4f}\n")
            fh.write(f"- `normalized_progress_mean`: {summarize_numeric(r.get('normalized_progress') for r in rows)['mean']}\n")
        else:
            fh.write("- No rows.\n")


if __name__ == "__main__":
    main()
