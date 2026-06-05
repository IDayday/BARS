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
    parser = argparse.ArgumentParser(description="Build labeled closed-loop contract dataset from CLP probe outputs.")
    parser.add_argument("--probe_files", nargs="+", required=True)
    parser.add_argument("--out_path", default=None, help="CLP1 output JSONL path.")
    parser.add_argument("--out_jsonl", default=None, help="Backward-compatible output JSONL path.")
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--contractive_progress_threshold", type=float, default=0.2)
    parser.add_argument("--low_progress_threshold", type=float, default=0.0)
    parser.add_argument("--action_norm_high", type=float, default=0.95)
    args = parser.parse_args()
    if not args.out_path and not args.out_jsonl:
        parser.error("one of --out_path or --out_jsonl is required")
    return args


def load_probe_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for pattern in paths:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matches:
            if path.exists():
                rows.extend(
                    row
                    for row in iter_jsonl(path)
                    if row.get("record_type") in {"closed_loop_probe", "branchable_probe"}
                )
    return rows


def main() -> None:
    args = parse_args()
    out_path = args.out_path or args.out_jsonl
    thresholds = ContractThresholds(
        contractive_progress=args.contractive_progress_threshold,
        low_progress=args.low_progress_threshold,
        action_norm_high=args.action_norm_high,
    )
    rows = [contract_dataset_row(row, thresholds) for row in load_probe_rows(args.probe_files)]
    mark_policy_weak(rows)
    write_jsonl(out_path, rows)
    write_summary(args.out_summary, rows)
    print({"out_jsonl": out_path, "out_summary": args.out_summary, "rows": len(rows)})


def mark_policy_weak(rows: list[dict[str, Any]]) -> None:
    by_segment: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        segment_id = row.get("source_segment_id")
        if segment_id is None:
            continue
        by_segment.setdefault((row.get("env_name"), row.get("variant_source"), segment_id, row.get("horizon")), []).append(row)
    for records in by_segment.values():
        valid = [r for r in records if not r.get("failure_reason")]
        if not valid:
            continue
        all_negative = all(bool(r.get("label_contract_negative")) for r in valid)
        for row in records:
            row["label_policy_weak"] = bool(all_negative)


def write_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP1 Contract Dataset Summary\n\n")
        fh.write(f"- `num_rows`: {len(rows)}\n")
        if rows:
            for key in [
                "label_hit",
                "label_contractive",
                "label_contract_positive",
                "label_contract_negative",
                "label_negative_progress",
                "label_recovery_bad",
                "label_policy_weak",
                "label_unstable",
                "label_good_contract",
            ]:
                fh.write(f"- `{key}_rate`: {float(np.mean([bool(r.get(key)) for r in rows])):.4f}\n")
            fh.write(f"- `normalized_progress_mean`: {summarize_numeric(r.get('normalized_progress') for r in rows)['mean']}\n")
            fh.write("\n## By Env And Target Mode\n\n")
            fh.write("| env_name | target_mode | rows | contract_positive_rate | contract_negative_rate | hit_rate | progress_mean |\n")
            fh.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
            groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
            for row in rows:
                groups.setdefault((row.get("env_name"), row.get("target_mode")), []).append(row)
            for (env_name, target_mode), records in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
                progress = summarize_numeric(r.get("normalized_progress") for r in records)["mean"]
                fh.write(
                    "| "
                    + " | ".join(
                        [
                            str(env_name),
                            str(target_mode),
                            str(len(records)),
                            f"{float(np.mean([bool(r.get('label_contract_positive')) for r in records])):.4f}",
                            f"{float(np.mean([bool(r.get('label_contract_negative')) for r in records])):.4f}",
                            f"{float(np.mean([bool(r.get('hit')) for r in records])):.4f}",
                            "NA" if progress is None else f"{progress:.4f}",
                        ]
                    )
                    + " |\n"
                )
        else:
            fh.write("- No rows.\n")


if __name__ == "__main__":
    main()
