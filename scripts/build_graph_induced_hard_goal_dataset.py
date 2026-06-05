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

from cage.graph_induced_policy_dataset import hard_goal_example  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph-induced hard-goal policy-alignment dataset from CLP contract labels.")
    parser.add_argument("--contract_dataset", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--max_progress", type=float, default=0.05)
    return parser.parse_args()


def dphi_bin(value: Any) -> str:
    if value is None:
        return "unknown"
    v = float(value)
    if v < 4:
        return "<4"
    if v < 8:
        return "4-8"
    if v < 16:
        return "8-16"
    if v < 32:
        return "16-32"
    return ">=32"


def support_bin(value: Any) -> str:
    if value is None:
        return "unknown"
    v = float(value)
    if v < 0.25:
        return "<0.25"
    if v < 0.5:
        return "0.25-0.5"
    if v < 0.75:
        return "0.5-0.75"
    return ">=0.75"


def main() -> int:
    args = parse_args()
    rows = []
    for row in iter_jsonl(args.contract_dataset):
        example = hard_goal_example(row, max_progress=args.max_progress)
        if example is not None:
            rows.append(example.to_dict())
    write_jsonl(args.out_jsonl, rows)
    write_summary(args.out_summary, rows)
    print({"out_jsonl": args.out_jsonl, "out_summary": args.out_summary, "rows": len(rows)})
    return 0


def write_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = ["hard_positive", "hard_unlabeled", "hard_negative"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# CAGE-CLP1 Graph-Induced Hard Goal Dataset\n\n")
        fh.write(f"- `num_examples`: {len(rows)}\n")
        for category in categories:
            fh.write(f"- `{category}`: {sum(1 for r in rows if r.get('category') == category)}\n")
        supervision_rate = float(np.mean([bool(r.get("available_action_supervision")) for r in rows])) if rows else None
        fh.write(f"- `available_action_supervision_rate`: {supervision_rate}\n")
        fh.write(f"- `d_phi_mean`: {summarize_numeric(r.get('d_phi') for r in rows)['mean']}\n")
        fh.write(f"- `q_train_support_mean`: {summarize_numeric(r.get('q_train_support') for r in rows)['mean']}\n")
        fh.write("\n## Breakdown\n\n")
        fh.write("| group | value | count |\n")
        fh.write("| --- | --- | ---: |\n")
        for key in ["env_name", "target_mode", "category"]:
            counts: dict[str, int] = {}
            for row in rows:
                counts[str(row.get(key))] = counts.get(str(row.get(key)), 0) + 1
            for value, count in sorted(counts.items()):
                fh.write(f"| {key} | {value} | {count} |\n")
        for key, fn in [("d_phi_bin", dphi_bin), ("q_train_support_bin", support_bin)]:
            counts: dict[str, int] = {}
            for row in rows:
                counts[fn(row.get("d_phi" if key == "d_phi_bin" else "q_train_support"))] = counts.get(fn(row.get("d_phi" if key == "d_phi_bin" else "q_train_support")), 0) + 1
            for value, count in sorted(counts.items()):
                fh.write(f"| {key} | {value} | {count} |\n")
        fh.write("\n## Feasibility\n\n")
        fh.write(
            "Only `hard_positive` rows should be considered for supervised policy finetuning. "
            "`hard_negative` rows are suitable for contract/ranking losses, not naive behavior cloning.\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
