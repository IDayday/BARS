#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GROUP_KEYS = ["env_name", "seed", "task_id", "variant_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped train/val/test splits for CAGE closed-loop contracts.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--group_keys", nargs="+", default=DEFAULT_GROUP_KEYS)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_examples", type=int, default=100)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def group_key(row: dict[str, Any], keys: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in keys)


def grouped_split(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row, group_keys)].append(row)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    n_groups = len(items)
    n_train = int(round(n_groups * train_frac))
    n_val = int(round(n_groups * val_frac))
    if n_groups >= 3:
        n_train = min(max(n_train, 1), n_groups - 2)
        n_val = min(max(n_val, 1), n_groups - n_train - 1)
    train_groups = items[:n_train]
    val_groups = items[n_train : n_train + n_val]
    test_groups = items[n_train + n_val :]
    return {
        "train": [row for _, group_rows in train_groups for row in group_rows],
        "val": [row for _, group_rows in val_groups for row in group_rows],
        "test": [row for _, group_rows in test_groups for row in group_rows],
    }


def bool_value(row: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key in row:
            return bool(row.get(key))
    return False


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "num_examples": 0,
            "hit_rate": None,
            "contract_positive_rate": None,
            "negative_progress_rate": None,
            "final_goal_rate": None,
            "recovery_rate": None,
            "policy_weak_rate": None,
        }
    hit = [bool_value(row, "hit", "label_hit") for row in rows]
    positive = [bool_value(row, "label_contract_positive", "label_good_contract", "contract_positive") for row in rows]
    negative = [bool_value(row, "label_negative_progress", "negative_progress") for row in rows]
    final_goal = [bool(row.get("final_phase") or row.get("target_mode") == "final_goal") for row in rows]
    recovery = [bool(row.get("recovery_candidate") or row.get("target_mode") == "recovery_candidate") for row in rows]
    policy_weak = [bool_value(row, "label_policy_weak", "policy_weak") for row in rows]
    return {
        "num_examples": n,
        "hit_rate": sum(hit) / n,
        "contract_positive_rate": sum(positive) / n,
        "negative_progress_rate": sum(negative) / n,
        "final_goal_rate": sum(final_goal) / n,
        "recovery_rate": sum(recovery) / n,
        "policy_weak_rate": sum(policy_weak) / n,
        "env_counts": dict(Counter(str(row.get("env_name")) for row in rows)),
        "variant_counts": dict(Counter(str(row.get("variant_source", row.get("variant"))) for row in rows)),
    }


def underpowered_reason(splits: dict[str, list[dict[str, Any]]], min_examples: int) -> list[str]:
    reasons: list[str] = []
    total = sum(len(rows) for rows in splits.values())
    if total < min_examples:
        reasons.append(f"num_examples<{min_examples}")
    for name in ["train", "val", "test"]:
        if not splits.get(name):
            reasons.append(f"{name}_empty")
    for label_key in ["label_hit", "label_contract_positive", "label_negative_progress"]:
        values = []
        for rows in splits.values():
            for row in rows:
                if label_key in row:
                    values.append(bool(row[label_key]))
        if values and len(set(values)) < 2:
            reasons.append(f"{label_key}_single_class")
    return reasons


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    lines = [
        "# CAGE Contract Dataset Splits",
        "",
        f"- status: {summary['status']}",
        f"- input_path: `{summary['input_path']}`",
        f"- total_examples: {summary['total_examples']}",
        f"- group_keys: {', '.join(summary['group_keys'])}",
    ]
    if summary["underpowered_reasons"]:
        lines.append(f"- underpowered_reasons: {', '.join(summary['underpowered_reasons'])}")
    lines.extend(["", "| split | examples | hit | contract_positive | negative_progress | final_goal | recovery | policy_weak |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for split in ["train", "val", "test"]:
        stats = summary["splits"][split]
        lines.append(
            f"| {split} | {stats['num_examples']} | {fmt(stats['hit_rate'])} | {fmt(stats['contract_positive_rate'])} | "
            f"{fmt(stats['negative_progress_rate'])} | {fmt(stats['final_goal_rate'])} | {fmt(stats['recovery_rate'])} | {fmt(stats['policy_weak_rate'])} |"
        )
    (out_dir / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    splits = grouped_split(
        rows,
        group_keys=list(args.group_keys),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", split_rows)
    reasons = underpowered_reason(splits, int(args.min_examples))
    summary = {
        "status": "SPLIT_UNDERPOWERED" if reasons else "ok",
        "input_path": str(Path(args.input_jsonl).resolve()),
        "total_examples": len(rows),
        "group_keys": list(args.group_keys),
        "underpowered_reasons": reasons,
        "splits": {name: split_stats(split_rows) for name, split_rows in splits.items()},
    }
    write_summary(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "status": summary["status"], "total_examples": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
