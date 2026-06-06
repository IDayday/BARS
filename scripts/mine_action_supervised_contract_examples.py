#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine action-supervised hard-positive ECG contract examples.")
    parser.add_argument("--contract_dataset_path", default="results/cage_clp1/datasets/closed_loop_contracts.jsonl")
    parser.add_argument("--segment_capture_roots", nargs="*", default=["results/cage_clp1/segment_capture", "results/cage_clp1/segment_capture_candidate"])
    parser.add_argument("--ogbench_dataset_cache", default="")
    parser.add_argument("--out_jsonl", default="results/cage_ecg/policy_alignment/action_supervised_contract_examples.jsonl")
    parser.add_argument("--out_report", default="reports/stage36_action_supervised_contract_mining.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract_rows = load_jsonl(Path(args.contract_dataset_path))
    segment_lookup = load_segment_lookup(args.segment_capture_roots or [])
    examples = mine_examples(contract_rows, segment_lookup)
    out_path = Path(args.out_jsonl)
    write_jsonl(out_path, examples)
    summary = write_report(Path(args.out_report), examples, args.ogbench_dataset_cache)
    print(json.dumps({"out_jsonl": str(out_path), **summary}, sort_keys=True))
    return 0


def mine_examples(rows: list[dict[str, Any]], segment_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if not is_candidate(row):
            continue
        source_segment_id = str(row.get("source_segment_id") or "")
        segment = segment_lookup.get(source_segment_id, {})
        action_sequence = first_present(segment, ["action_sequence", "actions", "policy_actions"])
        first_action = first_present(segment, ["first_action", "action"])
        if action_sequence is not None:
            action_available = True
            quality = "sequence"
            source = "segment_capture"
            failure = None
        elif first_action is not None:
            action_available = True
            quality = "first_action_only"
            source = "segment_capture"
            failure = None
        elif segment:
            action_available = False
            quality = "missing"
            source = None
            failure = "segment_capture_has_no_action_fields"
        else:
            action_available = False
            quality = "missing"
            source = None
            failure = "no_matching_segment_capture"
        examples.append(
            {
                "env_name": row.get("env_name"),
                "seed": row.get("seed"),
                "source_segment_id": source_segment_id or None,
                "phi_s": row.get("phi_s", row.get("phi_start")),
                "phi_g": row.get("phi_g", row.get("phi_target")),
                "state_ref_s": segment.get("start_state_ref"),
                "target_mode": row.get("target_mode", row.get("pair_source")),
                "action_sequence": action_sequence,
                "first_action": first_action,
                "action_available": action_available,
                "action_source": source,
                "horizon": row.get("horizon"),
                "contract_lcb": row.get("contract_lcb"),
                "predicted_hit": row.get("predicted_hit"),
                "predicted_negative_progress": row.get("predicted_negative_progress"),
                "label_positive_contract": bool(row.get("label_contract_positive") or row.get("label_good_contract") or row.get("hit")),
                "label_negative_contract": bool(row.get("label_contract_negative") or row.get("negative_progress")),
                "final_phase": bool(row.get("final_phase", False)),
                "recovery_candidate": bool(row.get("recovery_candidate", False)),
                "supervision_quality": quality,
                "failure_reason_if_missing": failure,
            }
        )
    return examples


def is_candidate(row: dict[str, Any]) -> bool:
    return bool(
        row.get("label_contract_positive")
        or row.get("label_good_contract")
        or row.get("hit")
        or row.get("final_phase")
        or row.get("recovery_candidate")
    )


def load_segment_lookup(roots: list[str]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in segment_files(roots):
        for row in load_jsonl(path):
            segment_id = row.get("segment_id")
            if segment_id:
                lookup[str(segment_id)] = row
            compact_id = compact_segment_id(row)
            if compact_id:
                lookup.setdefault(compact_id, row)
    return lookup


def compact_segment_id(row: dict[str, Any]) -> str | None:
    env = row.get("env_name")
    seed = row.get("seed")
    task = row.get("task_id")
    ep = row.get("episode_idx")
    seg = row.get("segment_idx")
    if env is None or seed is None or task is None or ep is None or seg is None:
        return None
    return f"{env}__seed{int(float(seed))}__task{int(float(task))}__ep{int(float(ep))}__seg{int(float(seg))}"


def write_report(path: Path, examples: list[dict[str, Any]], ogbench_dataset_cache: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(examples)
    action_count = sum(1 for row in examples if row.get("action_available"))
    positives_with_action = sum(1 for row in examples if row.get("label_positive_contract") and row.get("action_available"))
    final_with_action = sum(1 for row in examples if row.get("final_phase") and row.get("action_available"))
    recovery_with_action = sum(1 for row in examples if row.get("recovery_candidate") and row.get("action_available"))
    missing = Counter(str(row.get("failure_reason_if_missing") or "action_available") for row in examples)
    summary = {
        "total_candidates": total,
        "action_available_count": action_count,
        "action_supervision_rate": rate(action_count, total),
        "positive_with_action_count": positives_with_action,
        "final_goal_with_action_count": final_with_action,
        "recovery_with_action_count": recovery_with_action,
        "missing_reason_counts": dict(missing),
        "ogbench_dataset_cache": ogbench_dataset_cache or None,
        "bc_feasible": bool(positives_with_action > 0),
    }
    lines = [
        "# Stage36 Action-Supervised Contract Mining",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_candidates | {total} |",
        f"| action_available_count | {action_count} |",
        f"| action_supervision_rate | {fmt(summary['action_supervision_rate'])} |",
        f"| positive_with_action_count | {positives_with_action} |",
        f"| final_goal_with_action_count | {final_with_action} |",
        f"| recovery_with_action_count | {recovery_with_action} |",
        "",
        "## Missing Reason Counts",
        "",
        json.dumps(summary["missing_reason_counts"], indent=2, sort_keys=True),
        "",
        "## Interpretation",
        "",
        "如果 action_supervision_rate 仍接近 0，则不能做 BC，只能做 ranking/contrastive/conservative filtering，或继续恢复带 action 的 segment trace / OGBench trajectory supervision。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def segment_files(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file() and path.name.endswith("_segments.jsonl"):
            out.append(path)
        elif path.exists():
            out.extend(sorted(path.glob("*_segments.jsonl")))
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
