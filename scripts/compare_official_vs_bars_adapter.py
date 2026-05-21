#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SAME_FLAG_COLUMNS = [
    "same_env_id",
    "same_task_id_list",
    "same_goal_sampling",
    "same_start_sampling",
    "same_observation_format",
    "same_goal_format",
    "same_success_source",
    "same_success_threshold",
    "same_max_episode_steps",
    "same_policy_checkpoint",
    "same_tdr_checkpoint",
    "same_graph_checkpoint",
    "same_eval_seed",
    "same_episode_count",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md-out", default="")
    parser.add_argument("--max-gap-pp", type=float, default=2.0)
    args = parser.parse_args()

    official = read_csv(Path(args.official))
    adapter = read_csv(Path(args.adapter))
    adapter_by_key = {(r.get("env"), r.get("seed")): r for r in adapter}
    rows: list[dict[str, Any]] = []
    for off in official:
        key = (off.get("env"), off.get("seed"))
        ad = adapter_by_key.get(key, {})
        official_score = as_float(off.get("official_eval_score"))
        adapter_score = as_float(ad.get("adapter_score"))
        gap_pp = None if official_score is None or adapter_score is None else 100.0 * (adapter_score - official_score)
        same_flags = {col: as_bool(ad.get(col)) for col in SAME_FLAG_COLUMNS}
        protocol_match = bool(ad) and all(same_flags.values())
        gap_ok = gap_pp is not None and abs(gap_pp) <= args.max_gap_pp
        baseline_status = off.get("certification_status", "")
        baseline_evidence = off.get("evidence_class", "")
        if baseline_status != "PASS_BASELINE_CERTIFICATION" or baseline_evidence not in {"E2_CERTIFIED_BASELINE_DIAGNOSTIC", "E4_FULL_BUDGET_TRAINED_METHOD"}:
            adapter_status = "SKIP_BASELINE_UNCERTIFIED"
            reason = "adapter comparison is protocol evidence only because baseline certification did not pass"
        elif protocol_match and gap_ok:
            adapter_status = "PASS_ADAPTER_CERTIFICATION"
            reason = "adapter gap and protocol equality checks pass"
        else:
            adapter_status = "FAIL_ADAPTER_CERTIFICATION"
            reason = "adapter gap exceeds threshold or protocol equality check failed"
        rows.append(
            {
                "env": off.get("env", ""),
                "seed": off.get("seed", ""),
                "adapter_variant": ad.get("adapter_variant", ""),
                "fallback_mode": ad.get("fallback_mode", ""),
                "official_eval_score": official_score if official_score is not None else "",
                "bars_adapter_score": adapter_score if adapter_score is not None else "",
                "adapter_gap_pp": gap_pp if gap_pp is not None else "",
                "max_gap_pp": args.max_gap_pp,
                "gap_within_2pp": gap_ok,
                "protocol_match": protocol_match,
                "baseline_certification_status": baseline_status,
                "baseline_evidence_class": baseline_evidence,
                "adapter_certification_status": adapter_status,
                "evidence_class": "PROTOCOL_DEBUG_ONLY" if adapter_status != "PASS_ADAPTER_CERTIFICATION" else "E3_SAME_BACKBONE_METHOD_COMPARISON",
                "reason": reason,
                **same_flags,
                "official_eval_csv": off.get("eval_csv", ""),
                "adapter_eval_csv": ad.get("adapter_eval_csv", ""),
                "policy_checkpoint": ad.get("policy_checkpoint", ""),
                "tdr_checkpoint": ad.get("tdr_checkpoint", ""),
                "graph_checkpoint": ad.get("graph_checkpoint", ""),
            }
        )
    out = Path(args.out)
    write_csv(out, rows)
    md_out = Path(args.md_out) if args.md_out else out.with_name(out.stem.replace("official_vs_adapter", "adapter_mismatch_report") + ".md")
    lines = [
        "# Round 002 Adapter Mismatch Report",
        "",
        "The adapter comparison is evaluated only as certification if the baseline row is already certified.",
        "",
        "| env | adapter | official | adapter | gap pp | protocol match | adapter status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {env} | {variant} | {official} | {adapter} | {gap} | {match} | {status} |".format(
                env=row["env"],
                variant=row["adapter_variant"],
                official=row["official_eval_score"],
                adapter=row["bars_adapter_score"],
                gap=row["adapter_gap_pp"],
                match=row["protocol_match"],
                status=row["adapter_certification_status"],
            )
        )
    md_out.write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
