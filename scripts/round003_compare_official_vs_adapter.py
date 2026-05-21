#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from round003_lib import as_float, normalize_task_id_list, read_csv, read_json, write_csv


SAME_COLUMNS = [
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

FIELDNAMES = [
    "env",
    "seed",
    "adapter_variant",
    "fallback_mode",
    "official_eval_score",
    "official_eval_score_pp",
    "bars_adapter_score",
    "bars_adapter_score_pp",
    "adapter_gap_pp",
    "abs_adapter_gap_pp",
    "max_gap_pp",
    "gap_within_2pp",
    "protocol_match",
    "baseline_certification",
    "adapter_certification_status",
    "evidence_class",
    "reason",
    *SAME_COLUMNS,
    "official_eval_csv",
    "adapter_eval_csv",
    "policy_checkpoint",
    "tdr_checkpoint",
    "graph_checkpoint",
]


def baseline_status_by_env(path: Path) -> dict[str, str]:
    data = read_json(path, {})
    out = {}
    for row in data.get("rows", []):
        out[str(row.get("env", ""))] = str(row.get("baseline_certification_status", ""))
    return out


def same(a: Any, b: Any) -> bool:
    return str(a).strip() == str(b).strip()


def compare_row(off: dict[str, str], ad: dict[str, str], baseline_status: str, max_gap_pp: float) -> dict[str, Any]:
    official = as_float(off.get("official_eval_score"))
    official_pp = as_float(off.get("official_eval_score_pp"))
    adapter = as_float(ad.get("adapter_score"))
    adapter_pp = as_float(ad.get("adapter_score_pp"))
    gap_pp = adapter_pp - official_pp if adapter_pp is not None and official_pp is not None else None
    same_flags = {
        "same_env_id": same(off.get("env"), ad.get("env")),
        "same_task_id_list": normalize_task_id_list(off.get("task_id_list", "")) == normalize_task_id_list(ad.get("task_id_list", "")),
        "same_goal_sampling": same(off.get("goal_sampling"), ad.get("goal_sampling")),
        "same_start_sampling": same(off.get("start_sampling"), ad.get("start_sampling")),
        "same_observation_format": same(off.get("observation_format"), ad.get("observation_format")),
        "same_goal_format": same(off.get("goal_format"), ad.get("goal_format")),
        "same_success_source": same(off.get("success_source"), ad.get("success_source")),
        "same_success_threshold": same(off.get("success_threshold"), ad.get("success_threshold")),
        "same_max_episode_steps": same(off.get("max_episode_steps"), ad.get("max_episode_steps")),
        "same_policy_checkpoint": same(off.get("policy_checkpoint"), ad.get("policy_checkpoint")),
        "same_tdr_checkpoint": same(off.get("tdr_checkpoint"), ad.get("tdr_checkpoint")),
        "same_graph_checkpoint": same(off.get("graph_checkpoint"), ad.get("graph_checkpoint")),
        "same_eval_seed": same(off.get("eval_seed"), ad.get("eval_seed")),
        "same_episode_count": same(off.get("episode_count"), ad.get("episode_count")),
    }
    protocol_match = bool(ad) and all(same_flags.values()) and str(off.get("protocol_match", "")).lower() == "true"
    gap_ok = gap_pp is not None and abs(gap_pp) <= max_gap_pp
    if baseline_status != "PASS_BASELINE_CERTIFICATION":
        status = "SKIP_BASELINE_UNCERTIFIED"
        evidence = "PROTOCOL_DEBUG_ONLY"
        reason = "baseline certification did not pass"
    elif not ad:
        status = "FAIL_ADAPTER_MISMATCH"
        evidence = "PROTOCOL_DEBUG_ONLY"
        reason = "missing adapter row"
    elif ad.get("job_status") not in {"completed", "cached"}:
        status = "FAIL_ADAPTER_MISMATCH"
        evidence = "PROTOCOL_DEBUG_ONLY"
        reason = f"adapter job_status={ad.get('job_status')}"
    elif protocol_match and gap_ok:
        status = "PASS_ADAPTER_CERTIFICATION"
        evidence = "E3_SAME_BACKBONE_METHOD_COMPARISON"
        reason = "adapter gap and all protocol equality checks pass"
    else:
        status = "FAIL_ADAPTER_MISMATCH"
        evidence = "PROTOCOL_DEBUG_ONLY"
        failed = [k for k, v in same_flags.items() if not v]
        reason = "adapter gap exceeds 2pp or protocol equality check failed"
        if failed:
            reason += f"; failed_flags={','.join(failed)}"
    return {
        "env": off.get("env", ""),
        "seed": off.get("seed", ""),
        "adapter_variant": ad.get("adapter_variant", ""),
        "fallback_mode": ad.get("fallback_mode", ""),
        "official_eval_score": "" if official is None else official,
        "official_eval_score_pp": "" if official_pp is None else official_pp,
        "bars_adapter_score": "" if adapter is None else adapter,
        "bars_adapter_score_pp": "" if adapter_pp is None else adapter_pp,
        "adapter_gap_pp": "" if gap_pp is None else gap_pp,
        "abs_adapter_gap_pp": "" if gap_pp is None else abs(gap_pp),
        "max_gap_pp": max_gap_pp,
        "gap_within_2pp": gap_ok,
        "protocol_match": protocol_match,
        "baseline_certification": baseline_status,
        "adapter_certification_status": status,
        "evidence_class": evidence,
        "reason": reason,
        **same_flags,
        "official_eval_csv": off.get("eval_csv", ""),
        "adapter_eval_csv": ad.get("adapter_eval_csv", ""),
        "policy_checkpoint": ad.get("policy_checkpoint", ""),
        "tdr_checkpoint": ad.get("tdr_checkpoint", ""),
        "graph_checkpoint": ad.get("graph_checkpoint", ""),
    }


def aggregate_adapter_status(rows: list[dict[str, Any]], baseline_statuses: dict[str, str]) -> str:
    if not any(s == "PASS_BASELINE_CERTIFICATION" for s in baseline_statuses.values()):
        return "SKIP_BASELINE_UNCERTIFIED"
    if any(row.get("adapter_certification_status") == "PASS_ADAPTER_CERTIFICATION" for row in rows):
        return "PASS_ADAPTER_CERTIFICATION"
    return "FAIL_ADAPTER_MISMATCH"


def write_md(path: Path, rows: list[dict[str, Any]], aggregate: str) -> None:
    lines = [
        "# Round 003 Adapter Certification",
        "",
        f"- Adapter certification: {aggregate}",
        "- The aggregate PASS means at least one official-artifact env has row-level PASS; diagnostics are only unlocked on row-level PASS envs.",
        "- PASS_ADAPTER_CERTIFICATION requires the same official checkpoint, graph, TDR/policy, env, task IDs, goal/start sampling, success threshold, max episode length, seed, and episode count, with abs(adapter_gap_pp) <= 2.0.",
        "",
        "| env | official pp | adapter pp | gap pp | protocol match | status | reason |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {env} | {official} | {adapter} | {gap} | {match} | {status} | {reason} |".format(
                env=row.get("env", ""),
                official=row.get("official_eval_score_pp", ""),
                adapter=row.get("bars_adapter_score_pp", ""),
                gap=row.get("adapter_gap_pp", ""),
                match=row.get("protocol_match", ""),
                status=row.get("adapter_certification_status", ""),
                reason=row.get("reason", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", default="reports/round_003_gas_official_eval.csv")
    parser.add_argument("--adapter", default="reports/round_003_bars_adapter_eval.csv")
    parser.add_argument("--baseline-certification", default="reports/round_003_baseline_certification.json")
    parser.add_argument("--out", default="reports/round_003_official_vs_adapter.csv")
    parser.add_argument("--md-out", default="reports/round_003_adapter_certification.md")
    parser.add_argument("--max-gap-pp", type=float, default=2.0)
    args = parser.parse_args()

    official = read_csv(Path(args.official))
    adapter = read_csv(Path(args.adapter))
    ad_by_key = {(r.get("env", ""), r.get("seed", "")): r for r in adapter}
    baseline_statuses = baseline_status_by_env(Path(args.baseline_certification))
    rows = []
    for off in official:
        key = (off.get("env", ""), off.get("seed", ""))
        rows.append(compare_row(off, ad_by_key.get(key, {}), baseline_statuses.get(key[0], ""), args.max_gap_pp))
    write_csv(Path(args.out), rows, FIELDNAMES)
    aggregate = aggregate_adapter_status(rows, baseline_statuses)
    write_md(Path(args.md_out), rows, aggregate)
    print(json.dumps({"adapter_certification": aggregate, "rows": len(rows), "out": args.out}, sort_keys=True))


if __name__ == "__main__":
    main()
