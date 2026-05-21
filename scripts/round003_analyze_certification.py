#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from round003_lib import PRIMARY_QUESTION, SECONDARY_QUESTION, as_float, read_csv, write_json


def row_status(row: dict[str, str]) -> tuple[str, str]:
    artifact = row.get("artifact_status", "")
    score_pp = as_float(row.get("official_eval_score_pp"))
    lower_pp = as_float(row.get("lower_bound_pp"))
    protocol_match = str(row.get("protocol_match", "")).lower() == "true"
    num_task_ids = int(float(row.get("num_task_ids") or 0))
    rollouts_per_task = int(float(row.get("rollouts_per_task") or 0))
    eval_seed = row.get("eval_seed", "")
    job_status = row.get("job_status", "")
    if artifact == "LOCAL_UNDERTRAINED":
        return "FAIL_UNDERTRAINED_BASELINE", "local artifact is below the required full public training budget"
    if artifact != "OFFICIAL_FULL_BUDGET":
        return "SKIP_ARTIFACT_UNAVAILABLE", f"artifact_status={artifact}"
    if job_status not in {"completed", "cached"}:
        return "FAIL_PUBLIC_REPRODUCTION", f"official eval job_status={job_status}"
    if score_pp is None or lower_pp is None:
        return "FAIL_PUBLIC_REPRODUCTION", "missing official eval score or public lower bound"
    if score_pp < lower_pp:
        return "FAIL_PUBLIC_REPRODUCTION", f"official score {score_pp:.2f}pp below lower bound {lower_pp:.2f}pp"
    if not protocol_match:
        return "FAIL_PUBLIC_REPRODUCTION", "protocol_match=false"
    if num_task_ids != 5:
        return "FAIL_PUBLIC_REPRODUCTION", f"num_task_ids={num_task_ids}; expected 5"
    if rollouts_per_task < 49:
        return "FAIL_PUBLIC_REPRODUCTION", f"rollouts_per_task={rollouts_per_task}; expected >=49"
    if eval_seed == "":
        return "FAIL_PUBLIC_REPRODUCTION", "eval_seed not recorded"
    return "PASS_BASELINE_CERTIFICATION", "official full-budget artifact score meets public lower bound under matched protocol"


def aggregate_status(statuses: list[str]) -> str:
    if any(s == "PASS_BASELINE_CERTIFICATION" for s in statuses):
        return "PASS_BASELINE_CERTIFICATION"
    if any(s == "FAIL_PUBLIC_REPRODUCTION" for s in statuses):
        return "FAIL_PUBLIC_REPRODUCTION"
    if any(s == "FAIL_UNDERTRAINED_BASELINE" for s in statuses):
        return "FAIL_UNDERTRAINED_BASELINE"
    return "SKIP_ARTIFACT_UNAVAILABLE"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 003 Baseline Certification",
        "",
        f"Primary question: {PRIMARY_QUESTION}",
        "",
        f"Secondary question: {SECONDARY_QUESTION}",
        "",
        "## Gate Result",
        "",
        f"- Baseline certification: {payload['baseline_certification']}",
        f"- Certified envs: {', '.join(payload['certified_envs']) if payload['certified_envs'] else 'none'}",
        "",
        "## Rows",
        "",
        "| env | artifact | score pp | lower bound pp | protocol match | status | reason |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {env} | {artifact} | {score} | {lower} | {protocol} | {status} | {reason} |".format(
                env=row.get("env", ""),
                artifact=row.get("artifact_status", ""),
                score=row.get("official_eval_score_pp", ""),
                lower=row.get("lower_bound_pp", ""),
                protocol=row.get("protocol_match", ""),
                status=row.get("baseline_certification_status", ""),
                reason=row.get("certification_reason", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- PASS rows are certified baseline diagnostics only; they do not by themselves certify BARS.",
            "- No p_bridge, integrated BARS-v3, or failure-taxonomy causal interpretation is unlocked until adapter certification also passes.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-eval", default="reports/round_003_gas_official_eval.csv")
    parser.add_argument("--out", default="reports/round_003_baseline_certification.json")
    parser.add_argument("--md-out", default="reports/round_003_baseline_certification.md")
    args = parser.parse_args()

    rows = read_csv(Path(args.official_eval))
    enriched: list[dict[str, Any]] = []
    for row in rows:
        status, reason = row_status(row)
        out_row = dict(row)
        out_row["baseline_certification_status"] = status
        out_row["certification_reason"] = reason
        enriched.append(out_row)
    statuses = [r["baseline_certification_status"] for r in enriched]
    certified_envs = [r["env"] for r in enriched if r["baseline_certification_status"] == "PASS_BASELINE_CERTIFICATION"]
    payload = {
        "round": 3,
        "primary_question": PRIMARY_QUESTION,
        "secondary_question": SECONDARY_QUESTION,
        "baseline_certification": aggregate_status(statuses),
        "certified_envs": certified_envs,
        "status_counts": dict(Counter(statuses)),
        "official_eval_rows": len(enriched),
        "rows": enriched,
    }
    write_json(Path(args.out), payload)
    write_md(Path(args.md_out), payload)
    print(json.dumps({"baseline_certification": payload["baseline_certification"], "certified_envs": certified_envs}, sort_keys=True))


if __name__ == "__main__":
    main()
