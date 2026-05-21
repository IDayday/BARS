#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_PATH = Path("research_state/bars_research_state.json")
DECISION_LEDGER = Path("research_state/decision_ledger.jsonl")
HYPOTHESIS_LEDGER = Path("research_state/hypothesis_ledger.jsonl")
OPEN_BLOCKERS = Path("research_state/open_blockers.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def round_name(n: int) -> str:
    return f"round_{n:03d}"


def update_hypothesis(round_num: int, gate: str) -> None:
    rows = []
    if HYPOTHESIS_LEDGER.exists():
        rows = [json.loads(line) for line in HYPOTHESIS_LEDGER.read_text().splitlines() if line.strip()]
    changed = False
    for row in rows:
        if row.get("hypothesis_id") == "H001":
            row["status"] = "SUPPORTED" if gate == "PASS_FAILURE_LABEL_QUALITY" else "REFUTED"
            row["evidence_files"] = [f"reports/{round_name(round_num)}_label_integrity.json", f"reports/{round_name(round_num)}_failure_atlas_all_variants.csv"]
            row["decision"] = "Protocol labels are usable for next-round decisions." if gate == "PASS_FAILURE_LABEL_QUALITY" else "Protocol labels remain blocking."
            changed = True
    if changed:
        HYPOTHESIS_LEDGER.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, default=0)
    args = p.parse_args()
    state = read_json(STATE_PATH, {})
    round_num = int(args.round or state.get("round", 1) or 1)
    rn = round_name(round_num)
    gate_status = read_json(Path("rounds") / rn / "gate_status.json", {})
    gates = gate_status.get("gates", {})
    label_gate = gates.get("failure_label_quality", "FAIL_FAILURE_LABEL_QUALITY")
    failed_jobs = 0
    failed_path = Path("rounds") / rn / "failed_jobs.csv"
    if failed_path.exists():
        failed_jobs = max(0, len(failed_path.read_text().splitlines()) - 1)
    if failed_jobs:
        decision = "STOP_PROTOCOL_BLOCKER_UNRESOLVED"
        reason = f"{failed_jobs} Round {round_num} jobs failed."
        next_track = "protocol_repair"
        next_batch = "Fix failed jobs and rerun all-variant failure-atlas labeling."
    elif label_gate == "PASS_FAILURE_LABEL_QUALITY":
        decision = "CONTINUE"
        reason = "Protocol label quality passed; oracle-headroom discovery is now the smallest decision-changing next batch."
        next_track = "oracle_headroom"
        next_batch = "Run Stage25 oracle-headroom scan v2 on candidate natural envs and audit reset/set_state reliability."
    else:
        decision = "STOP_PROTOCOL_BLOCKER_UNRESOLVED"
        reason = f"Failure label quality gate is {label_gate}."
        next_track = "protocol_repair"
        next_batch = "Improve fallback labeler/debug trace coverage and rerun Round 1."

    state["updated_at"] = now()
    state["global_decision"] = decision
    state["stop_reason"] = None if decision == "CONTINUE" else reason
    state["active_tracks"] = [next_track]
    state["active_scientific_question"] = "Do any available natural benchmarks contain path-useful oracle bridges?" if next_track == "oracle_headroom" else "Are labels still blocking protocol truthfulness?"
    state["next_round_plan"] = {
        "round": round_num + 1,
        "primary_question": state["active_scientific_question"],
        "track": next_track,
        "experiment_batch": next_batch,
        "commands": [
            "bash scripts/stage25_oracle_headroom_scan_v2.sh CONFIG=configs/stage25_oracle_scan_matrix.json ENVS=scene-play-v0,kitchen-partial-v0,visual-scene-play-v0,visual-antmaze-large-explore-v0 SEEDS=0 GRAPH_IDS=G3 GPUS=${GPUS:-0,1,2,3} EDGE_EXEC_PILOT=1 TOP_K_BRIDGE=4 MAX_SOURCES=200 WAIT=1",
            "python scripts/stage25_analyze.py --reports-root reports --oracle-reports-root reports/stage25_oracle_scan_tmp --oracle-artifact-root artifacts/stage25 --failure-atlas reports/stage25_failure_atlas_all_variants.csv --min-episodes 100",
        ]
        if next_track == "oracle_headroom"
        else [],
    }
    if decision == "CONTINUE":
        blockers = read_json(OPEN_BLOCKERS, {"blockers": []})
        for b in blockers.get("blockers", []):
            if b.get("track") == "protocol_repair":
                b["status"] = "closed"
                b["closed_in_round"] = round_num
        write_json(OPEN_BLOCKERS, blockers)
    write_json(STATE_PATH, state)
    append_jsonl(
        DECISION_LEDGER,
        {
            "round": round_num,
            "decision": decision,
            "reason": reason,
            "next_track": next_track,
            "next_experiment_batch": next_batch,
            "risks": ["Oracle labels may be unreliable if set_state_rate remains low."],
            "files": [str(Path("rounds") / rn / "gate_status.json"), f"reports/{rn}_reflection.md"],
        },
    )
    update_hypothesis(round_num, label_gate)
    next_plan = [
        f"# {rn} Next Plan",
        "",
        f"Decision: {decision}",
        f"Reason: {reason}",
        "",
        f"Next track: {next_track}",
        f"Next experiment batch: {next_batch}",
        "",
        "Non-negotiables remain active: no fallback planner evidence, no p_bridge before oracle headroom, no integrated BARS-v3 before oracle+p_bridge.",
    ]
    (Path("rounds") / rn / "next_plan.md").write_text("\n".join(next_plan) + "\n")
    Path(f"reports/{rn}_next_plan.md").write_text("\n".join(next_plan) + "\n")
    summary = {
        "ROUND": rn,
        "PRIMARY_QUESTION": gate_status.get("primary_question", ""),
        "EXPERIMENTS_COMPLETED": "protocol_repair_all_variant_failure_atlas",
        "FAILED_JOBS": failed_jobs,
        "PRIMARY_RESULTS": gate_status.get("details", {}),
        "GATES": gates,
        "DECISION": decision,
        "NEXT_ROUND": state.get("next_round_plan"),
        "PACKAGE": "pending_package_script",
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
