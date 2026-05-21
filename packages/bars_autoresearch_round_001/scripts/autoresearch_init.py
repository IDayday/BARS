#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_DIR = Path("research_state")
STATE_PATH = STATE_DIR / "bars_research_state.json"
HYPOTHESIS_LEDGER = STATE_DIR / "hypothesis_ledger.jsonl"
EXPERIMENT_LEDGER = STATE_DIR / "experiment_ledger.jsonl"
DECISION_LEDGER = STATE_DIR / "decision_ledger.jsonl"
CURRENT_BEST = STATE_DIR / "current_best_method.json"
OPEN_BLOCKERS = STATE_DIR / "open_blockers.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def append_jsonl_if_empty(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--max-wall-hours-per-round", type=float, default=12)
    p.add_argument("--max-gpu-hours-per-round", type=float, default=72)
    p.add_argument("--max-parallel-jobs", type=int, default=12)
    args = p.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    created = now()
    state = read_json(STATE_PATH, {})
    if not state:
        state = {
            "project": "BARS",
            "stage": "autonomous_loop",
            "round": 0,
            "created_at": created,
            "updated_at": created,
            "git_head": git_head(),
            "global_decision": "CONTINUE",
            "active_scientific_question": "Q1: Are failure labels complete and trustworthy across all variants?",
            "active_tracks": ["protocol_repair"],
            "closed_tracks": [],
            "best_positive_signal": None,
            "strongest_negative_signal": "Stage24 found weak reachability effect, failed local drift repair, no scene-play oracle path-level headroom, boundary coverage collapse, and D4RL protocol hold.",
            "must_not_do": [
                "Do not use direct-goal/progress fallback as planner evidence.",
                "Do not train p_bridge until PASS_ORACLE_HEADROOM.",
                "Do not run integrated BARS-v3 until PASS_ORACLE_HEADROOM and PASS_P_BRIDGE.",
                "Keep boundary diagnostic-only until coverage/gap/AUROC gates pass.",
                "Keep D4RL protocol/debug-only until PASS_D4RL_PROTOCOL_AUDIT.",
            ],
            "next_round_plan": {
                "round": 1,
                "primary_question": "Are all Stage24 variants labeled consistently enough for autonomous decisions?",
                "track": "protocol_repair",
            },
            "stop_reason": None,
        }
    state.setdefault("limits", {})
    state["limits"].update(
        {
            "max_rounds": args.max_rounds,
            "max_wall_hours_per_round": args.max_wall_hours_per_round,
            "max_gpu_hours_per_round": args.max_gpu_hours_per_round,
            "max_parallel_jobs": args.max_parallel_jobs,
        }
    )
    state["updated_at"] = now()
    state["git_head"] = git_head()
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    append_jsonl_if_empty(
        HYPOTHESIS_LEDGER,
        [
            {
                "round": 1,
                "hypothesis_id": "H001",
                "question": "Are all Stage24 variants labelable under one failure-atlas protocol?",
                "prediction": "Failed rows have non-empty primary_failure_type and unclassified failure rate <= 0.20.",
                "experiment": "all_variant_failure_atlas_repair",
                "status": "OPEN",
                "evidence_files": [],
                "decision": "",
            },
            {
                "round": 2,
                "hypothesis_id": "H002",
                "question": "Do any available natural benchmarks contain path-useful oracle bridges?",
                "prediction": "At least one candidate env has oracle bridge usage >= 0.20 and useful_bridge_score >= 0.20.",
                "experiment": "oracle_headroom_scan_v2",
                "status": "OPEN",
                "evidence_files": [],
                "decision": "",
            },
            {
                "round": 4,
                "hypothesis_id": "H003",
                "question": "Is local execution drift still a causal bottleneck after protocol-aligned labeling?",
                "prediction": "F4_LOCAL_EXECUTION_DRIFT remains measurable and can be reduced without success loss.",
                "experiment": "local_drift_v2",
                "status": "OPEN",
                "evidence_files": [],
                "decision": "",
            },
        ],
    )
    for path in [EXPERIMENT_LEDGER, DECISION_LEDGER]:
        path.touch(exist_ok=True)
    if not CURRENT_BEST.exists():
        CURRENT_BEST.write_text(
            json.dumps(
                {
                    "method": "official_control_GAS_baseline",
                    "status": "baseline_only",
                    "evidence": "Stage23 protocol repair aligned BARS adapter to official GAS, but no Stage24 BARS component passed main gates.",
                    "report_files": ["reports/stage24_gate_status.json", "reports/stage24_decisions.md"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if not OPEN_BLOCKERS.exists():
        OPEN_BLOCKERS.write_text(
            json.dumps(
                {
                    "blockers": [
                        {
                            "id": "B001",
                            "track": "protocol_repair",
                            "status": "open",
                            "description": "Failure labels must cover all variants before local-drift claims are interpreted.",
                        },
                        {
                            "id": "B002",
                            "track": "oracle_headroom",
                            "status": "open",
                            "description": "No tested env has path-level oracle bridge headroom.",
                        },
                        {
                            "id": "B003",
                            "track": "boundary",
                            "status": "open",
                            "description": "Boundary coverage is far below re-entry threshold.",
                        },
                        {
                            "id": "B004",
                            "track": "d4rl",
                            "status": "open",
                            "description": "D4RL remains protocol/debug-only until audit passes.",
                        },
                    ]
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps({"state": str(STATE_PATH), "round": state.get("round", 0), "decision": state.get("global_decision")}, sort_keys=True))


if __name__ == "__main__":
    main()
