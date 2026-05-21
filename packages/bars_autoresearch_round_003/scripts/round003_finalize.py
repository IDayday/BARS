#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from round003_lib import (
    PRIMARY_QUESTION,
    SECONDARY_QUESTION,
    as_float,
    max_episode_steps,
    read_csv,
    read_json,
    required_train_steps,
    way_steps,
    write_csv,
    write_json,
)


RECLASS_FIELDNAMES = [
    "stage",
    "run_root",
    "env",
    "algorithm_or_variant",
    "condition",
    "fallback_mode",
    "train_steps",
    "official_artifact_used",
    "public_target_available",
    "baseline_certification_status",
    "adapter_certification_status",
    "evidence_class",
    "allowed_claim_level",
    "downgrade_reason",
    "report_file",
    "row_count",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def adapter_aggregate(compare_rows: list[dict[str, str]], baseline_status: str) -> str:
    if baseline_status != "PASS_BASELINE_CERTIFICATION":
        return "SKIP_BASELINE_UNCERTIFIED"
    if any(r.get("adapter_certification_status") == "PASS_ADAPTER_CERTIFICATION" for r in compare_rows):
        return "PASS_ADAPTER_CERTIFICATION"
    return "FAIL_ADAPTER_MISMATCH"


def write_medium_plan(report_path: Path, command_path: Path) -> None:
    envs = ["antmaze-medium-stitch-v0", "antmaze-medium-navigate-v0"]
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export PYTHONPATH=${PYTHONPATH:-$PWD}",
        "export WANDB_MODE=disabled",
        "export WANDB_DISABLED=true",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "export MUJOCO_GL=${MUJOCO_GL:-osmesa}",
        "GAS_REPO_PATH=${GAS_REPO_PATH:-external_src/GAS}",
        "OUT_ROOT=${OUT_ROOT:-artifacts/gas_full_budget_round003}",
        "GPU=${GPU:-0}",
        "",
        "cd \"$GAS_REPO_PATH\"",
        "",
    ]
    report = [
        "# Round 003 Medium Full-Budget Training Plan",
        "",
        "Medium official artifacts remain unavailable in the official checkpoint listing. These commands are a plan, not a reduced substitute.",
        "",
        "Evidence class before completion: E1_BASELINE_REGISTRY. Any reduced 10k/100k run must be labeled E0_SMOKE_ONLY.",
        "",
    ]
    for env in envs:
        slug = env.replace("-v0", "")
        tdr_dir = f"$OUT_ROOT/{env}/seed0/tdr"
        graph_dir = f"$OUT_ROOT/{env}/seed0/graph"
        policy_dir = f"$OUT_ROOT/{env}/seed0/policy"
        flags = (
            f"--env_name {env} --seed 0 --gpu $GPU --agent_config.encoder not_used "
            f"--agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 "
            f"--agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps {way_steps(env)}"
        )
        commands = [
            f"python pretrain_tdr.py --run_tdr_project Round003_medium_tdr --run_group round003_{slug} --save_tdr_dir {tdr_dir} --train_steps 1000000 --log_interval 5000 --save_interval 100000 {flags}",
            f"python construct_graph.py --run_group round003_{slug} --save_graph_dir {graph_dir} --te_threshold 0.99 --tdr_path {tdr_dir}/round003_{slug}/params_1000000.pkl {flags}",
            f"python train_policy.py --run_policy_project Round003_medium_policy --run_group round003_{slug} --save_policy_dir {policy_dir} --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path {tdr_dir}/round003_{slug}/params_1000000.pkl {flags}",
            f"python evaluate_gas.py --run_eval_project Round003_medium_eval --run_group round003_{slug} --save_eval_dir $OUT_ROOT/{env}/seed0/eval --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path {graph_dir}/round003_{slug}/keygraph.pkl --policy_path {policy_dir}/round003_{slug}/params_1000000.pkl {flags}",
        ]
        report.extend(
            [
                f"## {env}",
                "",
                f"- Required train steps: {required_train_steps(env)}",
                f"- Max episode steps: {max_episode_steps(env)}",
                "- Evaluation: official 49 eval + 1 video episode per task, equivalent to 50 rollouts/task across task IDs 1-5.",
                "",
                "```bash",
                *commands,
                "```",
                "",
            ]
        )
        command_lines.extend([f"# {env}", *commands, ""])
    command_path.write_text("\n".join(command_lines) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report))


def write_reclassification(path: Path, md_path: Path, baseline: dict[str, Any], compare_rows: list[dict[str, str]]) -> None:
    prior_rows = read_csv(Path("reports/round_002_prior_evidence_reclassification.csv"))
    rows: list[dict[str, Any]] = [dict(r) for r in prior_rows]
    baseline_rows = baseline.get("rows", [])
    for row in baseline_rows:
        status = row.get("baseline_certification_status", "")
        rows.append(
            {
                "stage": "Round003",
                "run_root": row.get("eval_csv", ""),
                "env": row.get("env", ""),
                "algorithm_or_variant": "GAS_official_evaluate_gas",
                "condition": "baseline_certification",
                "fallback_mode": "none",
                "train_steps": row.get("train_steps", ""),
                "official_artifact_used": row.get("artifact_status") == "OFFICIAL_FULL_BUDGET",
                "public_target_available": row.get("public_mean_pp", "") != "",
                "baseline_certification_status": status,
                "adapter_certification_status": "PENDING_ADAPTER_CERTIFICATION",
                "evidence_class": "E2_CERTIFIED_BASELINE_DIAGNOSTIC" if status == "PASS_BASELINE_CERTIFICATION" else "E1_BASELINE_REGISTRY",
                "allowed_claim_level": "CERTIFIED_BASELINE_ONLY" if status == "PASS_BASELINE_CERTIFICATION" else "PROTOCOL_DEBUG_ONLY",
                "downgrade_reason": row.get("certification_reason", ""),
                "report_file": "reports/round_003_baseline_certification.md",
                "row_count": 1,
            }
        )
    for row in compare_rows:
        status = row.get("adapter_certification_status", "")
        rows.append(
            {
                "stage": "Round003",
                "run_root": row.get("adapter_eval_csv", ""),
                "env": row.get("env", ""),
                "algorithm_or_variant": row.get("adapter_variant", ""),
                "condition": "adapter_certification",
                "fallback_mode": row.get("fallback_mode", "none"),
                "train_steps": "official_artifact",
                "official_artifact_used": True,
                "public_target_available": True,
                "baseline_certification_status": row.get("baseline_certification", ""),
                "adapter_certification_status": status,
                "evidence_class": "E3_SAME_BACKBONE_METHOD_COMPARISON" if status == "PASS_ADAPTER_CERTIFICATION" else "PROTOCOL_DEBUG_ONLY",
                "allowed_claim_level": "SAME_BACKBONE_MECHANISM_ON_CERTIFIED_ENV" if status == "PASS_ADAPTER_CERTIFICATION" else "PROTOCOL_DEBUG_ONLY",
                "downgrade_reason": row.get("reason", ""),
                "report_file": "reports/round_003_adapter_certification.md",
                "row_count": 1,
            }
        )
    write_csv(path, rows, RECLASS_FIELDNAMES)
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counts[(str(row.get("evidence_class", "")), str(row.get("allowed_claim_level", "")))] += int(row.get("row_count") or 1)
    certified_envs = [r.get("env", "") for r in compare_rows if r.get("adapter_certification_status") == "PASS_ADAPTER_CERTIFICATION"]
    lines = [
        "# Round 003 Prior Evidence Reclassification",
        "",
        "Round003 adds official-artifact baseline and adapter certification rows. Prior rows are not upgraded unless the exact env and artifact lineage are certified.",
        "",
        f"- Certified Round003 envs: {', '.join(certified_envs) if certified_envs else 'none'}",
        "- Medium rows remain downgraded because their local GAS artifacts are 100000-step artifacts, not public-quality 1000000-step artifacts.",
        "- Prior giant/large/scene diagnostics are not automatically upgraded unless their artifact lineage is the same official checkpoint/graph certified in Round003.",
        "",
        "## Summary",
        "",
        "| evidence class | allowed claim level | grouped rows |",
        "| --- | --- | ---: |",
    ]
    for (evidence, allowed), count in sorted(counts.items()):
        lines.append(f"| {evidence} | {allowed} | {count} |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


def write_round_docs(
    gate_path: Path,
    reflection_path: Path,
    next_plan_path: Path,
    commands_path: Path,
    jobs_path: Path,
    failed_jobs_path: Path,
    gate: dict[str, Any],
    baseline: dict[str, Any],
    compare_rows: list[dict[str, str]],
) -> None:
    round_dir = reflection_path.parent
    round_dir.mkdir(parents=True, exist_ok=True)
    write_json(gate_path, gate)
    write_json(round_dir / "gate_status.json", gate)
    certified = [r.get("env", "") for r in compare_rows if r.get("adapter_certification_status") == "PASS_ADAPTER_CERTIFICATION"]
    reflection = [
        "# Round 003 Reflection",
        "",
        f"Generated at: {now()}",
        "",
        "## Primary Question",
        PRIMARY_QUESTION,
        "",
        "## Result",
        f"- Baseline certification: {gate['baseline_certification']}",
        f"- Adapter certification: {gate['adapter_certification']}",
        f"- Certified envs: {', '.join(certified) if certified else 'none'}",
        "",
        "## What Was Done",
        "- Audited official GAS artifact availability and local lineage.",
        "- Evaluated official full-budget GAS artifacts with the official evaluator where artifacts were certification-ready.",
        "- Evaluated the BARS adapter in official-control mode with fallback_mode=none on certified baseline rows only.",
        "- Reclassified prior evidence under the Round003 gates.",
        "",
        "## Interpretation Boundary",
        f"- Scientific interpretation: {gate['scientific_interpretation']}",
        "- No p_bridge or integrated BARS-v3 was run.",
        "- Boundary remains diagnostic-only.",
    ]
    reflection_path.write_text("\n".join(reflection) + "\n")
    if gate["next_round"] == "CERTIFIED_ORACLE_SCAN":
        plan = [
            "# Round 004 Next Plan",
            "",
            "Run certified failure diagnostics and oracle-headroom scan only on Round003 certified envs.",
            "",
            f"- Certified envs: {', '.join(certified)}",
            "- Use fallback_mode=none for planner evidence.",
            "- Do not train p_bridge until PASS_ORACLE_HEADROOM.",
            "- Keep medium tasks on artifact acquisition or full-budget training until their own baseline and adapter certification pass.",
        ]
    elif gate["next_round"] == "ADAPTER_REPAIR":
        plan = [
            "# Round 004 Next Plan",
            "",
            "Repair the BARS adapter/evaluation loop against certified official GAS rows. Do not run oracle-headroom as scientific evidence.",
        ]
    else:
        plan = [
            "# Round 004 Next Plan",
            "",
            "Continue artifact acquisition or full-budget medium training. Scientific interpretation remains on hold.",
        ]
    next_plan_path.write_text("\n".join(plan) + "\n")
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export PYTHONPATH=$PWD",
        "export WANDB_MODE=disabled",
        "export WANDB_DISABLED=true",
        "python scripts/round003_audit_official_gas_artifacts.py",
        "bash scripts/round003_run_official_gas_eval.sh ENVS=antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0 SEEDS=0 TASK_IDS=1,2,3,4,5 EVAL_EPISODES_PER_TASK=50 USE_OFFICIAL_ARTIFACTS=1 FULL_BUDGET_ONLY=1 ROUND=003",
        "python scripts/round003_analyze_certification.py",
        "bash scripts/round003_run_bars_adapter_eval.sh ENVS=$(python - <<'PY'\nimport json\nx=json.load(open('reports/round_003_baseline_certification.json'))\nprint(','.join(x.get('certified_envs', [])))\nPY\n) SEEDS=0 TASK_IDS=1,2,3,4,5 EVAL_EPISODES_PER_TASK=50 VARIANT=gas_shortest_official_control FALLBACK_MODE=none USE_OFFICIAL_ARTIFACTS=1 ROUND=003",
        "python scripts/round003_compare_official_vs_adapter.py",
        "python scripts/round003_finalize.py",
    ]
    commands_path.write_text("\n".join(commands) + "\n")
    jobs = [
        "job\tstatus\treport",
        "artifact_audit\tcompleted\treports/round_003_official_artifact_audit.md",
        f"official_gas_eval\t{gate['baseline_certification']}\treports/round_003_baseline_certification.md",
        f"adapter_eval\t{gate['adapter_certification']}\treports/round_003_adapter_certification.md",
        "prior_reclassification\tcompleted\treports/round_003_prior_evidence_reclassification.md",
        "gate_status\tcompleted\treports/round_003_gate_status.json",
    ]
    jobs_path.write_text("\n".join(jobs) + "\n")
    failed = ["job,env,status,reason,log"]
    for row in baseline.get("rows", []):
        if row.get("baseline_certification_status") != "PASS_BASELINE_CERTIFICATION":
            failed.append(
                "{job},{env},{status},{reason},{log}".format(
                    job="official_gas_eval",
                    env=row.get("env", ""),
                    status=row.get("baseline_certification_status", ""),
                    reason=str(row.get("certification_reason", "")).replace(",", ";"),
                    log=row.get("log", ""),
                )
            )
    for row in compare_rows:
        if row.get("adapter_certification_status") not in {"PASS_ADAPTER_CERTIFICATION", "SKIP_BASELINE_UNCERTIFIED"}:
            failed.append(
                "{job},{env},{status},{reason},{log}".format(
                    job="adapter_eval",
                    env=row.get("env", ""),
                    status=row.get("adapter_certification_status", ""),
                    reason=str(row.get("reason", "")).replace(",", ";"),
                    log="",
                )
            )
    failed_jobs_path.write_text("\n".join(failed) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-certification", default="reports/round_003_baseline_certification.json")
    parser.add_argument("--official-vs-adapter", default="reports/round_003_official_vs_adapter.csv")
    parser.add_argument("--audit", default="reports/round_003_official_artifact_audit.csv")
    parser.add_argument("--gate-out", default="reports/round_003_gate_status.json")
    args = parser.parse_args()

    baseline = read_json(Path(args.baseline_certification), {})
    compare_rows = read_csv(Path(args.official_vs_adapter))
    audit_rows = read_csv(Path(args.audit))
    baseline_status = baseline.get("baseline_certification", "SKIP_ARTIFACT_UNAVAILABLE")
    adapter_status = adapter_aggregate(compare_rows, baseline_status)
    any_adapter_pass = adapter_status == "PASS_ADAPTER_CERTIFICATION"
    if any_adapter_pass:
        scientific = "ALLOW_CERTIFIED_DIAGNOSTICS_ON_CERTIFIED_ENVS"
        oracle = "READY_FOR_CERTIFIED_ORACLE_SCAN"
        next_round = "CERTIFIED_ORACLE_SCAN"
    elif baseline_status == "PASS_BASELINE_CERTIFICATION":
        scientific = "HOLD_SCIENTIFIC_INTERPRETATION_BASELINE_UNCERTIFIED"
        oracle = "SKIP_BASELINE_UNCERTIFIED"
        next_round = "ADAPTER_REPAIR"
    else:
        scientific = "HOLD_SCIENTIFIC_INTERPRETATION_BASELINE_UNCERTIFIED"
        oracle = "SKIP_BASELINE_UNCERTIFIED"
        next_round = "ARTIFACT_ACQUISITION"
    gate = {
        "round": 3,
        "primary_question": PRIMARY_QUESTION,
        "secondary_question": SECONDARY_QUESTION,
        "baseline_registry": "PASS_BASELINE_REGISTRY",
        "official_artifact_audit": "PASS_ARTIFACT_AUDIT" if audit_rows else "SKIP_ARTIFACT_UNAVAILABLE",
        "baseline_certification": baseline_status,
        "adapter_certification": adapter_status,
        "scientific_interpretation": scientific,
        "oracle_headroom": oracle,
        "p_bridge": "SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM",
        "boundary": "HOLD_BOUNDARY_DIAGNOSTIC_ONLY",
        "integrated": "SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE",
        "next_round": next_round,
        "details": {
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_status_short": git_value(["status", "--short"]),
            "baseline_certified_envs": baseline.get("certified_envs", []),
            "adapter_pass_envs": [r.get("env", "") for r in compare_rows if r.get("adapter_certification_status") == "PASS_ADAPTER_CERTIFICATION"],
            "adapter_gap_pp": {
                r.get("env", ""): as_float(r.get("adapter_gap_pp"))
                for r in compare_rows
                if r.get("adapter_gap_pp", "") != ""
            },
            "audit_status_counts": dict(Counter(r.get("artifact_status", "") for r in audit_rows)),
        },
    }
    write_medium_plan(
        Path("reports/round_003_medium_full_budget_training_plan.md"),
        Path("commands/round_003_medium_full_budget_training_commands.sh"),
    )
    write_reclassification(
        Path("reports/round_003_prior_evidence_reclassification.csv"),
        Path("reports/round_003_prior_evidence_reclassification.md"),
        baseline,
        compare_rows,
    )
    write_round_docs(
        Path(args.gate_out),
        Path("rounds/round_003/reflection.md"),
        Path("rounds/round_003/next_plan.md"),
        Path("rounds/round_003/commands.sh"),
        Path("rounds/round_003/jobs.tsv"),
        Path("rounds/round_003/failed_jobs.csv"),
        gate,
        baseline,
        compare_rows,
    )
    print(json.dumps({"gate": args.gate_out, "next_round": gate["next_round"], "adapter_certification": adapter_status}, sort_keys=True))


if __name__ == "__main__":
    main()
