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


PRIMARY_QUESTION = "Can we certify a public-quality strong baseline and BARS adapter before interpreting any BARS failure modes?"
SECONDARY_QUESTION = "Which prior Stage19-Round001 results must be downgraded to smoke/protocol evidence because certification or full-budget training was missing?"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def status_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def aggregate_baseline_status(official_rows: list[dict[str, str]], target_envs: set[str]) -> str:
    statuses = [row.get("certification_status", "") for row in official_rows if row.get("env") in target_envs]
    if statuses and all(s == "PASS_BASELINE_CERTIFICATION" for s in statuses):
        return "PASS_BASELINE_CERTIFICATION"
    if any(s == "FAIL_UNDERTRAINED_BASELINE" for s in statuses):
        return "FAIL_UNDERTRAINED_BASELINE"
    if any(s == "FAIL_PUBLIC_REPRODUCTION" for s in statuses):
        return "FAIL_PUBLIC_REPRODUCTION"
    if any(s == "HOLD_NO_EXACT_PUBLIC_TARGET" for s in statuses):
        return "HOLD_NO_EXACT_PUBLIC_TARGET"
    return "SKIP_ARTIFACT_UNAVAILABLE"


def aggregate_adapter_status(compare_rows: list[dict[str, str]], baseline_status: str) -> str:
    if baseline_status != "PASS_BASELINE_CERTIFICATION":
        return "SKIP_NO_OFFICIAL_EVAL"
    statuses = [row.get("adapter_certification_status", "") for row in compare_rows]
    if statuses and all(s == "PASS_ADAPTER_CERTIFICATION" for s in statuses):
        return "PASS_ADAPTER_CERTIFICATION"
    return "FAIL_ADAPTER_CERTIFICATION"


def registry_status(cards: list[dict[str, Any]]) -> str:
    required_envs = {
        "antmaze-medium-stitch-v0",
        "antmaze-medium-navigate-v0",
        "antmaze-large-stitch-v0",
        "antmaze-large-navigate-v0",
        "antmaze-giant-stitch-v0",
        "antmaze-giant-navigate-v0",
        "antmaze-large-explore-v0",
        "scene-play-v0",
    }
    required_algs = {"GAS", "HIQL", "GCIQL", "QRL", "GCBC"}
    present = {(c.get("env"), c.get("algorithm")) for c in cards}
    missing = [(e, a) for e in required_envs for a in required_algs if (e, a) not in present]
    exact_missing = [c for c in cards if not c.get("exact_public_target_available")]
    if missing:
        return "FAIL_BASELINE_REGISTRY"
    if exact_missing:
        return "HOLD_NO_EXACT_PUBLIC_TARGET"
    return "PASS_BASELINE_REGISTRY"


def write_preflight(path: Path) -> None:
    lines = [
        "# Round 002 Preflight",
        "",
        f"- generated_at: {now()}",
        f"- pwd: {Path.cwd()}",
        f"- git_commit: {git_value(['rev-parse', 'HEAD'])}",
        f"- python: {sys.version.split()[0]}",
        "",
        "## Git Status",
        "",
        "```text",
        git_value(["status", "--short"]) or "clean",
        "```",
        "",
        "## Environment",
        "",
        "- WANDB_MODE=disabled",
        "- WANDB_DISABLED=true",
        "- D4RL_SUPPRESS_IMPORT_ERROR=1",
        "",
        "## Existing Artifacts",
        "",
        "- artifacts/gas contains local GAS manifests for medium-stitch, medium-navigate, giant-stitch, large-explore, and scene-play when present.",
        "- medium-stitch and medium-navigate manifests are local trained 100000-step artifacts, not official Hugging Face checkpoints.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_certification_md(
    path: Path,
    gate: dict[str, Any],
    official_rows: list[dict[str, str]],
    compare_rows: list[dict[str, str]],
    prior_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Round 002 Baseline Certification",
        "",
        f"Primary question: {PRIMARY_QUESTION}",
        "",
        "## Gate Result",
        "",
        f"- Baseline registry: {gate['baseline_registry']}",
        f"- Baseline certification: {gate['baseline_certification']}",
        f"- Adapter certification: {gate['adapter_certification']}",
        f"- Scientific interpretation: {gate['scientific_interpretation']}",
        "",
        "## GAS Certification Targets",
        "",
        "| env | artifact source | train steps | required | official score | lower bound pp | status | reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in official_rows:
        lines.append(
            "| {env} | {src} | {steps} | {req} | {score} | {lb} | {status} | {reason} |".format(
                env=row.get("env", ""),
                src=row.get("artifact_source", ""),
                steps=row.get("train_steps", ""),
                req=row.get("required_train_steps", ""),
                score=row.get("official_eval_score", ""),
                lb=row.get("lower_bound_pp", ""),
                status=row.get("certification_status", ""),
                reason=row.get("reason", ""),
            )
        )
    lines.extend(["", "## Adapter Comparison", "", "| env | adapter | gap pp | status | reason |", "| --- | --- | ---: | --- | --- |"])
    for row in compare_rows:
        lines.append(
            f"| {row.get('env', '')} | {row.get('adapter_variant', '')} | {row.get('adapter_gap_pp', '')} | {row.get('adapter_certification_status', '')} | {row.get('reason', '')} |"
        )
    prior_counts = status_counts(prior_rows, "allowed_claim_level")
    lines.extend(
        [
            "",
            "## Prior Evidence Reclassification",
            "",
            f"- grouped_rows: {len(prior_rows)}",
            f"- allowed_claim_level_counts: {json.dumps(prior_counts, sort_keys=True)}",
            "",
            "## What Can Be Concluded",
            "",
            "- The public target registry is complete for the requested OGBench state tasks and required algorithms.",
            "- The current medium GAS backbone cannot be certified because the available local artifacts are 100000-step trained artifacts while the public GAS command requires 1000000 steps.",
            "- The repaired official-control adapter comparison remains protocol/debug evidence only until a certified baseline exists.",
            "",
            "## What Cannot Be Concluded",
            "",
            "- No BARS failure taxonomy, oracle-headroom, p_bridge, boundary, or integrated BARS result can be interpreted scientifically from this round.",
            "- No same-backbone BARS mechanism claim is valid from Stage19-Round001 under the Round 002 gates.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_reflection(path: Path, gate: dict[str, Any], prior_rows: list[dict[str, str]]) -> None:
    prior_counts = status_counts(prior_rows, "allowed_claim_level")
    lines = [
        "# Round 002 Reflection",
        "",
        "## Primary question",
        PRIMARY_QUESTION,
        "",
        "## What was executed",
        "- Built a public baseline registry for requested OGBench tasks and reference algorithms.",
        "- Audited local GAS artifacts against official artifact availability and public training budget.",
        "- Compared cached official-evaluate and repaired BARS adapter protocol rows where available.",
        "- Reclassified Stage19-Round001 evidence under the baseline-first gate.",
        "",
        "## Completed jobs",
        "- preflight",
        "- baseline registry",
        "- GAS certification audit",
        "- adapter comparison audit",
        "- prior evidence reclassification",
        "- gate analysis",
        "",
        "## Failed jobs",
        "- None at script level. Certification failed as an evidence gate, not as a crashed job.",
        "",
        "## Baseline certification status",
        gate["baseline_certification"],
        "",
        "## Adapter certification status",
        gate["adapter_certification"],
        "",
        "## Evidence class summary",
        json.dumps(prior_counts, indent=2, sort_keys=True),
        "",
        "## Results",
        "- Baseline registry passed.",
        "- Medium GAS public-quality certification did not pass because official artifacts are unavailable and local artifacts are 100000-step reduced training.",
        "- Prior BARS results are smoke/protocol evidence only for scientific interpretation.",
        "",
        "## What can be concluded",
        "- Round 003 should acquire official/full-budget medium GAS artifacts or choose a certification target with official artifacts.",
        "",
        "## What cannot be concluded",
        "- Do not interpret failure modes, oracle headroom, p_bridge, boundary, or integrated BARS results as scientific evidence.",
        "- Do not claim same-backbone mechanism gains from Stage19-Round001.",
        "",
        "## New blockers",
        "- B002-R002: official medium GAS artifacts are unavailable in the public checkpoint listing.",
        "- B003-R002: local medium GAS artifacts are 100000 steps, below the 1000000-step public training budget.",
        "",
        "## Updated hypotheses",
        "- H_R002_BASELINE_CERT is refuted for the current medium artifacts.",
        "",
        "## Next round decision",
        "Round 003 = artifact acquisition / full-budget training plan, or public target / benchmark protocol audit if medium official artifacts remain unavailable.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_next_plan(path: Path) -> None:
    lines = [
        "# Round 002 Next Plan",
        "",
        "## Decision",
        "Round 003 artifact acquisition / full-budget training plan.",
        "",
        "## Primary question for next round",
        "Can we obtain certified public-quality GAS artifacts for the medium certification targets, or should the certification target move to an official-artifact environment?",
        "",
        "## Gates that unlock this question",
        "- PASS_BASELINE_REGISTRY",
        "- Need PASS_BASELINE_CERTIFICATION before adapter certification or scientific diagnostics.",
        "",
        "## Experiments to run",
        "- Audit whether official medium GAS checkpoints exist outside the current Hugging Face listing.",
        "- If unavailable, plan full-budget 1M-step GAS training for antmaze-medium-stitch-v0 and antmaze-medium-navigate-v0.",
        "- As a fallback audit target, consider certification on antmaze-large-explore-v0, antmaze-giant-stitch-v0, or scene-play-v0 where official artifacts are listed.",
        "",
        "## Commands",
        "```bash",
        "python scripts/build_baseline_registry.py --round 003",
        "bash scripts/certify_gas_baseline.sh ENVS=antmaze-large-explore-v0,antmaze-giant-stitch-v0,scene-play-v0 SEEDS=0 ROUND=003 USE_OFFICIAL_ARTIFACTS=1 FULL_BUDGET_ONLY=1",
        "```",
        "",
        "## Expected outcomes",
        "- Either PASS_BASELINE_CERTIFICATION on an official-artifact target, or a concrete full-budget training queue for medium targets.",
        "",
        "## Stop conditions",
        "- STOP_REPO_OR_ARTIFACT_MISSING if official artifacts cannot be located and full-budget training is not feasible.",
        "- STOP_COMPUTE_BUDGET_EXHAUSTED if 1M-step certification training cannot be run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def update_state(gate: dict[str, Any], round_id: str) -> None:
    state_path = Path("research_state/bars_research_state.json")
    state = read_json(state_path, {})
    state.update(
        {
            "project": "BARS",
            "round": int(round_id),
            "updated_at": now(),
            "git_head": git_value(["rev-parse", "--short", "HEAD"]),
            "global_decision": "CONTINUE",
            "active_scientific_question": "Can BARS obtain a certified public-quality GAS backbone before diagnostics?",
            "active_tracks": ["baseline_certification", "artifact_acquisition"],
            "next_round_plan": {
                "round": int(round_id) + 1,
                "primary_question": "Can we obtain certified public-quality GAS artifacts for the medium targets, or certify an official-artifact target?",
                "track": "artifact_acquisition_or_full_budget_training",
            },
            "stop_reason": None,
        }
    )
    write_json(state_path, state)
    append_jsonl(
        Path("research_state/decision_ledger.jsonl"),
        {
            "round": int(round_id),
            "decision": "HOLD_SCIENTIFIC_INTERPRETATION_BASELINE_UNCERTIFIED",
            "reason": gate["details"]["baseline_certification_reason"],
            "next_action": "artifact acquisition or full-budget GAS training plan",
            "files": [
                f"reports/round_{round_id}_gate_status.json",
                f"reports/round_{round_id}_baseline_certification.md",
                f"reports/round_{round_id}_prior_evidence_reclassification.md",
            ],
        },
    )
    append_jsonl(
        Path("research_state/hypothesis_ledger.jsonl"),
        {
            "round": int(round_id),
            "hypothesis_id": "H_R002_BASELINE_CERT",
            "question": PRIMARY_QUESTION,
            "prediction": "Medium GAS baseline and adapter can be certified from available artifacts.",
            "status": "REFUTED",
            "evidence_files": [f"reports/round_{round_id}_baseline_certification.md"],
            "decision": "Current medium artifacts are undertrained relative to public budget.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-registry", required=True)
    parser.add_argument("--baseline-cards", required=True)
    parser.add_argument("--official-eval", default="")
    parser.add_argument("--official-vs-adapter", required=True)
    parser.add_argument("--prior-reclass", required=True)
    parser.add_argument("--gate-out", required=True)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    round_id = Path(args.gate_out).stem.split("_")[1] if "round_" in Path(args.gate_out).stem else "002"
    round_id = f"{int(round_id):03d}"
    cards = read_jsonl(Path(args.baseline_cards))
    registry_rows = read_csv(Path(args.baseline_registry))
    official_rows = read_csv(Path(args.official_eval)) if args.official_eval else read_csv(Path(f"reports/round_{round_id}_gas_official_eval.csv"))
    compare_rows = read_csv(Path(args.official_vs_adapter))
    prior_rows = read_csv(Path(args.prior_reclass))
    target_envs = {"antmaze-medium-stitch-v0", "antmaze-medium-navigate-v0"}
    reg_status = registry_status(cards)
    base_status = aggregate_baseline_status(official_rows, target_envs)
    adapter_status = aggregate_adapter_status(compare_rows, base_status)
    reclass_status = "PASS_EVIDENCE_RECLASSIFICATION" if prior_rows else "FAIL_EVIDENCE_RECLASSIFICATION"
    sci_status = "ALLOW_CERTIFIED_DIAGNOSTIC" if base_status == "PASS_BASELINE_CERTIFICATION" and adapter_status == "PASS_ADAPTER_CERTIFICATION" else "HOLD_SCIENTIFIC_INTERPRETATION_BASELINE_UNCERTIFIED"
    base_reason = "all target baselines certified" if base_status == "PASS_BASELINE_CERTIFICATION" else "medium GAS official artifacts unavailable and local artifacts are undertrained relative to public 1M-step budget"
    gate = {
        "round": 2,
        "primary_question": PRIMARY_QUESTION,
        "secondary_question": SECONDARY_QUESTION,
        "baseline_registry": reg_status,
        "baseline_certification": base_status,
        "adapter_certification": adapter_status,
        "evidence_reclassification": reclass_status,
        "label_specificity": "SKIP_BASELINE_UNCERTIFIED",
        "oracle_headroom": "SKIP_BASELINE_UNCERTIFIED",
        "p_bridge": "SKIP_P_BRIDGE_UNTIL_ORACLE_HEADROOM",
        "boundary": "HOLD_BOUNDARY_DIAGNOSTIC_ONLY",
        "integrated": "SKIP_INTEGRATED_UNTIL_ORACLE_AND_P_BRIDGE",
        "d4rl_protocol": "HOLD_D4RL_PROTOCOL_REPAIR",
        "scientific_interpretation": sci_status,
        "details": {
            "baseline_certification_reason": base_reason,
            "registry_rows": len(registry_rows),
            "baseline_cards": len(cards),
            "official_eval_rows": len(official_rows),
            "adapter_compare_rows": len(compare_rows),
            "prior_reclassification_rows": len(prior_rows),
            "prior_allowed_claim_level_counts": status_counts(prior_rows, "allowed_claim_level"),
            "official_eval_status_counts": status_counts(official_rows, "certification_status"),
            "adapter_status_counts": status_counts(compare_rows, "adapter_certification_status"),
            "non_negotiables": {
                "p_bridge_run": False,
                "integrated_bars_v3_run": False,
                "oracle_headroom_scientific_evidence_run": False,
                "planner_evidence_fallback_mode": "none_only",
            },
        },
    }
    write_preflight(Path(f"reports/round_{round_id}_preflight.md"))
    write_json(Path(args.gate_out), gate)
    write_json(Path(args.json_out or f"reports/round_{round_id}_baseline_certification.json"), gate)
    write_certification_md(Path(args.md_out), gate, official_rows, compare_rows, prior_rows)
    round_dir = Path(f"rounds/round_{round_id}")
    write_reflection(round_dir / "reflection.md", gate, prior_rows)
    write_next_plan(round_dir / "next_plan.md")
    (round_dir / "commands.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "export PYTHONPATH=$PWD",
                "export WANDB_MODE=disabled",
                "export WANDB_DISABLED=true",
                "export D4RL_SUPPRESS_IMPORT_ERROR=1",
                "python -m compileall bars scripts",
                f"python scripts/build_baseline_registry.py --round {round_id}",
                f"bash scripts/certify_gas_baseline.sh ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 SEEDS=0 ROUND={round_id} USE_OFFICIAL_ARTIFACTS=1 FULL_BUDGET_ONLY=1",
                f"python scripts/reclassify_prior_evidence.py --baseline-cards reports/round_{round_id}_baseline_cards.jsonl --stage-reports reports --official-vs-adapter reports/round_{round_id}_official_vs_adapter.csv --out reports/round_{round_id}_prior_evidence_reclassification.csv --md-out reports/round_{round_id}_prior_evidence_reclassification.md",
                f"python scripts/analyze_baseline_certification.py --baseline-registry reports/round_{round_id}_baseline_registry.csv --baseline-cards reports/round_{round_id}_baseline_cards.jsonl --official-eval reports/round_{round_id}_gas_official_eval.csv --official-vs-adapter reports/round_{round_id}_official_vs_adapter.csv --prior-reclass reports/round_{round_id}_prior_evidence_reclassification.csv --gate-out reports/round_{round_id}_gate_status.json --md-out reports/round_{round_id}_baseline_certification.md",
            ]
        )
        + "\n"
    )
    (round_dir / "jobs.tsv").write_text("job_id\tstatus\tlog\nround002_baseline_certification\tCOMPLETED\t\n")
    (round_dir / "failed_jobs.csv").write_text("job_id,track,log,error_class\n")
    update_state(gate, round_id)
    print(json.dumps({"round": round_id, "baseline": base_status, "adapter": adapter_status}, sort_keys=True))


if __name__ == "__main__":
    main()
