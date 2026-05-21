#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_PATH = Path("research_state/bars_research_state.json")
EXPERIMENT_LEDGER = Path("research_state/experiment_ledger.jsonl")


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


def round_name(round_num: int) -> str:
    return f"round_{round_num:03d}"


def run_cmd(cmd: list[str], log_path: Path, timeout_sec: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))
    env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH','')}"
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout_sec)
    return int(proc.returncode)


def classify_error(log_path: Path) -> str:
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    checks = [
        ("dependency/import", ["ModuleNotFoundError", "ImportError", "No module named"]),
        ("CUDA/JAX/MuJoCo", ["CUDA", "cudnn", "XLA", "jaxlib", "MuJoCo", "mujoco", "EGL", "GLFW"]),
        ("env/dataset", ["ogbench", "dataset", "No registered env", "Environment"]),
        ("checkpoint/artifact", ["checkpoint", "params_", "keygraph", "artifact", "pickle", "Missing GAS keygraph"]),
        ("protocol", ["primary_failure_type", "label", "protocol"]),
    ]
    low = text.lower()
    for name, needles in checks:
        if any(n.lower() in low for n in needles):
            return name
    return "unknown"


def command_rows(round_dir: Path, rn: str, track: str) -> list[dict[str, str]]:
    if track == "oracle_headroom":
        return [
            {
                "job_id": "smoke_compileall",
                "track": "oracle_headroom",
                "command": "python -m compileall bars scripts",
                "log": str(round_dir / "logs" / "compileall.log"),
            },
            {
                "job_id": "oracle_headroom_scan_v2_pilot",
                "track": "oracle_headroom",
                "command": f"bash scripts/stage25_oracle_headroom_scan_v2.sh ENVS=scene-play-v0 SEEDS=0 GRAPH_IDS=G3 TOP_K_BRIDGE=2 MAX_SOURCES=20 EDGE_EXEC_PILOT=1 QUICK=1 ORACLE_REPORTS_ROOT=reports/{rn}_oracle_scan_tmp ARTIFACT_ROOT=artifacts/stage25_autoresearch_{rn} WAIT=1",
                "log": str(round_dir / "logs" / "oracle_headroom_scan_v2.log"),
            },
            {
                "job_id": "stage25_analyze_oracle_context",
                "track": "oracle_headroom",
                "command": f"python scripts/stage25_analyze.py --reports-root reports --oracle-reports-root reports/{rn}_oracle_scan_tmp --oracle-artifact-root artifacts/stage25_autoresearch_{rn} --failure-atlas reports/stage25_failure_atlas_all_variants.csv --min-episodes 100",
                "log": str(round_dir / "logs" / "stage25_analyze_oracle.log"),
            },
        ]
    return [
        {
            "job_id": "smoke_compileall",
            "track": "protocol_repair",
            "command": "python -m compileall bars scripts",
            "log": str(round_dir / "logs" / "compileall.log"),
        },
        {
            "job_id": "failure_atlas_all_variants",
            "track": "protocol_repair",
            "command": f"python scripts/stage25_enrich_failure_atlas_all_variants.py --eval-roots runs_stage24_reachability_confirm,runs_stage24_local_drift --out reports/{rn}_failure_atlas_all_variants.csv --summary-out reports/{rn}_failure_atlas_summary.csv --integrity-out reports/{rn}_label_integrity.json --min-episodes 100",
            "log": str(round_dir / "logs" / "failure_atlas.log"),
        },
        {
            "job_id": "stage25_analyze_round_context",
            "track": "protocol_repair",
            "command": f"python scripts/stage25_analyze.py --reports-root reports --failure-atlas reports/{rn}_failure_atlas_all_variants.csv --min-episodes 100",
            "log": str(round_dir / "logs" / "stage25_analyze.log"),
        },
    ]


def render_round_docs(round_dir: Path, round_num: int, primary_question: str) -> None:
    rn = round_name(round_num)
    (round_dir / "README.md").write_text(
        f"# {rn}\n\nPrimary question: {primary_question}\n\nTrack: protocol_repair\n\n"
        "This round repairs and audits all-variant failure-atlas labels using Stage24 eval roots.\n"
    )
    (round_dir / "plan.md").write_text(
        "\n".join(
            [
                f"# {rn} Plan",
                "",
                f"Primary question: {primary_question}",
                "",
                "Experiments:",
                "- Static compile smoke.",
                "- Stage25 all-variant failure-atlas enrichment on Stage24 reachability/local-drift roots.",
                "- Stage25 analyzer smoke with the round atlas.",
                "",
                "Gates:",
                "- PASS_FAILURE_LABEL_QUALITY if failed rows have labels, complete cells are present, and unclassified rate <= 20%.",
                "- Otherwise keep protocol repair active or stop if unresolved.",
            ]
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--max-wall-hours", type=float, default=12)
    p.add_argument("--max-gpu-hours", type=float, default=72)
    p.add_argument("--max-parallel-jobs", type=int, default=12)
    args = p.parse_args()

    state = read_json(STATE_PATH, {})
    if not state:
        raise SystemExit("research_state missing; run scripts/autoresearch_init.py first")
    if state.get("global_decision") != "CONTINUE":
        print(state.get("global_decision"))
        return
    max_rounds = int(state.get("limits", {}).get("max_rounds", 8))
    round_num = int(args.round or (int(state.get("round", 0)) + 1))
    if round_num > max_rounds:
        state["global_decision"] = "STOP_MAX_ROUNDS_REACHED"
        state["stop_reason"] = f"round {round_num} exceeds max_rounds {max_rounds}"
        write_json(STATE_PATH, state)
        print("STOP_MAX_ROUNDS_REACHED")
        return

    rn = round_name(round_num)
    round_dir = Path("rounds") / rn
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "logs").mkdir(exist_ok=True)
    plan = state.get("next_round_plan") or {}
    track = str(plan.get("track") or ("protocol_repair" if round_num == 1 else "oracle_headroom"))
    primary_question = str(plan.get("primary_question") or "Are all Stage24 variants labeled consistently enough for autonomous decisions?")
    render_round_docs(round_dir, round_num, primary_question)
    rows = command_rows(round_dir, rn, track)
    status_by_job = {r["job_id"]: {"status": "PENDING", "returncode": "", "error_class": "", "command": r["command"]} for r in rows}
    jobs_path = round_dir / "jobs.tsv"
    with open(jobs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "track", "command", "log", "status", "returncode", "error_class"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "status": "PENDING", "returncode": "", "error_class": ""})
    (round_dir / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(r["command"] for r in rows) + "\n")
    os.chmod(round_dir / "commands.sh", 0o755)

    failed: list[dict[str, str]] = []
    completed = 0
    for row in rows:
        cmd = shlex.split(row["command"])
        log_path = Path(row["log"])
        rc = run_cmd(cmd, log_path, timeout_sec=int(args.max_wall_hours * 3600))
        status = "COMPLETED" if rc == 0 else "FAILED"
        err = "" if rc == 0 else classify_error(log_path)
        status_by_job[row["job_id"]] = {"status": status, "returncode": rc, "error_class": err, "command": row["command"]}
        completed += int(rc == 0)
        if rc != 0:
            failed.append({"job_id": row["job_id"], "track": row["track"], "log": str(log_path), "error_class": err})
        # Rewrite full job table each step for inspectability.
        with open(jobs_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["job_id", "track", "command", "log", "status", "returncode", "error_class"], delimiter="\t")
            writer.writeheader()
            for r in rows:
                s = status_by_job[r["job_id"]]
                writer.writerow({**r, "command": s["command"], "status": s["status"], "returncode": s["returncode"], "error_class": s["error_class"]})
    failed_path = round_dir / "failed_jobs.csv"
    with open(failed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "track", "log", "error_class"])
        writer.writeheader()
        writer.writerows(failed)
    (round_dir / "smoke_results.md").write_text(
        f"# {rn} Smoke Results\n\n- compileall: {'COMPLETED' if not any(f['job_id']=='smoke_compileall' for f in failed) else 'FAILED'}\n"
        f"- failed_jobs: {len(failed)}\n"
    )
    state["round"] = round_num
    state["active_scientific_question"] = primary_question
    state["updated_at"] = now()
    write_json(STATE_PATH, state)
    append_jsonl(
        EXPERIMENT_LEDGER,
        {
            "round": round_num,
            "experiment_id": f"E{round_num:03d}",
            "track": track,
            "commands_file": str(round_dir / "commands.sh"),
            "started_at": "",
            "completed_at": now(),
            "status": "COMPLETED" if not failed else "PARTIAL",
            "n_jobs": len(rows),
            "n_failed_jobs": len(failed),
            "result_files": [
                f"reports/{rn}_failure_atlas_all_variants.csv",
                f"reports/{rn}_failure_atlas_summary.csv",
                f"reports/{rn}_label_integrity.json",
            ],
            "primary_metrics": {},
            "gate_status": {},
        },
    )
    print(json.dumps({"round": rn, "completed": completed, "failed_jobs": len(failed)}, sort_keys=True))


if __name__ == "__main__":
    main()
