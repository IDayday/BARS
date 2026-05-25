#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "corrupt_status", "error": repr(exc)}


def process_alive(pid: str | int | None) -> bool:
    if pid in (None, ""):
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat = stat_path.read_text(encoding="utf-8", errors="ignore")
        fields = stat.split()
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    except FileNotFoundError:
        return False
    except Exception:
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def nvidia_smi_rows() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return [{"index": "n/a", "name": "nvidia-smi failed", "memory_used": "", "memory_free": "", "util": ""}]
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used": parts[2],
                    "memory_free": parts[3],
                    "util": parts[4],
                }
            )
    return rows


def last_csv_row(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = deque(csv.DictReader(f), maxlen=1)
        return rows[0] if rows else {}
    except Exception:
        return {}


def latest_file(root: Path, pattern: str) -> Path | None:
    paths = list(root.glob(pattern))
    if not paths:
        return None
    return sorted(paths, key=lambda p: (p.stat().st_mtime, str(p)))[-1]


def latest_train_step(root: Path) -> int:
    row = last_csv_row(latest_file(root, "**/train.csv"))
    raw = row.get("step", "")
    try:
        return int(float(raw))
    except Exception:
        return 0


def latest_eval_csv(out_seed: Path, run_seed: Path) -> Path | None:
    return latest_file(out_seed / "policy", "**/eval.csv") or latest_file(run_seed / "evaluate_gas", "**/eval.csv")


def latest_score(eval_csv: Path | None) -> float | None:
    row = last_csv_row(eval_csv)
    raw = row.get("eval/overall_episode.success")
    if raw in (None, ""):
        return None
    try:
        return float(raw) * 100.0
    except Exception:
        return None


def phase_and_steps(args: argparse.Namespace, row: dict[str, str], status: dict[str, Any]) -> dict[str, Any]:
    env = row["env"]
    seed = row["seed"]
    out_seed = Path(args.out_root) / env / f"seed{seed}"
    run_seed = Path(args.run_root) / env / f"seed{seed}"
    config = status.get("config") or {}
    try:
        train_steps = int(config.get("train_steps") or 500000 if env.startswith("visual-") else 1000000)
    except Exception:
        train_steps = 500000 if env.startswith("visual-") else 1000000

    tdr_step = latest_train_step(out_seed / "tdr")
    policy_step = latest_train_step(out_seed / "policy")
    tdr_done = latest_file(out_seed / "tdr", f"**/params_{train_steps}.pkl") is not None
    graph_done = latest_file(out_seed / "graph", "**/keygraph.pkl") is not None
    policy_done = latest_file(out_seed / "policy", f"**/params_{train_steps}.pkl") is not None
    eval_csv = latest_eval_csv(out_seed, run_seed)

    if eval_csv is not None:
        phase = "evaluated"
    elif policy_done:
        phase = "await_eval"
    elif graph_done:
        phase = "policy"
    elif tdr_done:
        phase = "graph"
    elif tdr_step > 0:
        phase = "tdr"
    else:
        phase = "not_started"

    if tdr_done:
        tdr_step = max(tdr_step, train_steps)
    if policy_done:
        policy_step = max(policy_step, train_steps)
    pct = 100.0 * (min(tdr_step, train_steps) + min(policy_step, train_steps)) / max(1, 2 * train_steps)

    return {
        "phase": phase,
        "train_steps": train_steps,
        "tdr_step": tdr_step,
        "policy_step": policy_step,
        "pct": pct,
        "eval_csv": eval_csv,
        "score": latest_score(eval_csv),
    }


def tail_lines(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []


def render_snapshot(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    jobs_path = REPO_ROOT / "reports" / f"round_{args.round}_gas_dynamic_jobs.tsv"
    data_path = REPO_ROOT / "reports" / f"round_{args.round}_ogbench_download_status.tsv"
    run_root = Path(args.run_root)
    out_root = Path(args.out_root)
    pid_path = run_root / "_orchestrator" / "orchestrator.pid"
    orch_pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else ""
    orch_alive = process_alive(orch_pid)

    jobs = read_tsv(jobs_path)
    datasets = read_tsv(data_path)
    status_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    rows_enriched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for row in jobs:
        status_path = REPO_ROOT / row.get("status_file", "")
        status = read_json(status_path)
        display_status = status.get("status") or row.get("status") or "unknown"
        info = phase_and_steps(args, row, status)
        alive = process_alive(row.get("pid"))
        if display_status == "queued" and alive:
            display_status = "launching"
        elif display_status == "launched" and not alive:
            display_status = "queued_after_interrupt"
        status_counts[str(display_status)] += 1
        phase_counts[str(info["phase"])] += 1
        enriched = {**row, **info, "status": display_status, "alive": alive, "error": status.get("error", "")}
        rows_enriched.append(enriched)
        if display_status == "failed":
            failures.append(enriched)
        if info["score"] is not None:
            results.append(enriched)

    ready = [r for r in datasets if r.get("ready") == "True"]
    pending = [r for r in datasets if r.get("ready") != "True"]
    active = [r for r in rows_enriched if r["alive"] or r["status"] in {"launching"}]
    active_sorted = sorted(active, key=lambda r: (int(r.get("priority") or 999), r["env"], int(r["seed"])))[:18]
    failed_sorted = sorted(failures, key=lambda r: (int(r.get("priority") or 999), r["env"], int(r["seed"])))[:18]
    result_sorted = sorted(results, key=lambda r: (r["env"], int(r["seed"])))

    gpu_rows = nvidia_smi_rows()
    event_tail = tail_lines(run_root / "_orchestrator" / "events.jsonl", 8)
    download_tail = tail_lines(run_root / "_orchestrator" / "download.log", 10)

    lines = [
        f"# Round {args.round} GAS Monitor Snapshot",
        "",
        f"- Generated: {now_iso()}",
        f"- Evidence class: `E4_FULL_BUDGET_TRAINED_METHOD` pending completed eval rows.",
        f"- Baseline-only monitor: no BARS mechanism interpretation.",
        f"- Orchestrator PID: `{orch_pid or 'missing'}` (`{'alive' if orch_alive else 'not alive'}`)",
        f"- Jobs table: `{rel(jobs_path)}`",
        f"- Dataset table: `{rel(data_path)}`",
        f"- Artifact root: `{args.out_root}`",
        "",
        "## Queue Status",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for key, count in sorted(status_counts.items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Phase Status", "", "| phase | count |", "| --- | ---: |"])
    for key, count in sorted(phase_counts.items()):
        lines.append(f"| {key} | {count} |")

    lines.extend(
        [
            "",
            "## Datasets",
            "",
            f"- Ready: `{len(ready)}/{len(datasets)}`",
            f"- Pending: `{', '.join(r['env'] for r in pending[:12]) or 'none'}`",
            "",
            "## GPU",
            "",
            "| gpu | used_mb | free_mb | util_pct | name |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in gpu_rows:
        lines.append(f"| {r['index']} | {r['memory_used']} | {r['memory_free']} | {r['util']} | {r['name']} |")

    lines.extend(
        [
            "",
            "## Active Jobs",
            "",
            "| env | seed | gpu | status | phase | tdr | policy | pct | pid | alive |",
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in active_sorted:
        lines.append(
            f"| {r['env']} | {r['seed']} | {r.get('gpu','')} | {r['status']} | {r['phase']} | "
            f"{r['tdr_step']}/{r['train_steps']} | {r['policy_step']}/{r['train_steps']} | {r['pct']:.1f} | "
            f"{r.get('pid','')} | {r['alive']} |"
        )
    if not active_sorted:
        lines.append("| none |  |  |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Failures",
            "",
            "| env | seed | phase | tdr | policy | error |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for r in failed_sorted:
        err = str(r.get("error") or "").replace("|", "/")[:180]
        lines.append(f"| {r['env']} | {r['seed']} | {r['phase']} | {r['tdr_step']} | {r['policy_step']} | {err} |")
    if not failed_sorted:
        lines.append("| none |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Completed Eval Results",
            "",
            "| env | seed | score_pp | eval_csv |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for r in result_sorted[:40]:
        score = "" if r["score"] is None else f"{r['score']:.1f}"
        lines.append(f"| {r['env']} | {r['seed']} | {score} | `{rel(r['eval_csv'])}` |")
    if not result_sorted:
        lines.append("| none |  |  |  |")

    lines.extend(["", "## Download Log Tail", "", "```text"])
    lines.extend(download_tail or ["no download log yet"])
    lines.extend(["```", "", "## Event Tail", "", "```json"])
    lines.extend(event_tail or ["no events yet"])
    lines.extend(["```", ""])

    summary = {
        "time": now_iso(),
        "orchestrator_pid": orch_pid,
        "orchestrator_alive": orch_alive,
        "dataset_ready": len(ready),
        "dataset_total": len(datasets),
        "status_counts": dict(status_counts),
        "phase_counts": dict(phase_counts),
        "active_jobs": len(active),
        "failures": len(failures),
        "completed_eval": len(result_sorted),
    }
    return "\n".join(lines), summary


def emit_snapshot(args: argparse.Namespace) -> None:
    report, summary = render_snapshot(args)
    monitor_root = Path(args.run_root) / "_monitor"
    monitor_root.mkdir(parents=True, exist_ok=True)
    latest_path = REPO_ROOT / "reports" / f"round_{args.round}_gas_dynamic_monitor_latest.md"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(report + "\n", encoding="utf-8")

    with (monitor_root / "monitor.log").open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")
        f.write(report + "\n")
    with (monitor_root / "monitor_snapshots.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(report, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--round", default="006")
    p.add_argument("--run-root", default="runs_round006_gas_dynamic")
    p.add_argument("--out-root", default="artifacts/gas_selftrain_round006")
    p.add_argument("--interval-seconds", type=int, default=1800)
    p.add_argument("--watch", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    while True:
        emit_snapshot(args)
        if not args.watch:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
