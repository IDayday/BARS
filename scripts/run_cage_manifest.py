#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from build_cage_eval_command import command_to_string


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run initialized jobs from a CAGE experiment manifest.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--max_jobs", type=int, default=1)
    parser.add_argument("--parallel_jobs", type=int, default=1)
    parser.add_argument("--rerun_succeeded", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def status_path_for(manifest_path: Path, rows: list[dict[str, Any]]) -> Path:
    if rows and rows[0].get("output_root"):
        output_root = Path(rows[0]["output_root"])
    else:
        output_root = manifest_path.parent.parent
    name = manifest_path.name.replace("manifest", "status")
    if name == manifest_path.name:
        name = manifest_path.stem + "_status.jsonl"
    return output_root / "status" / name


def write_status(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_existing_status(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for row in load_jsonl(path):
        job_id = row.get("job_id")
        if job_id:
            latest[str(job_id)] = row
    return latest


def skipped_record(row: dict[str, Any], status: str, reason: str | None = None) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "job_id": row.get("job_id"),
        "env_name": row.get("env_name"),
        "seed": row.get("seed"),
        "variant": row.get("variant"),
        "command": row.get("command"),
        "start_time": now,
        "end_time": now,
        "return_code": None,
        "status": status,
        "skip_reason": reason,
        "stdout_path": None,
        "stderr_path": None,
        "cage_trace_path": row.get("cage_trace_path"),
        "result_path": row.get("result_path"),
    }


def run_job(row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    command = row.get("command")
    if not isinstance(command, list) or not command:
        return skipped_record(row, "skipped", row.get("error") or "missing command")
    command = list(command)
    if command and command[0] == "python":
        command[0] = sys.executable

    output_root = Path(row["output_root"])
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{row['job_id']}.stdout"
    stderr_path = log_root / f"{row['job_id']}.stderr"
    Path(row["result_path"]).parent.mkdir(parents=True, exist_ok=True)
    if row.get("cage_trace_path"):
        Path(row["cage_trace_path"]).parent.mkdir(parents=True, exist_ok=True)

    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        env = os.environ.copy()
        env.setdefault("WANDB_MODE", "disabled")
        with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
            proc = subprocess.run(
                command,
                cwd=str(repo_root / "external_src" / "GAS"),
                env=env,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                check=False,
            )
        return_code = int(proc.returncode)
        status = "succeeded" if proc.returncode == 0 else "failed"
        error = None
    except Exception as exc:
        return_code = None
        status = "failed"
        error = repr(exc)
    end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {
        "job_id": row["job_id"],
        "env_name": row["env_name"],
        "seed": row["seed"],
        "variant": row["variant"],
        "command": command,
        "start_time": start,
        "end_time": end,
        "return_code": return_code,
        "status": status,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "cage_trace_path": row.get("cage_trace_path"),
        "result_path": row.get("result_path"),
    }
    if error is not None:
        record["error"] = error
    return record


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest_path)
    rows = load_jsonl(manifest_path)
    if args.dry_run:
        for row in rows:
            if row.get("status") == "initialized":
                print(command_to_string(row.get("command")))
            else:
                print(f"# skip {row.get('job_id')}: {row.get('status')} {row.get('error', '')}".rstrip())
        return 0

    status_path = status_path_for(manifest_path, rows)
    existing_status = load_existing_status(status_path)
    repo_root = Path(args.repo_root)
    pending: list[dict[str, Any]] = []
    for row in rows:
        job_id = str(row.get("job_id"))
        prior = existing_status.get(job_id)
        if row.get("status") != "initialized":
            write_status(status_path, skipped_record(row, "skipped", row.get("status")))
            continue
        if prior and prior.get("status") == "succeeded" and not args.rerun_succeeded:
            continue
        pending.append(row)
    limit = min(len(pending), int(args.max_jobs))
    selected = pending[:limit]
    if int(args.parallel_jobs) <= 1:
        for row in selected:
            write_status(status_path, run_job(row, repo_root))
    else:
        workers = max(1, int(args.parallel_jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_job, row, repo_root) for row in selected]
            for future in as_completed(futures):
                write_status(status_path, future.result())
    print(json.dumps({"status_path": str(status_path), "executed_jobs": len(selected), "parallel_jobs": int(args.parallel_jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
