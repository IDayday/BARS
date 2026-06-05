#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from build_cage_eval_command import command_to_string


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run initialized jobs from a CAGE experiment manifest.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--max_jobs", type=int, default=1)
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
    repo_root = Path(args.repo_root)
    ran = 0
    for row in rows:
        if row.get("status") != "initialized":
            write_status(status_path, skipped_record(row, "skipped", row.get("status")))
            continue
        if ran >= args.max_jobs:
            continue
        record = run_job(row, repo_root)
        write_status(status_path, record)
        ran += 1
    print(json.dumps({"status_path": str(status_path), "executed_jobs": ran}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
