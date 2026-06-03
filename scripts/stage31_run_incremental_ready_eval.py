#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from stage30_official_gas_common import write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _episode_csv(root: Path, env_name: str, seed: str) -> Path:
    return root / env_name / f"seed{seed}" / "instrumentation" / "official_gas_episode_traces.csv"


def _done_in_any(env_name: str, seed: str, roots: list[Path]) -> bool:
    return any(_episode_csv(root, env_name, seed).exists() and _episode_csv(root, env_name, seed).stat().st_size > 0 for root in roots)


def _select_ready_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = _read_csv(Path(args.inventory_csv))
    baseline_roots = [Path(x) for x in args.skip_run_roots.split(",") if x.strip()]
    baseline_roots.append(Path(args.out_root))
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("artifact_status") != "READY_OFFICIAL_GAS":
            continue
        env_name = row.get("env_name", "")
        seed = str(int(float(row.get("seed", "0"))))
        if args.env_regex:
            import re

            if not re.search(args.env_regex, env_name):
                continue
        if args.seeds:
            wanted = {x.strip() for x in args.seeds.split(",") if x.strip()}
            if seed not in wanted:
                continue
        if args.skip_existing and _done_in_any(env_name, seed, baseline_roots):
            continue
        selected.append(row)
        if args.limit > 0 and len(selected) >= args.limit:
            break
    return selected


def _launch(row: dict[str, str], args: argparse.Namespace, gpu: str, index: int) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    env_name = row["env_name"]
    seed = str(int(float(row["seed"])))
    out_dir = Path(args.out_root) / env_name / f"seed{seed}" / "instrumentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out_root) / f"{env_name}_seed{seed}.instrumentation.log"
    python_executable = args.python_executable or sys.executable
    command = [
        python_executable,
        "scripts/stage30_official_gas_instrument.py",
        "--artifact-root",
        args.artifact_root,
        "--gas-repo-path",
        args.gas_repo_path,
        "--out-root",
        str(out_dir),
        "--envs",
        env_name,
        "--seeds",
        seed,
        "--task-ids",
        args.task_ids,
        "--max-task-id",
        str(args.max_task_id),
        "--episodes",
        str(args.episodes),
        "--eval-on-cpu",
        str(args.eval_on_cpu),
        "--gpu",
        str(gpu),
        "--fallback-mode",
        args.fallback_mode,
        "--recover-dataset-indices",
        str(args.recover_dataset_indices),
        "--node-map-batch-size",
        str(args.node_map_batch_size),
        "--node-map-tolerance",
        str(args.node_map_tolerance),
    ]
    fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=args.repo_root, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy())
    meta = {
        "stage": "stage31_incremental_ready_eval",
        "evidence_class": "OFFICIAL_GAS_INCREMENTAL_READY_EVAL_STATUS",
        "env_name": env_name,
        "seed": seed,
        "pid": proc.pid,
        "gpu": gpu,
        "eval_on_cpu": args.eval_on_cpu,
        "episodes": args.episodes,
        "task_ids": args.task_ids,
        "max_task_id": args.max_task_id,
        "out_dir": str(out_dir),
        "episode_csv": str(_episode_csv(Path(args.out_root), env_name, seed)),
        "log_path": str(log_path),
        "status": "RUNNING",
        "command": " ".join(shlex.quote(x) for x in command),
        "_log_fh": fh,
        "_start_time": time.time(),
    }
    return proc, meta


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage31 official GAS eval for newly READY artifact rows.")
    parser.add_argument("--inventory-csv", required=True)
    parser.add_argument("--out-root", default="runs_stage31_official_gas/incremental_ready_eval_active")
    parser.add_argument("--skip-run-roots", default="runs_stage31_official_gas/wide_atlas_active")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--env-regex", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task-ids", default="auto")
    parser.add_argument("--max-task-id", type=int, default=5)
    parser.add_argument("--eval-on-cpu", type=int, default=0)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--fallback-mode", default="none")
    parser.add_argument("--recover-dataset-indices", type=int, default=0)
    parser.add_argument("--node-map-batch-size", type=int, default=4096)
    parser.add_argument("--node-map-tolerance", type=float, default=1e-5)
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--skip-existing", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "stage31_incremental_ready_eval_status.csv"
    selected = _select_ready_rows(args)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] or ["0"]
    pending = list(selected)
    active: list[tuple[subprocess.Popen[Any], dict[str, Any]]] = []
    finished: list[dict[str, Any]] = []
    job_index = 0
    while pending or active:
        while pending and len(active) < max(1, args.max_jobs):
            row = pending.pop(0)
            gpu = gpus[job_index % len(gpus)]
            proc, meta = _launch(row, args, gpu, job_index)
            active.append((proc, meta))
            job_index += 1
        write_csv(status_path, [_public(meta) for _, meta in active] + [_public(row) for row in finished])
        time.sleep(5)
        still: list[tuple[subprocess.Popen[Any], dict[str, Any]]] = []
        for proc, meta in active:
            code = proc.poll()
            if code is None:
                still.append((proc, meta))
                continue
            fh = meta.get("_log_fh")
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
            meta["returncode"] = code
            meta["duration_sec"] = time.time() - float(meta.get("_start_time", time.time()))
            meta["status"] = "COMPLETE" if code == 0 and Path(str(meta["episode_csv"])).exists() else "FAILED"
            finished.append(meta)
        active = still
    write_csv(status_path, [_public(row) for row in finished])
    subprocess.check_call(
        [
            sys.executable,
            "scripts/stage31_collect_wide_atlas.py",
            "--run-root",
            args.out_root,
            "--inventory-csv",
            args.inventory_csv,
            "--out-root",
            str(out_root / "global"),
        ],
        cwd=args.repo_root,
    )
    print(status_path)


if __name__ == "__main__":
    main()
