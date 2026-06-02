#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from stage30_official_gas_common import configure_official_env, ensure_ogbench_default_symlinks, gas_agent_flag_args, write_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _truthy(value: Any) -> bool:
    return str(value) in {"1", "1.0", "true", "True"}


def _selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = _read_csv(Path(args.queue_csv))
    env_re = re.compile(args.env_regex) if args.env_regex else None
    selected: list[dict[str, str]] = []
    for row in rows:
        env_name = row.get("env_name", "")
        if env_re and not env_re.search(env_name):
            continue
        if row.get("artifact_status") != "MISSING_POLICY":
            continue
        if row.get("dataset_type") != "ogbench" or row.get("observation_type") != "state":
            continue
        if not _truthy(row.get("tdr_exists")) or not _truthy(row.get("keygraph_exists")):
            continue
        if not _truthy(row.get("train_dataset_exists")):
            continue
        target_policy = Path(row["artifact_seed_root"]) / "policy" / f"params_{args.train_steps}.pkl"
        if args.skip_existing and target_policy.exists():
            continue
        selected.append(row)
        if args.limit > 0 and len(selected) >= args.limit:
            break
    return selected


def _latest_params(raw_root: Path, start_time: float) -> Path | None:
    params = sorted(raw_root.rglob("params_*.pkl"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    params = [p for p in params if p.exists() and p.stat().st_mtime >= start_time - 1.0]
    return params[-1] if params else None


def _status_row(row: dict[str, str], status: str, **extra: Any) -> dict[str, Any]:
    out = {
        "stage": "stage31_missing_policy_training_backfill",
        "evidence_class": "OFFICIAL_GAS_POLICY_TRAINING_BACKFILL_STATUS",
        "env_name": row.get("env_name", ""),
        "seed": row.get("seed", ""),
        "artifact_seed_root": row.get("artifact_seed_root", ""),
        "tdr_path": row.get("tdr_path", ""),
        "keygraph_path": row.get("keygraph_path", ""),
        "status": status,
    }
    out.update(extra)
    return out


def _promote_policy(row: dict[str, str], raw_root: Path, start_time: float, train_steps: int) -> tuple[str, str]:
    latest = _latest_params(raw_root, start_time)
    if latest is None:
        return "", "NO_TRAINED_POLICY_FOUND"
    policy_dir = Path(row["artifact_seed_root"]) / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    target = policy_dir / f"params_{train_steps}.pkl"
    shutil.copy2(latest, target)
    return str(target), "PROMOTED_TO_ARTIFACT_POLICY_DIR"


def _launch(row: dict[str, str], args: argparse.Namespace, gpu: str, log_path: Path) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    env_name = row["env_name"]
    seed = int(float(row["seed"]))
    ensure_ogbench_default_symlinks(env_name, dataset_dir=Path(args.dataset_root) / "ogbench")
    raw_root = Path(args.out_root) / "raw_policy" / env_name / f"seed{seed}"
    raw_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train_policy.py",
        "--env_name",
        env_name,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--run_group",
        args.run_group,
        "--save_policy_dir",
        str(raw_root.resolve()),
        "--tdr_path",
        str(Path(row["tdr_path"]).resolve()),
        "--train_steps",
        str(args.train_steps),
        "--log_interval",
        str(args.log_interval),
        "--save_interval",
        str(args.save_interval),
        *gas_agent_flag_args(env_name),
    ]
    run_env = os.environ.copy()
    run_env.update(configure_official_env(gpu))
    run_env.setdefault("WANDB_MODE", "disabled")
    run_env.setdefault("WANDB_DISABLED", "true")
    run_env.setdefault("BARS_USE_TENSORBOARD", "0")
    fh = log_path.open("w", encoding="utf-8")
    start_time = time.time()
    proc = subprocess.Popen(command, cwd=str(Path(args.gas_repo_path)), stdout=fh, stderr=subprocess.STDOUT, env=run_env)
    meta = {
        "pid": proc.pid,
        "gpu": gpu,
        "log_path": str(log_path),
        "raw_policy_root": str(raw_root),
        "start_time": start_time,
        "command": " ".join(command),
        "_log_fh": fh,
    }
    return proc, meta


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    write_csv(path, public_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch official GAS policy training for Stage31 missing-policy artifacts.")
    parser.add_argument("--queue-csv", default="runs_stage31_official_gas/wide_artifact_inventory_active/official_gas_missing_ckpt_training_queue.csv")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--dataset-root", default="/mnt/project/offlinerl_datasets")
    parser.add_argument("--out-root", default="runs_stage31_official_gas/missing_policy_training_backfill_active")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1000000)
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--run-group", default="stage31_missing_policy_backfill")
    parser.add_argument("--env-regex", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", type=int, default=1)
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "stage31_missing_policy_training_status.csv"
    selected = _selected_rows(args)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        gpus = ["0"]
    status_rows: list[dict[str, Any]] = [
        _status_row(
            row,
            "DRY_RUN_SELECTED" if args.dry_run else "PENDING",
            train_steps=args.train_steps,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
        )
        for row in selected
    ]
    _write_status(status_path, status_rows)
    if args.dry_run:
        print(status_path)
        return

    active: list[tuple[subprocess.Popen[Any], dict[str, str], dict[str, Any]]] = []
    completed: list[dict[str, Any]] = []
    pending = list(selected)
    job_index = 0
    while pending or active:
        while pending and len(active) < max(1, args.max_jobs):
            row = pending.pop(0)
            gpu = gpus[job_index % len(gpus)]
            log_path = out_dir / "logs" / f"{row['env_name']}_seed{row['seed']}.train_policy.log"
            proc, meta = _launch(row, args, gpu, log_path)
            active.append((proc, row, meta))
            completed.append(_status_row(row, "RUNNING", **{k: v for k, v in meta.items() if not k.startswith("_")}))
            job_index += 1
            _write_status(status_path, completed + [_status_row(r, "PENDING") for r in pending])
        time.sleep(args.poll_interval)
        still_active: list[tuple[subprocess.Popen[Any], dict[str, str], dict[str, Any]]] = []
        for proc, row, meta in active:
            code = proc.poll()
            if code is None:
                still_active.append((proc, row, meta))
                continue
            fh = meta.get("_log_fh")
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
            policy_path = ""
            promote_status = ""
            if code == 0:
                policy_path, promote_status = _promote_policy(row, Path(meta["raw_policy_root"]), float(meta["start_time"]), args.train_steps)
            completed.append(
                _status_row(
                    row,
                    "COMPLETE" if code == 0 and policy_path else "FAILED",
                    returncode=code,
                    promoted_policy_path=policy_path,
                    promote_status=promote_status,
                    log_path=meta.get("log_path", ""),
                    gpu=meta.get("gpu", ""),
                )
            )
        active = still_active
        _write_status(status_path, completed + [_status_row(r, "PENDING") for r in pending])
    print(status_path)


if __name__ == "__main__":
    main()
