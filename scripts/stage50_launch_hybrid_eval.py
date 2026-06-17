#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage30_official_gas_common import (
    configure_official_env,
    ensure_ogbench_default_symlinks,
    final_goal_threshold,
    gas_agent_flag_args,
    official_gas_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/root/miniconda3/envs/gcrlo/bin/python")
DEFAULT_GAS_REPO = REPO_ROOT / "external_src" / "GAS"
DEFAULT_DATASET_DIR = Path("/mnt/project/offlinerl_datasets/ogbench")
DEFAULT_HYBRID_ROOT = (
    REPO_ROOT
    / "runs_stage49_sequence_actor_contract"
    / "20260616_071148"
    / "caplite_hybrid_stage45_sequence"
)


@dataclass(frozen=True)
class EvalJob:
    name: str
    env_name: str
    keygraph_rel: str
    policy_rel: str


DEFAULT_JOBS = [
    EvalJob(
        name="gn_hybrid_w0p25",
        env_name="antmaze-giant-navigate-v0",
        keygraph_rel="antmaze-giant-navigate-v0/patched_graph_contract_w0p25/keygraph.pkl",
        policy_rel="runs_stage38_official_gas_artifacts/antmaze-giant-navigate-v0/seed0/policy/params_1000000.pkl",
    ),
    EvalJob(
        name="large_explore_hybrid_w0p25",
        env_name="antmaze-large-explore-v0",
        keygraph_rel="antmaze-large-explore-v0/patched_graph_contract_w0p25/keygraph.pkl",
        policy_rel="runs_stage39_official_gas_artifacts/antmaze-large-explore-v0/seed0/policy/params_1000000.pkl",
    ),
    EvalJob(
        name="scene_hybrid_w2",
        env_name="scene-play-v0",
        keygraph_rel="scene-play-v0/patched_graph_contract_w2/keygraph.pkl",
        policy_rel="runs_stage40_official_gas_artifacts/scene-play-v0/seed0/policy/params_1000000.pkl",
    ),
]


STATUS_FIELDS = [
    "time",
    "job",
    "env_name",
    "seed",
    "gpu",
    "eval_on_cpu",
    "episodes",
    "status",
    "pid",
    "returncode",
    "duration_sec",
    "keygraph_path",
    "policy_path",
    "eval_csv",
    "log_path",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch predeclared Stage50 hybrid GAS evaluations on one GPU."
    )
    parser.add_argument("--gpu", default="3", help="Single visible GPU id. Default: 3.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-video-episodes", type=int, default=0)
    parser.add_argument(
        "--eval-on-cpu",
        type=int,
        default=-1,
        help="Override eval_on_cpu. -1 uses the official protocol registry.",
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gas-repo", type=Path, default=DEFAULT_GAS_REPO)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--hybrid-root", type=Path, default=DEFAULT_HYBRID_ROOT)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Defaults to runs_stage50_hybrid_eval_gpu3/<utc stamp>.",
    )
    parser.add_argument(
        "--jobs",
        default="all",
        help="Comma-separated job names or env names from the built-in Stage50 suite.",
    )
    parser.add_argument("--run-eval-project", default="stage50_hybrid_eval")
    parser.add_argument("--wait", type=int, default=0, help="Wait for all launched jobs.")
    parser.add_argument("--dry-run", type=int, default=0)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def selected_jobs(value: str) -> list[EvalJob]:
    if value.strip().lower() == "all":
        return list(DEFAULT_JOBS)
    wanted = {part.strip() for part in value.split(",") if part.strip()}
    selected = [
        job
        for job in DEFAULT_JOBS
        if job.name in wanted or job.env_name in wanted
    ]
    missing = sorted(wanted - {job.name for job in selected} - {job.env_name for job in selected})
    if missing:
        raise SystemExit(f"Unknown Stage50 eval jobs: {','.join(missing)}")
    return selected


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = configure_official_env(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["OGBENCH_DATASET_DIR"] = str(args.dataset_dir)
    env["BARS_OGBENCH_DATASET_DIR"] = str(args.dataset_dir)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MUJOCO_GL", "egl")
    pythonpath = [
        str(REPO_ROOT / "external_src" / "tmd-release"),
        str(args.gas_repo),
        str(REPO_ROOT),
    ]
    existing = [part for part in env.get("PYTHONPATH", "").split(":") if part]
    for part in existing:
        if part not in pythonpath:
            pythonpath.append(part)
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def append_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})


def append_command(path: Path, cwd: Path, env: dict[str, str], cmd: list[str]) -> None:
    visible_env = {
        "OGBENCH_DATASET_DIR": env.get("OGBENCH_DATASET_DIR", ""),
        "PYTHONPATH": env.get("PYTHONPATH", ""),
        "WANDB_MODE": env.get("WANDB_MODE", ""),
        "WANDB_DISABLED": env.get("WANDB_DISABLED", ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": env.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "MUJOCO_GL": env.get("MUJOCO_GL", ""),
        "MUJOCO_EGL_DEVICE_ID": env.get("MUJOCO_EGL_DEVICE_ID", ""),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] ")
        fh.write(f"cwd={shlex.quote(str(cwd))} ")
        fh.write("env ")
        for key, value in visible_env.items():
            if value:
                fh.write(f"{key}={shlex.quote(value)} ")
        fh.write("setsid ")
        fh.write(" ".join(shlex.quote(part) for part in cmd))
        fh.write("\n")


def prepare(args: argparse.Namespace, jobs: list[EvalJob]) -> None:
    if "," in str(args.gpu):
        raise SystemExit("Stage50 hybrid eval is constrained to one GPU; pass one --gpu id.")
    if not args.python_bin.exists():
        raise SystemExit(f"Missing python binary: {args.python_bin}")
    if not args.gas_repo.exists():
        raise SystemExit(f"Missing GAS repo: {args.gas_repo}")
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        for suffix in (".npz", "-val.npz"):
            dataset = args.dataset_dir / f"{job.env_name}{suffix}"
            if not dataset.exists():
                raise SystemExit(f"Missing local dataset: {dataset}")
        ensure_ogbench_default_symlinks(job.env_name, dataset_dir=args.dataset_dir)


def job_command(
    *,
    args: argparse.Namespace,
    job: EvalJob,
    keygraph_path: Path,
    policy_path: Path,
    save_eval_dir: Path,
    eval_csv: Path,
) -> tuple[list[str], int]:
    protocol = official_gas_protocol(job.env_name) or {}
    eval_on_cpu = int(protocol.get("eval_on_cpu", 1)) if args.eval_on_cpu < 0 else int(args.eval_on_cpu)
    cmd = [
        str(args.python_bin),
        "evaluate_gas.py",
        "--run_eval_project",
        args.run_eval_project,
        "--run_group",
        job.name,
        "--env_name",
        job.env_name,
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--save_eval_dir",
        str(save_eval_dir),
        "--eval_result_path",
        str(eval_csv),
        "--eval_on_cpu",
        str(eval_on_cpu),
        "--eval_episodes",
        str(args.eval_episodes),
        "--eval_video_episodes",
        str(args.eval_video_episodes),
        "--eval_final_goal_threshold",
        str(final_goal_threshold(job.env_name)),
        "--keygraph_path",
        str(keygraph_path),
        "--policy_path",
        str(policy_path),
        *gas_agent_flag_args(job.env_name),
    ]
    return cmd, eval_on_cpu


def main() -> None:
    args = parse_args()
    args.python_bin = resolve_repo_path(args.python_bin)
    args.gas_repo = resolve_repo_path(args.gas_repo)
    args.dataset_dir = resolve_repo_path(args.dataset_dir)
    args.hybrid_root = resolve_repo_path(args.hybrid_root)
    if args.out_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        args.out_root = REPO_ROOT / "runs_stage50_hybrid_eval_gpu3" / stamp
    else:
        args.out_root = resolve_repo_path(args.out_root)

    jobs = selected_jobs(args.jobs)
    prepare(args, jobs)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(exist_ok=True)
    (args.out_root / "raw_eval").mkdir(exist_ok=True)
    (args.out_root / "eval_csv").mkdir(exist_ok=True)
    latest = REPO_ROOT / "runs_stage50_hybrid_eval_gpu3" / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(args.out_root)

    env = build_env(args)
    status_path = args.out_root / "stage50_hybrid_eval_status.csv"
    command_log = args.out_root / "commands.log"
    rows: list[dict[str, Any]] = []
    active: list[tuple[subprocess.Popen[Any], dict[str, Any], Any]] = []

    for job in jobs:
        keygraph_path = (args.hybrid_root / job.keygraph_rel).resolve()
        policy_path = resolve_repo_path(Path(job.policy_rel))
        if not keygraph_path.exists():
            raise SystemExit(f"Missing keygraph for {job.name}: {keygraph_path}")
        if not policy_path.exists():
            raise SystemExit(f"Missing policy for {job.name}: {policy_path}")
        save_eval_dir = args.out_root / "raw_eval" / job.name
        eval_csv = args.out_root / "eval_csv" / f"{job.name}.csv"
        log_path = args.out_root / "logs" / f"{job.name}.log"
        cmd, eval_on_cpu = job_command(
            args=args,
            job=job,
            keygraph_path=keygraph_path,
            policy_path=policy_path,
            save_eval_dir=save_eval_dir,
            eval_csv=eval_csv,
        )
        append_command(command_log, args.gas_repo, env, cmd)
        row: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "job": job.name,
            "env_name": job.env_name,
            "seed": args.seed,
            "gpu": args.gpu,
            "eval_on_cpu": eval_on_cpu,
            "episodes": args.eval_episodes,
            "status": "DRY_RUN" if args.dry_run else "RUNNING",
            "keygraph_path": str(keygraph_path),
            "policy_path": str(policy_path),
            "eval_csv": str(eval_csv),
            "log_path": str(log_path),
            "command": " ".join(shlex.quote(part) for part in cmd),
        }
        rows.append(row)
        if args.dry_run:
            continue
        fh = log_path.open("wb")
        proc = subprocess.Popen(
            ["setsid", *cmd],
            cwd=args.gas_repo,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        row["pid"] = proc.pid
        row["_start_time"] = time.time()
        active.append((proc, row, fh))
        (args.out_root / "logs" / f"{job.name}.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        append_status(status_path, rows)

    append_status(status_path, rows)
    if not args.wait or args.dry_run:
        print(args.out_root)
        return

    while active:
        still: list[tuple[subprocess.Popen[Any], dict[str, Any], Any]] = []
        for proc, row, fh in active:
            code = proc.poll()
            if code is None:
                still.append((proc, row, fh))
                continue
            fh.close()
            row["returncode"] = code
            row["duration_sec"] = f"{time.time() - float(row.get('_start_time', time.time())):.3f}"
            row["status"] = "COMPLETE" if code == 0 and Path(str(row["eval_csv"])).exists() else "FAILED"
        active = still
        append_status(status_path, rows)
        if active:
            time.sleep(10)

    for row in rows:
        row.pop("_start_time", None)
    append_status(status_path, rows)
    print(status_path)


if __name__ == "__main__":
    main()
