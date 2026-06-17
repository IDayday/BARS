#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage30_official_gas_common import configure_official_env, ensure_ogbench_default_symlinks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "stage32_official_gas_protocol_registry.json"
DEFAULT_GAS_REPO = REPO_ROOT / "external_src" / "GAS"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "gas_ogbench_offline_full_20260522_165138"
DEFAULT_DATASET_DIR = Path("/mnt/project/offlinerl_datasets/ogbench")
DEFAULT_D4RL_DATASET_DIR = Path("/mnt/project/offlinerl_datasets/d4rl")
DEFAULT_OGBENCH_PACKAGE_ROOT = REPO_ROOT / "external_src" / "tmd-release"
DEFAULT_PYTHON = Path("/root/miniconda3/envs/gcrlo/bin/python")


@dataclass(frozen=True)
class Job:
    env_name: str
    seed: int
    protocol: dict[str, Any]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("environments", raw if isinstance(raw, list) else [])
    return {str(row["env_name"]): dict(row) for row in rows}


def selected_envs(registry: dict[str, dict[str, Any]], requested: list[str] | None) -> list[str]:
    if requested:
        envs = requested
    else:
        envs = [env for env, row in registry.items() if int(row.get("selected_paper_env", 1))]
    missing = [env for env in envs if env not in registry]
    if missing:
        raise SystemExit(f"Missing envs in registry: {','.join(missing)}")
    return envs


def inventory_missing_jobs(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    out: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("can_run_eval_only") == "1" and row.get("artifact_status") == "COMPLETE":
                continue
            env_name = row.get("env_name", "")
            seed_raw = row.get("seed", "")
            if not env_name or not seed_raw:
                continue
            out.add((env_name, int(float(seed_raw))))
    return out


def agent_flags(protocol: dict[str, Any]) -> list[str]:
    return [
        "--agent_config.encoder",
        str(protocol["encoder"]),
        "--agent_config.discount",
        str(protocol["discount"]),
        "--agent_config.tdr_expectile",
        str(protocol["tdr_expectile"]),
        "--agent_config.alpha",
        str(protocol["alpha"]),
        "--agent_config.batch_size",
        str(protocol["batch_size"]),
        "--agent_config.p_aug",
        str(protocol["p_aug"]),
        "--agent_config.way_steps",
        str(protocol["way_steps"]),
    ]


def subprocess_env(args: argparse.Namespace, gpu: str) -> dict[str, str]:
    env = configure_official_env(gpu)
    env["PATH"] = f"{args.python_bin.parent}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["OGBENCH_DATASET_DIR"] = str(args.dataset_dir)
    env["D4RL_DATASET_DIR"] = str(args.d4rl_dataset_dir)
    python_path_parts = [str(REPO_ROOT), str(args.gas_repo)]
    if args.ogbench_package_root.exists():
        python_path_parts.append(str(args.ogbench_package_root))
    if env.get("PYTHONPATH"):
        python_path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_path_parts)
    env.setdefault("BARS_USE_TENSORBOARD", "0")
    return env


def is_d4rl_env(env_name: str, protocol: dict[str, Any] | None = None) -> bool:
    family = str((protocol or {}).get("env_family", ""))
    return family == "kitchen" or env_name == "kitchen-partial-v0"


def prepare_dataset(env_name: str, args: argparse.Namespace, gpu: str, protocol: dict[str, Any] | None = None) -> None:
    if is_d4rl_env(env_name, protocol):
        matches = sorted(args.d4rl_dataset_dir.glob("kitchen*.hdf5"))
        if not matches:
            raise RuntimeError(f"Missing local D4RL dataset for {env_name} under {args.d4rl_dataset_dir}")
        return
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    default_dir = Path.home() / ".ogbench" / "data"
    default_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", "-val.npz"):
        project_path = args.dataset_dir / f"{env_name}{suffix}"
        default_path = default_dir / f"{env_name}{suffix}"
        if not project_path.exists() and default_path.exists() and not default_path.is_symlink():
            shutil.move(str(default_path), str(project_path))
    missing = [
        suffix
        for suffix in (".npz", "-val.npz")
        if not (args.dataset_dir / f"{env_name}{suffix}").exists()
    ]
    if missing and args.download_datasets:
        cmd = [
            str(args.python_bin),
            "-c",
            "import sys; from ogbench.utils import download_datasets; "
            "download_datasets([sys.argv[1]], dataset_dir=sys.argv[2])",
            env_name,
            str(args.dataset_dir),
        ]
        subprocess.run(cmd, check=True, cwd=args.gas_repo, env=subprocess_env(args, gpu))
    missing = [
        suffix
        for suffix in (".npz", "-val.npz")
        if not (args.dataset_dir / f"{env_name}{suffix}").exists()
    ]
    if missing:
        raise RuntimeError(f"Missing local datasets for {env_name}: {','.join(missing)}")
    ensure_ogbench_default_symlinks(env_name, dataset_dir=args.dataset_dir)


def latest_file(root: Path, pattern: str) -> Path | None:
    matches = [path for path in root.rglob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def expected_param(root: Path, epoch: int | str | None) -> Path | None:
    if epoch:
        path = root / f"params_{epoch}.pkl"
        if path.exists():
            return path
    return latest_file(root, "params_*.pkl")


def copy_if_needed(src: Path, dst: Path, overwrite: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return dst
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    return dst


def copy_latest_flags(source_root: Path, target_seed_root: Path, kind: str, tag: str) -> None:
    flags = latest_file(source_root, "flags.json")
    if flags is None:
        return
    copy_if_needed(flags, target_seed_root / "_raw_gas" / kind / tag / "flags.json", overwrite=True)


def try_promote_from_sources(
    *,
    args: argparse.Namespace,
    job: Job,
    kind: str,
    pattern: str,
    target: Path,
) -> Path | None:
    if target.exists():
        return target
    for source_root in args.extra_source_roots:
        source_seed_root = source_root / job.env_name / f"seed{job.seed}"
        if not source_seed_root.exists():
            continue
        source = latest_file(source_seed_root / kind, pattern)
        if source is None:
            continue
        copy_if_needed(source, target)
        copy_latest_flags(source_seed_root / kind, target.parent.parent, kind, "promoted_source")
        if kind == "graph":
            keynodes = latest_file(source_seed_root / kind, "keynodes.pkl")
            if keynodes is not None:
                copy_if_needed(keynodes, target.parent / "keynodes.pkl")
        return target
    return target if target.exists() else None


def append_status(status_path: Path, lock: threading.Lock, row: dict[str, Any]) -> None:
    fields = [
        "time",
        "event",
        "env_name",
        "seed",
        "gpu",
        "phase",
        "status",
        "returncode",
        "duration_sec",
        "log_path",
        "tdr_path",
        "keygraph_path",
        "policy_path",
        "eval_csv",
        "command",
    ]
    with lock:
        exists = status_path.exists()
        with status_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in fields})


def run_phase(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    job: Job,
    gpu: str,
    phase: str,
    status_path: Path,
    status_lock: threading.Lock,
    paths: dict[str, str],
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = " ".join(cmd)
    append_status(
        status_path,
        status_lock,
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "phase_started",
            "env_name": job.env_name,
            "seed": job.seed,
            "gpu": gpu,
            "phase": phase,
            "status": "RUNNING",
            "log_path": str(log_path),
            "command": command,
            **paths,
        },
    )
    start = time.time()
    if dry_run:
        code = 0
    else:
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
        code = proc.returncode
    duration = time.time() - start
    append_status(
        status_path,
        status_lock,
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "phase_completed",
            "env_name": job.env_name,
            "seed": job.seed,
            "gpu": gpu,
            "phase": phase,
            "status": "COMPLETE" if code == 0 else "FAILED",
            "returncode": code,
            "duration_sec": f"{duration:.3f}",
            "log_path": str(log_path),
            "command": command,
            **paths,
        },
    )
    if code != 0:
        raise RuntimeError(f"{job.env_name} seed{job.seed} {phase} failed with return code {code}")


def skip_phase(
    *,
    status_path: Path,
    status_lock: threading.Lock,
    job: Job,
    gpu: str,
    phase: str,
    status: str,
    paths: dict[str, str],
) -> None:
    append_status(
        status_path,
        status_lock,
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "phase_skipped",
            "env_name": job.env_name,
            "seed": job.seed,
            "gpu": gpu,
            "phase": phase,
            "status": status,
            **paths,
        },
    )


def repair_metadata(
    *,
    args: argparse.Namespace,
    job: Job,
    gpu: str,
    log_path: Path,
    status_path: Path,
    status_lock: threading.Lock,
    paths: dict[str, str],
) -> None:
    out_csv = args.run_root / job.env_name / f"seed{job.seed}" / "metadata_repair.csv"
    cmd = [
        str(args.python_bin),
        "scripts/stage32_repair_gas_metadata.py",
        "--artifact-root",
        str(args.artifact_root),
        "--registry",
        str(args.registry),
        "--gas-repo-path",
        str(args.gas_repo),
        "--envs",
        job.env_name,
        "--seeds",
        str(job.seed),
        "--gpu",
        str(gpu),
        "--generate-dataset-embeddings",
        str(args.generate_dataset_embeddings),
        "--skip-existing-embeddings",
        "1",
        "--write-manifest",
        "1",
        "--out-csv",
        str(out_csv),
    ]
    run_phase(
        cmd=cmd,
        cwd=REPO_ROOT,
        env=subprocess_env(args, gpu),
        log_path=log_path,
        job=job,
        gpu=gpu,
        phase="metadata_repair",
        status_path=status_path,
        status_lock=status_lock,
        paths=paths,
        dry_run=args.dry_run,
    )


def run_job(job: Job, gpu: str, args: argparse.Namespace, status_path: Path, status_lock: threading.Lock) -> None:
    env_slug = job.env_name.removesuffix("-v0")
    run_group = f"stage35_full_gas_{env_slug}_seed{job.seed}"
    seed_root = args.artifact_root / job.env_name / f"seed{job.seed}"
    raw_root = args.run_root / job.env_name / f"seed{job.seed}" / "raw"
    log_dir = args.run_root / job.env_name / f"seed{job.seed}" / "logs"
    for path in (
        seed_root / "tdr",
        seed_root / "graph",
        seed_root / "policy",
        seed_root / "_raw_gas",
        raw_root,
        log_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {"tdr_path": "", "keygraph_path": "", "policy_path": "", "eval_csv": ""}
    flags = agent_flags(job.protocol)
    train_steps = str(job.protocol["train_steps"])
    tdr_epoch = str(job.protocol["tdr_checkpoint_epoch"])
    policy_epoch = str(job.protocol["policy_checkpoint_epoch"])
    tdr_target = seed_root / "tdr" / f"params_{tdr_epoch}.pkl"
    graph_target = seed_root / "graph" / "keygraph.pkl"
    policy_target = seed_root / "policy" / f"params_{policy_epoch}.pkl"
    env = subprocess_env(args, gpu)
    common = ["--env_name", job.env_name, "--seed", str(job.seed), "--gpu", str(gpu)]

    prepare_dataset(job.env_name, args, gpu, job.protocol)

    tdr_path = try_promote_from_sources(args=args, job=job, kind="tdr", pattern=f"params_{tdr_epoch}.pkl", target=tdr_target)
    if tdr_path is None:
        run_phase(
            cmd=[
                str(args.python_bin),
                "pretrain_tdr.py",
                "--run_tdr_project",
                "stage35_full_gas_tdr",
                "--run_group",
                run_group,
                *common,
                "--save_tdr_dir",
                str(raw_root / "tdr"),
                "--train_steps",
                train_steps,
                "--log_interval",
                str(args.log_interval),
                "--save_interval",
                str(args.save_interval),
                *flags,
            ],
            cwd=args.gas_repo,
            env=env,
            log_path=log_dir / "pretrain_tdr.log",
            job=job,
            gpu=gpu,
            phase="pretrain_tdr",
            status_path=status_path,
            status_lock=status_lock,
            paths=paths,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            tdr_path = tdr_target
        else:
            trained_tdr = latest_file(raw_root / "tdr", f"params_{tdr_epoch}.pkl")
            if trained_tdr is None:
                raise RuntimeError(f"Missing trained TDR checkpoint for {job.env_name} seed{job.seed}")
            tdr_path = copy_if_needed(trained_tdr, tdr_target, overwrite=args.overwrite_existing)
            copy_latest_flags(raw_root / "tdr", seed_root, "tdr", "trained")
    else:
        skip_phase(status_path=status_path, status_lock=status_lock, job=job, gpu=gpu, phase="pretrain_tdr", status="EXISTS", paths=paths)
    paths["tdr_path"] = str(tdr_path)

    keygraph_path = try_promote_from_sources(args=args, job=job, kind="graph", pattern="keygraph.pkl", target=graph_target)
    if keygraph_path is None:
        run_phase(
            cmd=[
                str(args.python_bin),
                "construct_graph.py",
                "--run_group",
                run_group,
                *common,
                "--save_graph_dir",
                str(raw_root / "graph"),
                "--te_threshold",
                str(job.protocol["te_threshold"]),
                "--tdr_path",
                str(tdr_path),
                *flags,
            ],
            cwd=args.gas_repo,
            env=env,
            log_path=log_dir / "construct_graph.log",
            job=job,
            gpu=gpu,
            phase="construct_graph",
            status_path=status_path,
            status_lock=status_lock,
            paths=paths,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            keygraph_path = graph_target
        else:
            trained_keygraph = latest_file(raw_root / "graph", "keygraph.pkl")
            if trained_keygraph is None:
                raise RuntimeError(f"Missing trained keygraph for {job.env_name} seed{job.seed}")
            keygraph_path = copy_if_needed(trained_keygraph, graph_target, overwrite=args.overwrite_existing)
            trained_keynodes = latest_file(raw_root / "graph", "keynodes.pkl")
            if trained_keynodes is not None:
                copy_if_needed(trained_keynodes, seed_root / "graph" / "keynodes.pkl", overwrite=args.overwrite_existing)
            copy_latest_flags(raw_root / "graph", seed_root, "graph", "trained")
    else:
        skip_phase(status_path=status_path, status_lock=status_lock, job=job, gpu=gpu, phase="construct_graph", status="EXISTS", paths=paths)
    paths["keygraph_path"] = str(keygraph_path)

    policy_path = try_promote_from_sources(args=args, job=job, kind="policy", pattern=f"params_{policy_epoch}.pkl", target=policy_target)
    if policy_path is None:
        run_phase(
            cmd=[
                str(args.python_bin),
                "train_policy.py",
                "--run_policy_project",
                "stage35_full_gas_policy",
                "--run_group",
                run_group,
                *common,
                "--save_policy_dir",
                str(raw_root / "policy"),
                "--train_steps",
                train_steps,
                "--log_interval",
                str(args.log_interval),
                "--save_interval",
                str(args.save_interval),
                "--tdr_path",
                str(tdr_path),
                *flags,
            ],
            cwd=args.gas_repo,
            env=env,
            log_path=log_dir / "train_policy.log",
            job=job,
            gpu=gpu,
            phase="train_policy",
            status_path=status_path,
            status_lock=status_lock,
            paths=paths,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            policy_path = policy_target
        else:
            trained_policy = latest_file(raw_root / "policy", f"params_{policy_epoch}.pkl")
            if trained_policy is None:
                raise RuntimeError(f"Missing trained policy checkpoint for {job.env_name} seed{job.seed}")
            policy_path = copy_if_needed(trained_policy, policy_target, overwrite=args.overwrite_existing)
            copy_latest_flags(raw_root / "policy", seed_root, "policy", "trained")
    else:
        skip_phase(status_path=status_path, status_lock=status_lock, job=job, gpu=gpu, phase="train_policy", status="EXISTS", paths=paths)
    paths["policy_path"] = str(policy_path)

    eval_csv = seed_root / "policy" / "eval.csv"
    eval_video_episodes = 0 if is_d4rl_env(job.env_name, job.protocol) else args.eval_video_episodes
    if not eval_csv.exists() or args.force_eval:
        run_phase(
            cmd=[
                str(args.python_bin),
                "evaluate_gas.py",
                "--run_eval_project",
                "stage35_full_gas_eval",
                "--run_group",
                run_group,
                *common,
                "--save_eval_dir",
                str(raw_root / "eval"),
                "--eval_on_cpu",
                str(int(job.protocol.get("eval_on_cpu", 1))),
                "--eval_episodes",
                str(args.eval_episodes),
                "--eval_video_episodes",
                str(eval_video_episodes),
                "--eval_final_goal_threshold",
                str(job.protocol["eval_final_goal_threshold"]),
                "--keygraph_path",
                str(keygraph_path),
                "--policy_path",
                str(policy_path),
                *flags,
            ],
            cwd=args.gas_repo,
            env=env,
            log_path=log_dir / "evaluate_gas.log",
            job=job,
            gpu=gpu,
            phase="evaluate_gas",
            status_path=status_path,
            status_lock=status_lock,
            paths=paths,
            dry_run=args.dry_run,
        )
    else:
        skip_phase(status_path=status_path, status_lock=status_lock, job=job, gpu=gpu, phase="evaluate_gas", status="EXISTS", paths=paths)
    if eval_csv.exists() or args.dry_run:
        paths["eval_csv"] = str(eval_csv)

    if args.repair_metadata:
        repair_metadata(
            args=args,
            job=job,
            gpu=gpu,
            log_path=log_dir / "metadata_repair.log",
            status_path=status_path,
            status_lock=status_lock,
            paths=paths,
        )

    append_status(
        status_path,
        status_lock,
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "job_completed",
            "env_name": job.env_name,
            "seed": job.seed,
            "gpu": gpu,
            "phase": "all",
            "status": "DRY_RUN" if args.dry_run else "COMPLETE",
            **paths,
        },
    )


def worker(gpu: str, jobs: queue.Queue[Job], args: argparse.Namespace, status_path: Path, status_lock: threading.Lock) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(job, gpu, args, status_path, status_lock)
        except Exception as exc:
            append_status(
                status_path,
                status_lock,
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": "job_failed",
                    "env_name": job.env_name,
                    "seed": job.seed,
                    "gpu": gpu,
                    "phase": "all",
                    "status": "FAILED",
                    "command": str(exc),
                },
            )
        finally:
            jobs.task_done()


def wait_for_pid(pid: int, status_path: Path, status_lock: threading.Lock, poll_interval: int) -> None:
    if pid <= 0:
        return
    while Path(f"/proc/{pid}").exists():
        append_status(
            status_path,
            status_lock,
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "waiting_for_pid",
                "status": "RUNNING",
                "command": str(pid),
            },
        )
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch missing official-config GAS training for all selected tasks.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--gas-repo", type=Path, default=DEFAULT_GAS_REPO)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "runs_stage35_full_gas_graph_limitations" / "training_active")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--d4rl-dataset-dir", type=Path, default=DEFAULT_D4RL_DATASET_DIR)
    parser.add_argument("--ogbench-package-root", type=Path, default=DEFAULT_OGBENCH_PACKAGE_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--inventory-csv", default="")
    parser.add_argument("--envs", default="")
    parser.add_argument("--seeds", default="44,45,46")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--missing-only", type=int, default=1)
    parser.add_argument("--extra-source-roots", default="")
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--wait-poll-interval", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=49)
    parser.add_argument("--eval-video-episodes", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--download-datasets", type=int, default=0)
    parser.add_argument("--repair-metadata", type=int, default=1)
    parser.add_argument("--generate-dataset-embeddings", type=int, default=1)
    parser.add_argument("--force-eval", type=int, default=0)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.registry = args.registry.resolve()
    args.gas_repo = args.gas_repo.resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.run_root = args.run_root.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    args.d4rl_dataset_dir = args.d4rl_dataset_dir.resolve()
    args.ogbench_package_root = args.ogbench_package_root.resolve()
    args.python_bin = args.python_bin.resolve()
    args.extra_source_roots = [Path(x).resolve() for x in parse_csv(args.extra_source_roots)]

    registry = load_registry(args.registry)
    envs = selected_envs(registry, parse_csv(args.envs) if args.envs else None)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    gpus = parse_csv(args.gpus)
    if not gpus:
        raise SystemExit("At least one GPU must be supplied.")
    inventory_csv = Path(args.inventory_csv).resolve() if args.inventory_csv else None
    missing = inventory_missing_jobs(inventory_csv) if inventory_csv else set()

    jobs: queue.Queue[Job] = queue.Queue()
    for env_name in envs:
        for seed in seeds:
            if args.missing_only and missing and (env_name, seed) not in missing:
                continue
            jobs.put(Job(env_name=env_name, seed=seed, protocol=registry[env_name]))

    args.run_root.mkdir(parents=True, exist_ok=True)
    status_path = args.run_root / "stage35_full_gas_training_status.csv"
    status_lock = threading.Lock()
    append_status(
        status_path,
        status_lock,
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "orchestrator_started",
            "status": "DRY_RUN" if args.dry_run else "RUNNING",
            "command": " ".join(sys.argv),
        },
    )
    wait_for_pid(args.wait_pid, status_path, status_lock, args.wait_poll_interval)
    max_workers = max(1, args.max_jobs)
    worker_gpus = [gpus[0]] * max_workers if len(gpus) == 1 else gpus[:max_workers]
    threads = [
        threading.Thread(target=worker, args=(gpu, jobs, args, status_path, status_lock), daemon=False)
        for gpu in worker_gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    append_status(
        status_path,
        status_lock,
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "orchestrator_completed",
            "status": "DRY_RUN" if args.dry_run else "COMPLETE",
        },
    )
    print(status_path)


if __name__ == "__main__":
    main()
