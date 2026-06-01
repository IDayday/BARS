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
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ARTIFACT_ROOT = "artifacts/gas_ogbench_offline_full_20260522_165138"
DATASET_ROOT = "/mnt/project/offlinerl_datasets/ogbench"
STAGE27_ROOT = "artifacts/stage27_gas"


@dataclass(frozen=True)
class EnvConfig:
    checkpoint: str


ENV_CONFIGS: dict[str, EnvConfig] = {
    "antmaze-giant-navigate-v0": EnvConfig(
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_giant_nav_tmd_actor_50k/sd000_20260527_192919/params_50000.pkl",
    ),
    "antmaze-giant-stitch-v0": EnvConfig(
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_giant_stitch_tmd_actor_resume50k_plus50k/sd000_20260527_192427/params_50000.pkl",
    ),
    "antmaze-medium-navigate-v0": EnvConfig(
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_mnav_tmd_actor_50k/sd000_20260527_174643/params_50000.pkl",
    ),
    "antmaze-medium-stitch-v0": EnvConfig(
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_tmd_actor_50k/sd000_20260527_150446/params_50000.pkl",
    ),
}


@dataclass
class Job:
    env_name: str
    gas_seed: int
    dataset_path: Path
    graph_dir: Path
    output_dir: Path
    cmd: list[str]


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _shell(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _graph_variants(text: str) -> list[str]:
    return _parse_csv(text)


def _graph_paths(graph_dir: Path, variants: list[str]) -> list[Path]:
    return [graph_dir / f"graph_{variant}.npz" for variant in variants]


def _build_job(env_name: str, gas_seed: int, args: argparse.Namespace) -> Job:
    cfg = ENV_CONFIGS[env_name]
    root = Path(args.stage27_root)
    dataset_path = root / "datasets" / env_name / f"gas_seed{gas_seed}" / "dataset.npz"
    graph_dir = root / "graphs" / env_name / f"gas_seed{gas_seed}"
    output_dir = root / "prepare_logs" / env_name / f"gas_seed{gas_seed}"

    export_cmd = [
        "python",
        "scripts/stage27_export_dataset.py",
        "--env",
        env_name,
        "--gas-seed",
        str(gas_seed),
        "--out",
        str(dataset_path),
        "--limit",
        str(args.dataset_limit),
        "--batch-size",
        str(args.batch_size),
        "--gas-artifact-root",
        ARTIFACT_ROOT,
        "--gas-gpu",
        "GPU_PLACEHOLDER",
        "--dataset-root",
        DATASET_ROOT,
    ]
    if not args.no_tmd_embeddings:
        export_cmd.extend(["--tmd-checkpoint", cfg.checkpoint])

    build_cmd = [
        "python",
        "scripts/stage27_build_graphs.py",
        "--dataset",
        str(dataset_path),
        "--out-dir",
        str(graph_dir),
        "--variants",
        *_graph_variants(args.variants),
        "--max-nodes",
        str(args.max_nodes),
        "--coverage-k",
        str(args.coverage_k),
        "--te-quantile",
        str(args.te_quantile),
        "--seed",
        str(gas_seed),
        "--calib-horizon",
        str(args.calib_horizon),
        "--random-negatives",
        str(args.random_negatives),
        "--hard-negatives",
        str(args.hard_negatives),
    ]
    shell_cmd = (
        "source /root/miniconda3/bin/activate gcrlo && "
        "export PYTHONPATH=.:external_src/GAS:external_src/tmd-release/impls:external_src/tmd-release:${PYTHONPATH:-} && "
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl D4RL_SUPPRESS_IMPORT_ERROR=1 "
        f"OGBENCH_DATASET_DIR={shlex.quote(DATASET_ROOT)} BARS_TMD_TEST_DATASET_ROOT={shlex.quote(DATASET_ROOT)} && "
        f"{_shell(export_cmd)} && {_shell(build_cmd)}"
    )
    return Job(env_name, gas_seed, dataset_path, graph_dir, output_dir, ["bash", "-lc", shell_cmd])


def _job_matrix(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for env_name in _parse_csv(args.envs):
        if env_name not in ENV_CONFIGS:
            raise KeyError(f"No EnvConfig for {env_name}")
        for gas_seed in _parse_csv_ints(args.gas_seeds):
            jobs.append(_build_job(env_name, gas_seed, args))
    return jobs


def _manifest_fields() -> list[str]:
    return [
        "timestamp",
        "status",
        "env",
        "gas_seed",
        "gpu",
        "returncode",
        "duration_sec",
        "dataset_path",
        "graph_dir",
        "output_dir",
        "command",
    ]


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_manifest_fields(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_manifest_fields(), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _record(latest: Path, invocation: Path, all_manifest: Path, rows: list[dict], row: dict) -> None:
    rows.append(row)
    _write_manifest(latest, rows)
    _write_manifest(invocation, rows)
    _append_manifest(all_manifest, row)


def _command_for_gpu(job: Job, gpu: str) -> list[str]:
    return [part.replace("GPU_PLACEHOLDER", gpu) for part in job.cmd]


def _skip_reason(job: Job, args: argparse.Namespace) -> Optional[str]:
    if not args.skip_existing:
        return None
    variants = _graph_variants(args.variants)
    if job.dataset_path.exists() and all(path.exists() for path in _graph_paths(job.graph_dir, variants)):
        return "already_completed"
    return None


def run(args: argparse.Namespace) -> int:
    jobs = _job_matrix(args)
    gpus = _parse_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must not be empty")
    if args.dry_run:
        for idx, job in enumerate(jobs):
            print(shlex.join(_command_for_gpu(job, gpus[idx % len(gpus)])))
        return 0

    run_id = time.strftime("%Y%m%d_%H%M%S")
    latest = Path(args.stage27_root) / "stage27_prepare_manifest.tsv"
    invocation = Path(args.stage27_root) / f"stage27_prepare_manifest_{run_id}.tsv"
    all_manifest = Path(args.stage27_root) / "stage27_prepare_manifest_all.tsv"
    manifest_rows: list[dict] = []
    pending = list(jobs)
    running: list[dict] = []
    max_workers = min(int(args.max_workers), len(gpus))
    next_gpu = 0
    while pending or running:
        while pending and len(running) < max_workers:
            job = pending.pop(0)
            gpu = gpus[next_gpu % len(gpus)]
            next_gpu += 1
            command = _command_for_gpu(job, gpu)
            row_base = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "env": job.env_name,
                "gas_seed": job.gas_seed,
                "gpu": gpu,
                "dataset_path": str(job.dataset_path),
                "graph_dir": str(job.graph_dir),
                "output_dir": str(job.output_dir),
                "command": shlex.join(command),
            }
            skip = _skip_reason(job, args)
            if skip:
                _record(latest, invocation, all_manifest, manifest_rows, {**row_base, "status": skip, "returncode": 0, "duration_sec": 0})
                continue
            job.output_dir.mkdir(parents=True, exist_ok=True)
            log = (job.output_dir / "prepare.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env.pop("JAX_PLATFORMS", None)
            env.pop("JAX_PLATFORM_NAME", None)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["MUJOCO_GL"] = "egl"
            start = time.time()
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append({"job": job, "proc": proc, "log": log, "start": start, "row_base": row_base})
            _record(latest, invocation, all_manifest, manifest_rows, {**row_base, "status": "started", "returncode": "", "duration_sec": ""})
        time.sleep(float(args.poll_sec))
        still = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still.append(item)
                continue
            item["log"].close()
            status = "completed" if rc == 0 else "failed"
            duration = time.time() - item["start"]
            row = {**item["row_base"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": status, "returncode": rc, "duration_sec": f"{duration:.1f}"}
            _record(latest, invocation, all_manifest, manifest_rows, row)
            print(f"{status}: {item['job'].env_name} gas={item['job'].gas_seed} rc={rc}")
        running = still
    return 0 if all(str(r.get("status")) in {"started", "completed", "already_completed"} for r in manifest_rows) else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Stage27 datasets and build graph variants")
    p.add_argument("--stage27-root", default=STAGE27_ROOT)
    p.add_argument("--envs", default="antmaze-giant-navigate-v0,antmaze-giant-stitch-v0,antmaze-medium-navigate-v0,antmaze-medium-stitch-v0")
    p.add_argument("--gas-seeds", default="44,45,46")
    p.add_argument("--variants", default="B0_GAS,B2_LONGHOP,B4_TMD_GATED,C1_EXEC_PENALTY,C2_EXEC_GATE,C3_EXEC_UNCERT,C4_EXEC_TMD")
    p.add_argument("--dataset-limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--max-nodes", type=int, default=2500)
    p.add_argument("--coverage-k", type=int, default=1000)
    p.add_argument("--te-quantile", type=float, default=0.85)
    p.add_argument("--calib-horizon", type=int, default=20)
    p.add_argument("--random-negatives", type=int, default=20000)
    p.add_argument("--hard-negatives", type=int, default=20000)
    p.add_argument("--no-tmd-embeddings", action="store_true")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--poll-sec", type=float, default=10.0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
