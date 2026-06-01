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
GRAPH_ROOT = "artifacts/stage27_gas/graphs"


@dataclass(frozen=True)
class EnvConfig:
    keygraph: str
    checkpoint: str


ENV_CONFIGS: dict[str, EnvConfig] = {
    "antmaze-giant-navigate-v0": EnvConfig(
        keygraph="artifacts/tmd_test/antmaze-giant-navigate-v0/0_tmd50k_q98_t995_n512_paircal/keygraph_tmd.pkl",
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_giant_nav_tmd_actor_50k/sd000_20260527_192919/params_50000.pkl",
    ),
    "antmaze-giant-stitch-v0": EnvConfig(
        keygraph="artifacts/tmd_test/antmaze-giant-stitch-v0/0_tmd100keff_q98_t995_n512_paircal/keygraph_tmd.pkl",
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_giant_stitch_tmd_actor_resume50k_plus50k/sd000_20260527_192427/params_50000.pkl",
    ),
    "antmaze-medium-navigate-v0": EnvConfig(
        keygraph="artifacts/tmd_test/antmaze-medium-navigate-v0/0_tmd50k_q75_t90_n256/keygraph_tmd.pkl",
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_mnav_tmd_actor_50k/sd000_20260527_174643/params_50000.pkl",
    ),
    "antmaze-medium-stitch-v0": EnvConfig(
        keygraph="artifacts/tmd_test/antmaze-medium-stitch-v0/0_tmd50k_q75_t90_n256/keygraph_tmd.pkl",
        checkpoint="artifacts/tmd_test_trained/OGBench/tmd_test_tmd_actor_50k/sd000_20260527_150446/params_50000.pkl",
    ),
}


VARIANT_TO_GRAPH = {
    "B0_GAS": ("stage27_graph_policy", "B0_GAS"),
    "B1_ADAPTIVE": ("stage27_adaptive_policy", "B0_GAS"),
    "B2_LONGHOP": ("stage27_graph_policy", "B2_LONGHOP"),
    "B3_ADAPTIVE_LONGHOP": ("stage27_adaptive_policy", "B2_LONGHOP"),
    "B4_TMD_GATED": ("stage27_graph_policy", "B4_TMD_GATED"),
    "C1_EXEC_PENALTY": ("stage27_graph_policy", "C1_EXEC_PENALTY"),
    "C1_ADAPTIVE_EXEC": ("stage27_adaptive_policy", "C1_EXEC_PENALTY"),
    "C2_EXEC_GATE": ("stage27_graph_policy", "C2_EXEC_GATE"),
    "C2_ADAPTIVE_GATE": ("stage27_adaptive_policy", "C2_EXEC_GATE"),
    "C3_EXEC_UNCERT": ("stage27_graph_policy", "C3_EXEC_UNCERT"),
    "C3_ADAPTIVE_UNCERT": ("stage27_adaptive_policy", "C3_EXEC_UNCERT"),
    "C4_EXEC_TMD": ("stage27_graph_policy", "C4_EXEC_TMD"),
    "C4_ADAPTIVE_EXEC_TMD": ("stage27_adaptive_policy", "C4_EXEC_TMD"),
}


@dataclass
class Job:
    env_name: str
    env_seed: int
    gas_seed: int
    episodes: int
    variant: str
    output_dir: Path
    cmd: list[str]
    graph_path: str = ""


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _variant_run_name(variant: str, episodes: int, gas_seed: int) -> str:
    return f"{variant.lower()}_alltasks_ep{episodes}_gasseed{gas_seed}"


def _graph_path(graph_root: str, env_name: str, gas_seed: int, graph_variant: str) -> Path:
    return Path(graph_root) / env_name / f"gas_seed{gas_seed}" / f"graph_{graph_variant}.npz"


def _build_job(
    env_name: str,
    env_seed: int,
    gas_seed: int,
    episodes: int,
    variant: str,
    runs_root: Path,
    graph_root: str,
) -> Optional[Job]:
    cfg = ENV_CONFIGS[env_name]
    output_dir = runs_root / f"eval_ep{episodes}" / env_name / f"seed{env_seed}" / _variant_run_name(variant, episodes, gas_seed)
    cmd = [
        "bash",
        "scripts/tmd_test_eval.sh",
        "--env",
        env_name,
        "--seed",
        str(env_seed),
        "--episodes",
        str(episodes),
        "--tasks",
        "all",
        "--keygraph-path",
        cfg.keygraph,
        "--tmd-checkpoint",
        cfg.checkpoint,
        "--gas-seed",
        str(gas_seed),
        "--gas-artifact-root",
        ARTIFACT_ROOT,
        "--gas-gpu",
        "GPU_PLACEHOLDER",
        "--output-dir",
        str(output_dir),
    ]
    graph_path = ""
    if variant == "GAS_BASE":
        cmd[2:2] = ["--mode", "gas_graph_policy"]
    elif variant == "GNAV_TMD_POSCTRL":
        if env_name != "antmaze-giant-navigate-v0":
            return None
        cmd[2:2] = ["--mode", "gas_graph_tmd_cost_policy"]
        cmd.extend(["--tmd-cost-weight", "0.25", "--tmd-cost-lookup-observations", "50000"])
    else:
        mode, graph_variant = VARIANT_TO_GRAPH[variant]
        graph_path = str(_graph_path(graph_root, env_name, gas_seed, graph_variant))
        cmd[2:2] = ["--mode", mode]
        cmd.extend(["--stage27-graph-path", graph_path, "--stage27-variant", variant])
    return Job(env_name, env_seed, gas_seed, episodes, variant, output_dir, cmd, graph_path=graph_path)


def _job_matrix(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    runs_root = Path(args.runs_root)
    for env_name in _parse_csv(args.envs):
        if env_name not in ENV_CONFIGS:
            raise KeyError(f"No EnvConfig for {env_name}")
        for env_seed in _parse_csv_ints(args.env_seeds):
            for gas_seed in _parse_csv_ints(args.gas_seeds):
                for variant in _parse_csv(args.variants):
                    if variant not in {"GAS_BASE", "GNAV_TMD_POSCTRL", *VARIANT_TO_GRAPH.keys()}:
                        raise KeyError(f"Unknown variant {variant}")
                    job = _build_job(env_name, env_seed, gas_seed, int(args.episodes), variant, runs_root, args.graph_root)
                    if job is not None:
                        jobs.append(job)
    return jobs


def _manifest_fields() -> list[str]:
    return [
        "timestamp",
        "status",
        "env",
        "env_seed",
        "gas_seed",
        "episodes",
        "variant",
        "gpu",
        "returncode",
        "duration_sec",
        "output_dir",
        "graph_path",
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
    return [gpu if part == "GPU_PLACEHOLDER" else part for part in job.cmd]


def _skip_reason(job: Job, skip_existing: bool) -> Optional[str]:
    if job.graph_path and not Path(job.graph_path).exists():
        return "missing_graph"
    if skip_existing and (job.output_dir / "eval.csv").exists() and not (job.output_dir / "eval_error.json").exists():
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
    latest = Path(args.runs_root) / "stage27_eval_manifest.tsv"
    invocation = Path(args.runs_root) / f"stage27_eval_manifest_{run_id}.tsv"
    all_manifest = Path(args.runs_root) / "stage27_eval_manifest_all.tsv"
    manifest_rows: list[dict] = []
    pending = list(jobs)
    running: list[dict] = []
    max_workers = int(args.max_workers)
    if max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    next_gpu = 0
    while pending or running:
        while pending and len(running) < max_workers:
            job = pending.pop(0)
            gpu = gpus[next_gpu % len(gpus)]
            next_gpu += 1
            command = _command_for_gpu(job, gpu)
            skip = _skip_reason(job, bool(args.skip_existing))
            row_base = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "env": job.env_name,
                "env_seed": job.env_seed,
                "gas_seed": job.gas_seed,
                "episodes": job.episodes,
                "variant": job.variant,
                "gpu": gpu,
                "output_dir": str(job.output_dir),
                "graph_path": job.graph_path,
                "command": shlex.join(command),
            }
            if skip:
                _record(latest, invocation, all_manifest, manifest_rows, {**row_base, "status": skip, "returncode": 1 if skip == "missing_graph" else 0, "duration_sec": 0})
                continue
            job.output_dir.mkdir(parents=True, exist_ok=True)
            log = (job.output_dir / "run.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env.pop("JAX_PLATFORMS", None)
            env.pop("JAX_PLATFORM_NAME", None)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["MUJOCO_GL"] = "egl"
            env["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
            env["OGBENCH_DATASET_DIR"] = DATASET_ROOT
            env["BARS_TMD_TEST_DATASET_ROOT"] = DATASET_ROOT
            start = time.time()
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append({"job": job, "gpu": gpu, "command": command, "proc": proc, "log": log, "start": start, "row_base": row_base})
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
            print(f"{status}: {item['job'].env_name} {item['job'].variant} seed={item['job'].env_seed} gas={item['job'].gas_seed} rc={rc}")
        running = still
    return 0 if all(str(r.get("status")) in {"started", "completed", "already_completed"} for r in manifest_rows) else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage27 BARS/GAS evaluation matrix")
    p.add_argument("--runs-root", default="runs_stage27_gas")
    p.add_argument("--graph-root", default=GRAPH_ROOT)
    p.add_argument("--envs", default="antmaze-giant-navigate-v0,antmaze-giant-stitch-v0,antmaze-medium-navigate-v0,antmaze-medium-stitch-v0")
    p.add_argument("--env-seeds", default="0,1,2")
    p.add_argument("--gas-seeds", default="44,45,46")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--variants", default="GAS_BASE,B0_GAS,B1_ADAPTIVE,B2_LONGHOP,B3_ADAPTIVE_LONGHOP,C1_EXEC_PENALTY,C2_EXEC_GATE,C3_EXEC_UNCERT,C4_EXEC_TMD,GNAV_TMD_POSCTRL")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
