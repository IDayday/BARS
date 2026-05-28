from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_ROOT = "artifacts/gas_ogbench_offline_full_20260522_165138"
DATASET_ROOT = "/mnt/project/offlinerl_datasets/ogbench"


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


@dataclass
class Job:
    env_name: str
    env_seed: int
    gas_seed: int
    episodes: int
    variant: str
    weight: float | None
    output_dir: Path
    cmd: list[str]

    @property
    def run_name(self) -> str:
        return self.output_dir.name


def _weight_code(weight: float) -> str:
    text = f"{weight:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _build_job(env_name: str, env_seed: int, gas_seed: int, episodes: int, variant: str, weight: float | None, root: Path) -> Job:
    cfg = ENV_CONFIGS[env_name]
    if variant == "gas":
        mode = "gas_graph_policy"
        run_name = f"gas_graph_policy_alltasks_ep{episodes}_gasseed{gas_seed}"
    else:
        assert weight is not None
        mode = "gas_graph_tmd_cost_policy"
        run_name = f"gas_graph_tmdcost_w{_weight_code(weight)}_alltasks_ep{episodes}_gasseed{gas_seed}"
    output_dir = root / "phase_b" / env_name / f"seed{env_seed}" / run_name
    cmd = [
        "bash",
        "scripts/tmd_test_eval.sh",
        "--mode",
        mode,
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
    if variant == "tmd_cost":
        cmd.extend(
            [
                "--tmd-cost-weight",
                str(weight),
                "--tmd-cost-lookup-observations",
                "50000",
            ]
        )
    return Job(env_name, env_seed, gas_seed, episodes, variant, weight, output_dir, cmd)


def _job_matrix(args: argparse.Namespace) -> list[Job]:
    root = Path(args.runs_root)
    envs = [x.strip() for x in args.envs.split(",") if x.strip()]
    env_seeds = _parse_csv_ints(args.env_seeds)
    gas_seeds = _parse_csv_ints(args.gas_seeds)
    weights = _parse_csv_floats(args.weights)
    jobs: list[Job] = []
    for env_name in envs:
        if env_name not in ENV_CONFIGS:
            raise KeyError(f"No EnvConfig for {env_name}")
        for env_seed in env_seeds:
            for gas_seed in gas_seeds:
                if args.include_gas:
                    jobs.append(_build_job(env_name, env_seed, gas_seed, args.episodes, "gas", None, root))
                for weight in weights:
                    jobs.append(_build_job(env_name, env_seed, gas_seed, args.episodes, "tmd_cost", weight, root))
    return jobs


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "status",
        "env",
        "env_seed",
        "gas_seed",
        "episodes",
        "variant",
        "weight",
        "gpu",
        "returncode",
        "duration_sec",
        "output_dir",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _append_manifest_row(path: Path, row: dict) -> None:
    fields = [
        "timestamp",
        "status",
        "env",
        "env_seed",
        "gas_seed",
        "episodes",
        "variant",
        "weight",
        "gpu",
        "returncode",
        "duration_sec",
        "output_dir",
        "command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _record_manifest_row(latest: Path, invocation: Path, all_manifest: Path, rows: list[dict], row: dict) -> None:
    rows.append(row)
    _write_manifest(latest, rows)
    _write_manifest(invocation, rows)
    _append_manifest_row(all_manifest, row)


def _command_for_gpu(job: Job, gpu: str) -> list[str]:
    return [gpu if part == "GPU_PLACEHOLDER" else part for part in job.cmd]


def _skip_reason(job: Job) -> str | None:
    eval_csv = job.output_dir / "eval.csv"
    err = job.output_dir / "eval_error.json"
    if eval_csv.exists() and not err.exists():
        return "already_completed"
    return None


def run(args: argparse.Namespace) -> int:
    jobs = _job_matrix(args)
    manifest_rows: list[dict] = []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    manifest = Path(args.runs_root) / "stage26_phase_b_manifest.tsv"
    invocation_manifest = Path(args.runs_root) / f"stage26_phase_b_manifest_{run_id}.tsv"
    all_manifest = Path(args.runs_root) / "stage26_phase_b_manifest_all.tsv"
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must not be empty")
    max_workers = min(int(args.max_workers), len(gpus))
    pending = list(jobs)
    running: list[dict] = []
    next_gpu_index = 0
    if args.dry_run:
        for idx, job in enumerate(jobs):
            gpu = gpus[idx % len(gpus)]
            command = _command_for_gpu(job, gpu)
            print(shlex.join(command))
        return 0
    while pending or running:
        while pending and len(running) < max_workers:
            job = pending.pop(0)
            skip = _skip_reason(job) if args.skip_existing else None
            gpu = gpus[next_gpu_index % len(gpus)]
            next_gpu_index += 1
            command = _command_for_gpu(job, gpu)
            if skip:
                _record_manifest_row(
                    manifest,
                    invocation_manifest,
                    all_manifest,
                    manifest_rows,
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "status": skip,
                        "env": job.env_name,
                        "env_seed": job.env_seed,
                        "gas_seed": job.gas_seed,
                        "episodes": job.episodes,
                        "variant": job.variant,
                        "weight": "" if job.weight is None else job.weight,
                        "gpu": gpu,
                        "returncode": 0,
                        "duration_sec": 0,
                        "output_dir": str(job.output_dir),
                        "command": shlex.join(command),
                    },
                )
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
            running.append({"job": job, "gpu": gpu, "command": command, "proc": proc, "log": log, "start": start})
            _record_manifest_row(
                manifest,
                invocation_manifest,
                all_manifest,
                manifest_rows,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": "started",
                    "env": job.env_name,
                    "env_seed": job.env_seed,
                    "gas_seed": job.gas_seed,
                    "episodes": job.episodes,
                    "variant": job.variant,
                    "weight": "" if job.weight is None else job.weight,
                    "gpu": gpu,
                    "returncode": "",
                    "duration_sec": "",
                    "output_dir": str(job.output_dir),
                    "command": shlex.join(command),
                },
            )
        time.sleep(float(args.poll_sec))
        still_running: list[dict] = []
        for item in running:
            proc = item["proc"]
            rc = proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log"].close()
            duration = time.time() - item["start"]
            job = item["job"]
            status = "completed" if rc == 0 else "failed"
            _record_manifest_row(
                manifest,
                invocation_manifest,
                all_manifest,
                manifest_rows,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": status,
                    "env": job.env_name,
                    "env_seed": job.env_seed,
                    "gas_seed": job.gas_seed,
                    "episodes": job.episodes,
                    "variant": job.variant,
                    "weight": "" if job.weight is None else job.weight,
                    "gpu": item["gpu"],
                    "returncode": rc,
                    "duration_sec": f"{duration:.1f}",
                    "output_dir": str(job.output_dir),
                    "command": shlex.join(item["command"]),
                },
            )
        running = still_running
    failed = [r for r in manifest_rows if r.get("status") == "failed"]
    print(f"jobs={len(jobs)} manifest_rows={len(manifest_rows)} failed={len(failed)} manifest={manifest}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage-26 Phase-B TMD-cost matrix.")
    parser.add_argument("--runs-root", default="runs_stage26_tmd_tdr")
    parser.add_argument(
        "--envs",
        default="antmaze-giant-navigate-v0",
        help="Comma-separated env names.",
    )
    parser.add_argument("--env-seeds", default="0,1,2")
    parser.add_argument("--gas-seeds", default="42,43")
    parser.add_argument("--weights", default="0.10,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--include-gas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
