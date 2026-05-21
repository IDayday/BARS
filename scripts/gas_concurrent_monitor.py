#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bars.external.gas_artifacts import official_eval_on_cpu, resolve_gas_artifacts


STATE_OGBENCH_ENVS = [
    "antmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "antmaze-medium-stitch-v0",
    "antmaze-large-stitch-v0",
    "antmaze-giant-stitch-v0",
    "antmaze-medium-explore-v0",
    "antmaze-large-explore-v0",
    "scene-play-v0",
]

VISUAL_OGBENCH_ENVS = [
    "visual-antmaze-medium-navigate-v0",
    "visual-antmaze-large-navigate-v0",
    "visual-antmaze-giant-navigate-v0",
    "visual-antmaze-medium-stitch-v0",
    "visual-antmaze-large-stitch-v0",
    "visual-antmaze-giant-stitch-v0",
    "visual-antmaze-medium-explore-v0",
    "visual-antmaze-large-explore-v0",
    "visual-scene-play-v0",
]

HUMANOID_OGBENCH_ENVS = [
    "humanoidmaze-medium-navigate-v0",
    "humanoidmaze-large-navigate-v0",
    "humanoidmaze-giant-navigate-v0",
    "humanoidmaze-medium-stitch-v0",
    "humanoidmaze-large-stitch-v0",
    "humanoidmaze-giant-stitch-v0",
]

ALWAYS_ENVS = ["kitchen-partial-v0"]
SUPPORTED_OGBENCH_ENVS = STATE_OGBENCH_ENVS + VISUAL_OGBENCH_ENVS + HUMANOID_OGBENCH_ENVS


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def ogbench_ready(data_dir: Path, env_name: str) -> bool:
    train = data_dir / f"{env_name}.npz"
    val = data_dir / f"{env_name}-val.npz"
    required = [train, val]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    suffixes = (".aria2", ".tmp", ".part", ".crdownload")
    if any((Path(str(path) + suffix)).exists() for path in required for suffix in suffixes):
        return False
    return all(zipfile.is_zipfile(path) for path in required)


def ready_envs(data_dir: Path, include_visual: bool, include_humanoid: bool) -> list[str]:
    candidates = STATE_OGBENCH_ENVS[:]
    if include_visual:
        candidates += VISUAL_OGBENCH_ENVS
    if include_humanoid:
        candidates += HUMANOID_OGBENCH_ENVS
    envs = ALWAYS_ENVS[:]
    envs += [env for env in candidates if ogbench_ready(data_dir, env)]
    return envs


def status_for(log_root: Path, env_name: str, seed: int) -> dict:
    return read_json(log_root / env_name / f"seed{seed}" / "status.json")


def job_known(log_root: Path, artifact_root: Path, env_name: str, seed: int, retry_failed: bool) -> bool:
    artifacts = resolve_gas_artifacts(env_name, seed, artifact_root)
    if artifacts.complete:
        return True
    status = status_for(log_root, env_name, seed).get("status")
    if status in {"running", "completed"}:
        return True
    if status == "failed" and not retry_failed:
        return True
    return False


def python_site_packages(python_bin: str) -> Path:
    code = "import site; print(site.getsitepackages()[0])"
    out = subprocess.check_output([python_bin, "-c", code], text=True).strip()
    return Path(out)


def cuda_lib_path(site_packages: Path) -> str:
    nvidia = site_packages / "nvidia"
    if not nvidia.exists():
        return ""
    paths = sorted(str(path) for path in nvidia.glob("*/lib") if path.is_dir())
    return ":".join(paths)


def shell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def make_job_script(
    path: Path,
    *,
    root: Path,
    python_bin: str,
    gas_repo_path: str,
    artifact_root: Path,
    log_root: Path,
    env_name: str,
    seed: int,
    gpu: str,
    eval_after_train: bool,
    eval_episodes: int,
    eval_video_episodes: int,
    eval_on_cpu: int,
) -> None:
    log_dir = log_root / env_name / f"seed{seed}"
    body = f"""#!/usr/bin/env bash
set +e
cd {shell_quote(root)}
export WANDB_MODE="${{WANDB_MODE:-disabled}}"
export WANDB_DISABLED="${{WANDB_DISABLED:-true}}"
export BARS_USE_TENSORBOARD="${{BARS_USE_TENSORBOARD:-0}}"
export TENSORBOARD_LOGDIR="{log_root}/tensorboard/{env_name}/seed{seed}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
export MUJOCO_GL="${{MUJOCO_GL:-egl}}"
export D4RL_SUPPRESS_IMPORT_ERROR="${{D4RL_SUPPRESS_IMPORT_ERROR:-1}}"
echo '{{"status":"running","stage":"train","env":"{env_name}","seed":{seed},"gpu":"{gpu}","started_at":"'$(date -Is)'"}}' > {shell_quote(log_dir / "status.json")}
echo "[gas_monitor] train env={env_name} seed={seed} gpu={gpu} started $(date -Is)"
{shell_quote(python_bin)} -m bars.external.gas_prepare \\
  --env {shell_quote(env_name)} \\
  --seed {seed} \\
  --artifact-root {shell_quote(artifact_root)} \\
  --gas-repo-path {shell_quote(gas_repo_path)} \\
  --gpu {shell_quote(gpu)} \\
  --quick 0 \\
  --prefer-pretrained 0 \\
  --train-if-missing 1 \\
  --export-embeddings 1
train_rc=$?
if [[ "$train_rc" -eq 0 && "{int(eval_after_train)}" == "1" ]]; then
  echo '{{"status":"running","stage":"eval","env":"{env_name}","seed":{seed},"gpu":"{gpu}","started_eval_at":"'$(date -Is)'"}}' > {shell_quote(log_dir / "status.json")}
  echo "[gas_monitor] eval env={env_name} seed={seed} gpu={gpu} started $(date -Is)"
  {shell_quote(python_bin)} scripts/gas_official_eval.py \\
    --env {shell_quote(env_name)} \\
    --seed {seed} \\
    --gpu {shell_quote(gpu)} \\
    --artifact-root {shell_quote(artifact_root)} \\
    --gas-repo-path {shell_quote(gas_repo_path)} \\
    --eval-episodes {eval_episodes} \\
    --eval-video-episodes {eval_video_episodes} \\
    --eval-on-cpu {eval_on_cpu}
  eval_rc=$?
else
  eval_rc=0
fi
if [[ "$train_rc" -eq 0 && "$eval_rc" -eq 0 ]]; then
  status=completed
  rc=0
else
  status=failed
  rc=$(( train_rc != 0 ? train_rc : eval_rc ))
fi
echo '{{"status":"'$status'","env":"{env_name}","seed":{seed},"gpu":"{gpu}","train_rc":"'$train_rc'","eval_rc":"'$eval_rc'","finished_at":"'$(date -Is)'"}}' > {shell_quote(log_dir / "status.json")}
echo "[gas_monitor] env={env_name} seed={seed} gpu={gpu} finished train_rc=$train_rc eval_rc=$eval_rc $(date -Is)"
exit "$rc"
"""
    path.write_text(body)
    path.chmod(0o755)


def launch_job(args: argparse.Namespace, env_name: str, seed: int, gpu: str) -> int:
    log_dir = args.log_root / env_name / f"seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_script = log_dir / "job.sh"
    log_path = log_dir / "run.log"
    make_job_script(
        job_script,
        root=ROOT,
        python_bin=args.python_bin,
        gas_repo_path=args.gas_repo_path,
        artifact_root=args.artifact_root,
        log_root=args.log_root,
        env_name=env_name,
        seed=seed,
        gpu=gpu,
        eval_after_train=args.eval_after_train,
        eval_episodes=args.eval_episodes,
        eval_video_episodes=args.eval_video_episodes,
        eval_on_cpu=official_eval_on_cpu(env_name),
    )
    if args.dry_run:
        print(f"[gas_monitor] dry-run launch env={env_name} seed={seed} gpu={gpu}")
        return -1
    log = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            ["bash", str(job_script)],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log.close()
    with open(args.pids_file, "a", buffering=1) as f:
        f.write(f"{env_name}\t{seed}\t{gpu}\t{proc.pid}\t{log_path}\tlaunched\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print(f"[gas_monitor] launched env={env_name} seed={seed} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
    return proc.pid


def launch_ready(args: argparse.Namespace, state: dict) -> int:
    envs = ready_envs(args.ogbench_dir, args.include_visual, args.include_humanoid)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    gpus = parse_csv(args.gpus)
    cursor = int(state.get("gpu_cursor", 0))
    launched = 0
    for env_name in envs:
        for seed in seeds:
            if job_known(args.log_root, args.artifact_root, env_name, seed, args.retry_failed):
                continue
            gpu = gpus[cursor % len(gpus)]
            cursor += 1
            launch_job(args, env_name, seed, gpu)
            launched += 1
    state["gpu_cursor"] = cursor
    state["last_ready_envs"] = envs
    state["last_scan_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(args.state_file, state)
    return launched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--python-bin", default="/root/miniconda3/envs/gcrlo/bin/python")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/gas_official_full_20260521"))
    p.add_argument("--log-root", type=Path, default=Path("runs_gas_official_full_20260521"))
    p.add_argument("--ogbench-dir", type=Path, default=Path("/mnt/project/offlinerl_datasets/ogbench"))
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--scan-interval", type=int, default=120)
    p.add_argument("--eval-after-train", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=49)
    p.add_argument("--eval-video-episodes", type=int, default=1)
    p.add_argument("--include-visual", type=int, default=1)
    p.add_argument("--include-humanoid", type=int, default=1)
    p.add_argument("--retry-failed", type=int, default=0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.artifact_root = (ROOT / args.artifact_root).resolve() if not args.artifact_root.is_absolute() else args.artifact_root
    args.log_root = (ROOT / args.log_root).resolve() if not args.log_root.is_absolute() else args.log_root
    args.state_file = args.log_root / "monitor_state.json"
    args.pids_file = args.log_root / "monitor_pids.tsv"
    args.eval_after_train = bool(args.eval_after_train)
    args.include_visual = bool(args.include_visual)
    args.include_humanoid = bool(args.include_humanoid)
    args.retry_failed = bool(args.retry_failed)
    if not parse_csv(args.gpus):
        raise ValueError("At least one GPU must be provided")
    return args


def configure_process_env(args: argparse.Namespace) -> None:
    site_packages = python_site_packages(args.python_bin)
    cuda_libs = cuda_lib_path(site_packages)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("BARS_USE_TENSORBOARD", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    py_paths = [str(ROOT), str(ROOT / args.gas_repo_path)]
    if os.environ.get("PYTHONPATH"):
        py_paths.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = ":".join(py_paths)
    ld_paths = []
    if cuda_libs:
        ld_paths.append(cuda_libs)
    ld_paths.append("/root/.mujoco/mujoco210/bin")
    if os.environ.get("LD_LIBRARY_PATH"):
        ld_paths.append(os.environ["LD_LIBRARY_PATH"])
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    nvcc_bin = site_packages / "nvidia" / "cuda_nvcc" / "bin"
    os.environ["PATH"] = f"{nvcc_bin}:{os.environ.get('PATH', '')}"


def main() -> None:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    configure_process_env(args)
    if not args.pids_file.exists():
        args.pids_file.write_text("env\tseed\tgpu\tpid\tlog\tstatus\tlaunched_at\n")
    state = read_json(args.state_file)
    if "gpu_cursor" not in state:
        existing = sum(1 for _ in args.log_root.glob("*/seed*/status.json"))
        state["gpu_cursor"] = existing
    while True:
        launched = launch_ready(args, state)
        print(
            f"[gas_monitor] scan complete launched={launched} ready_envs={len(state.get('last_ready_envs', []))} "
            f"next_scan={args.scan_interval}s",
            flush=True,
        )
        if args.once:
            return
        time.sleep(args.scan_interval)


if __name__ == "__main__":
    main()
