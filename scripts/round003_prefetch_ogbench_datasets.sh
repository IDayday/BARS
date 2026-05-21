#!/usr/bin/env bash
set -euo pipefail

ENVS="${ENVS:-antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0}"

for arg in "$@"; do
  case "$arg" in
    ENVS=*) ENVS="${arg#ENVS=}" ;;
    OGBENCH_DATASET_DIR=*) export OGBENCH_DATASET_DIR="${arg#OGBENCH_DATASET_DIR=}" ;;
    OGBENCH_DATASET_ENDPOINTS=*) export OGBENCH_DATASET_ENDPOINTS="${arg#OGBENCH_DATASET_ENDPOINTS=}" ;;
    BARS_OGBENCH_CN_ENDPOINTS=*) export BARS_OGBENCH_CN_ENDPOINTS="${arg#BARS_OGBENCH_CN_ENDPOINTS=}" ;;
    BARS_SHARED_DATASET_ROOT=*) export BARS_SHARED_DATASET_ROOT="${arg#BARS_SHARED_DATASET_ROOT=}" ;;
  esac
done

if [[ -f scripts/stage24_env_mirrors.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/stage24_env_mirrors.sh
fi

export PYTHONPATH="${PYTHONPATH:-$PWD}"
export BARS_DOWNLOAD_PARALLEL_CHUNKS="${BARS_DOWNLOAD_PARALLEL_CHUNKS:-1}"
export ENVS

python - <<'PY'
import os
from bars.data.ogbench_dataset import ensure_ogbench_dataset_files

envs = [x.strip() for x in os.environ.get("ENVS", "").split(",") if x.strip()]
dataset_dir = os.environ.get("OGBENCH_DATASET_DIR")
print(f"[round003_prefetch] dataset_dir={dataset_dir}", flush=True)
print(f"[round003_prefetch] endpoints={os.environ.get('OGBENCH_DATASET_ENDPOINTS')}", flush=True)
print(f"[round003_prefetch] shared_dirs={os.environ.get('OGBENCH_DATASET_SHARED_DIRS')}", flush=True)
for env in envs:
    train, val = ensure_ogbench_dataset_files(env, dataset_dir)
    print(f"[round003_prefetch] {env}: {train} | {val}", flush=True)
PY
