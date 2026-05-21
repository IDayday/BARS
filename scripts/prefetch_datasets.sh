#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"

DATA_ROOT="${DATA_ROOT:-/mnt/project/offlinerl_datasets}"
OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$DATA_ROOT/ogbench}"
D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-$DATA_ROOT/d4rl}"

D4RL_ENVS="${D4RL_ENVS:-antmaze-umaze-v2,antmaze-umaze-diverse-v2,antmaze-medium-play-v2,antmaze-medium-diverse-v2,antmaze-large-play-v2,antmaze-large-diverse-v2}"
OGBENCH_ENVS="${OGBENCH_ENVS:-antmaze-medium-stitch-v0,antmaze-medium-navigate-v0,antmaze-large-stitch-v0,antmaze-large-navigate-v0,antmaze-giant-stitch-v0,antmaze-giant-navigate-v0}"

D4RL_HF_BASE="${D4RL_HF_BASE:-https://hf-mirror.com/datasets/imone/D4RL/resolve/main}"
OGBENCH_BASE_URL="${OGBENCH_BASE_URL:-https://rail.eecs.berkeley.edu/datasets/ogbench}"

MUJOCO210_DIR="${MUJOCO210_DIR:-$HOME/.mujoco/mujoco210}"
CUDA_LD_PATH="${CUDA_LD_PATH:-$MUJOCO210_DIR/bin:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/lib/x86_64-linux-gnu}"

export LD_LIBRARY_PATH="$CUDA_LD_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OGBENCH_DATASET_DIR D4RL_DATASET_DIR
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl

NO_PROXY_ENV=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy)

env_python() {
  local env_name="$1"
  shift
  "$CONDA_BASE/envs/$env_name/bin/python" "$@"
}

is_domestic_url() {
  case "$1" in
    *hf-mirror.com*|*modelscope.cn*|*opendatalab.com*|*aliyun*|*tuna.tsinghua.edu.cn*|*ustc.edu.cn*|*bfsu.edu.cn*|*ctyun.cn*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_for_url() {
  local url="$1"
  shift
  if is_domestic_url "$url"; then
    "${NO_PROXY_ENV[@]}" "$@"
  else
    "$@"
  fi
}

aria2_download() {
  local url="$1"
  local dir="$2"
  local file="$3"
  mkdir -p "$dir"
  run_for_url "$url" aria2c \
    --console-log-level=warn \
    --summary-interval="${ARIA2_SUMMARY_INTERVAL:-30}" \
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --file-allocation=none \
    --max-tries=10 \
    --retry-wait=10 \
    --connect-timeout=20 \
    --timeout=90 \
    -x "${ARIA2_CONNECTIONS:-16}" \
    -s "${ARIA2_SPLITS:-16}" \
    -k 1M \
    -d "$dir" \
    -o "$file" \
    "$url"
}

d4rl_file_for_env() {
  case "$1" in
    antmaze-umaze-v2) echo "Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5" ;;
    antmaze-umaze-diverse-v2) echo "Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5" ;;
    antmaze-medium-play-v2) echo "Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5" ;;
    antmaze-medium-diverse-v2) echo "Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5" ;;
    antmaze-large-play-v2) echo "Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5" ;;
    antmaze-large-diverse-v2) echo "Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5" ;;
    kitchen-complete-v0) echo "mini_kitchen_microwave_kettle_light_slider-v0.hdf5" ;;
    kitchen-partial-v0) echo "kitchen_microwave_kettle_light_slider-v0.hdf5" ;;
    kitchen-mixed-v0) echo "kitchen_microwave_kettle_bottomburner_light-v0.hdf5" ;;
    *) return 1 ;;
  esac
}

d4rl_url_for_env() {
  case "$1" in
    kitchen-partial-v0)
      echo "http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5"
      ;;
    *)
      echo "$D4RL_HF_BASE/$2"
      ;;
  esac
}

valid_hdf5() {
  env_python gcrlo - "$1" <<'PY'
import sys, h5py
try:
    with h5py.File(sys.argv[1], "r") as f:
        ok = all(k in f for k in ("observations", "actions", "rewards", "terminals"))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

valid_npz() {
  env_python gcrlo - "$1" <<'PY'
import sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as zf:
        ok = zf.testzip() is None
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

prefetch_d4rl() {
  mkdir -p "$D4RL_DATASET_DIR"
  IFS=',' read -r -a envs <<< "$D4RL_ENVS"
  for env_name in "${envs[@]}"; do
    [[ -n "$env_name" ]] || continue
    local file
    if ! file="$(d4rl_file_for_env "$env_name")"; then
      echo "No fast mirror mapping for $env_name; falling back to d4rl's official downloader." >&2
      env_python gcrlo - "$env_name" <<'PY'
import sys
from bars.data.d4rl_dataset import prefetch_d4rl_dataset
print(prefetch_d4rl_dataset(sys.argv[1]), flush=True)
PY
      continue
    fi

    local path="$D4RL_DATASET_DIR/$file"
    if [[ -s "$path" ]] && valid_hdf5 "$path"; then
      echo "D4RL ready: $env_name -> $path"
      continue
    fi

    rm -f "$path" "$path.aria2"
    local url
    url="$(d4rl_url_for_env "$env_name" "$file")"
    echo "Downloading D4RL: $env_name"
    aria2_download "$url" "$D4RL_DATASET_DIR" "$file"
    valid_hdf5 "$path"
    echo "D4RL ready: $env_name -> $path"
  done
}

prefetch_ogbench() {
  mkdir -p "$OGBENCH_DATASET_DIR"
  local input_file
  input_file="$(mktemp -t ogbench_urls.XXXXXX)"
  trap 'rm -f "$input_file"' RETURN

  IFS=',' read -r -a envs <<< "$OGBENCH_ENVS"
  for env_name in "${envs[@]}"; do
    [[ -n "$env_name" ]] || continue
    for suffix in "" "-val"; do
      local file="${env_name}${suffix}.npz"
      local path="$OGBENCH_DATASET_DIR/$file"
      if [[ -s "$path" ]] && valid_npz "$path"; then
        echo "OGBench ready: $path"
        continue
      fi
      rm -f "$path" "$path.aria2"
      printf '%s/%s\n  out=%s\n' "${OGBENCH_BASE_URL%/}" "$file" "$file" >> "$input_file"
    done
  done

  if [[ -s "$input_file" ]]; then
    echo "Downloading OGBench from ${OGBENCH_BASE_URL%/}"
    local cmd=(
      aria2c
      --console-log-level=notice
      --summary-interval="${ARIA2_SUMMARY_INTERVAL:-30}"
      --continue=true
      --allow-overwrite=false
      --auto-file-renaming=false
      --file-allocation=none
      --max-tries=10
      --retry-wait=10
      --connect-timeout=20
      --timeout=90
      -x "${ARIA2_CONNECTIONS:-16}"
      -s "${ARIA2_SPLITS:-16}"
      -k 1M
      --max-concurrent-downloads="${ARIA2_CONCURRENT:-6}"
      -d "$OGBENCH_DATASET_DIR"
      -i "$input_file"
    )
    run_for_url "$OGBENCH_BASE_URL" "${cmd[@]}"
  fi

  for env_name in "${envs[@]}"; do
    [[ -n "$env_name" ]] || continue
    valid_npz "$OGBENCH_DATASET_DIR/${env_name}.npz"
    valid_npz "$OGBENCH_DATASET_DIR/${env_name}-val.npz"
    echo "OGBench ready: $env_name"
  done
}

case "${1:-all}" in
  d4rl) prefetch_d4rl ;;
  ogbench) prefetch_ogbench ;;
  all)
    prefetch_d4rl
    prefetch_ogbench
    ;;
  *)
    echo "Usage: $0 [d4rl|ogbench|all]" >&2
    exit 2
    ;;
esac
