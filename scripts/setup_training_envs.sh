#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_MAIN="${CONDA_MAIN:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.ctyun.cn/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.ctyun.cn}"
PYTORCH_CU121_INDEX="${PYTORCH_CU121_INDEX:-https://download.pytorch.org/whl/cu121}"
JAX_CUDA_LINKS="${JAX_CUDA_LINKS:-https://storage.googleapis.com/jax-releases/jax_cuda_releases.html}"
MUJOCO210_URL="${MUJOCO210_URL:-https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz}"

DATA_ROOT="${DATA_ROOT:-/mnt/project/offlinerl_datasets}"
OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$DATA_ROOT/ogbench}"
D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-$DATA_ROOT/d4rl}"
MUJOCO_HOME_DIR="${MUJOCO_HOME_DIR:-$HOME/.mujoco}"
MUJOCO210_DIR="${MUJOCO210_DIR:-$MUJOCO_HOME_DIR/mujoco210}"
CUDA_LD_PATH="${CUDA_LD_PATH:-$MUJOCO210_DIR/bin:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/lib/x86_64-linux-gnu}"

export LD_LIBRARY_PATH="$CUDA_LD_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OGBENCH_DATASET_DIR D4RL_DATASET_DIR
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl

env_python() {
  local env_name="$1"
  shift
  "$CONDA_BASE/envs/$env_name/bin/python" "$@"
}

env_pip() {
  local env_name="$1"
  shift
  env_python "$env_name" -m pip "$@"
}

create_env() {
  local env_name="$1"
  local py_ver="$2"
  if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "Reusing existing conda env: $env_name"
  else
    conda create -y -n "$env_name" --override-channels -c "$CONDA_MAIN" "python=$py_ver" pip
  fi
}

install_mujoco_py_system_deps() {
  if [[ "${INSTALL_SYSTEM_DEPS:-0}" != "1" ]] || ! command -v apt-get >/dev/null 2>&1; then
    return
  fi

  local apt_cmd=(apt-get)
  if [[ "$(id -u)" -ne 0 ]]; then
    apt_cmd=(sudo apt-get)
  fi

  "${apt_cmd[@]}" update
  DEBIAN_FRONTEND=noninteractive "${apt_cmd[@]}" install -y \
    libglew-dev libosmesa6-dev libgl1-mesa-dev libegl1-mesa-dev patchelf
}

install_mujoco210() {
  if [[ -d "$MUJOCO210_DIR" ]]; then
    return
  fi

  mkdir -p "$MUJOCO_HOME_DIR"
  local archive
  archive="$(mktemp -t mujoco210.XXXXXX.tar.gz)"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$MUJOCO210_URL" -o "$archive"
  else
    wget -O "$archive" "$MUJOCO210_URL"
  fi
  tar -xzf "$archive" -C "$MUJOCO_HOME_DIR"
  rm -f "$archive"
}

ensure_nvidia_lib64_link() {
  if [[ -e /usr/local/nvidia/lib64 ]] || [[ ! -e /usr/local/cuda/lib64 ]]; then
    return
  fi

  if [[ "$(id -u)" -eq 0 ]]; then
    mkdir -p /usr/local/nvidia
    ln -s /usr/local/cuda/lib64 /usr/local/nvidia/lib64
  else
    echo "Tip: create /usr/local/nvidia/lib64 -> /usr/local/cuda/lib64 for mujoco-py EGL builds if needed." >&2
  fi
}

set_common_vars() {
  local env_name="$1"
  conda env config vars set -n "$env_name" \
    LD_LIBRARY_PATH="$CUDA_LD_PATH" \
    OGBENCH_DATASET_DIR="$OGBENCH_DATASET_DIR" \
    D4RL_DATASET_DIR="$D4RL_DATASET_DIR" \
    D4RL_SUPPRESS_IMPORT_ERROR=1 \
    MUJOCO_GL=egl >/dev/null
}

install_external_runtime_pth() {
  local env_name="$1"
  local site_dir
  site_dir="$(env_python "$env_name" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  printf '%s\n' "$ROOT_DIR/external_runtime" > "$site_dir/bars_external_runtime.pth"
}

install_gcrlo() {
  create_env gcrlo 3.9
  install_mujoco_py_system_deps
  install_mujoco210
  ensure_nvidia_lib64_link
  set_common_vars gcrlo
  conda env config vars set -n gcrlo \
    LD_LIBRARY_PATH="$CUDA_LD_PATH" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True >/dev/null

  env_pip gcrlo install \
    -i "$PIP_INDEX" --trusted-host "$PIP_TRUSTED_HOST" \
    "setuptools==65.5.0" "wheel==0.45.1" "Cython<3" "numpy==1.26.4"

  env_pip gcrlo install \
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "torchaudio==2.5.1+cu121" \
    --index-url "$PYTORCH_CU121_INDEX" --no-deps

  env_pip gcrlo install \
    -i "$PIP_INDEX" --trusted-host "$PIP_TRUSTED_HOST" \
    "nvidia-cublas-cu12==12.1.3.1" \
    "nvidia-cuda-cupti-cu12==12.1.105" \
    "nvidia-cuda-nvrtc-cu12==12.1.105" \
    "nvidia-cuda-runtime-cu12==12.1.105" \
    "nvidia-cudnn-cu12==9.1.0.70" \
    "nvidia-cufft-cu12==11.0.2.54" \
    "nvidia-curand-cu12==10.3.2.106" \
    "nvidia-cusolver-cu12==11.4.5.107" \
    "nvidia-cusparse-cu12==12.1.0.106" \
    "nvidia-nccl-cu12==2.21.5" \
    "nvidia-nvjitlink-cu12==12.9.86" \
    "nvidia-nvtx-cu12==12.1.105" \
    "triton==3.1.0"

  env_pip gcrlo install \
    -i "$PIP_INDEX" --trusted-host "$PIP_TRUSTED_HOST" \
    "numpy==1.26.4" pandas scipy scikit-learn tqdm h5py \
    "gym==0.23.1" "d4rl==1.1" "ogbench==1.1.5" \
    "mujoco==3.1.6" "dm-control==1.0.20" cloudpickle networkx tensorboard pynvml

  env_pip gcrlo install -e "$ROOT_DIR"
}

install_scaling_crl() {
  create_env scaling-crl 3.10
  set_common_vars scaling-crl
  conda env config vars set -n scaling-crl \
    LD_LIBRARY_PATH="$CUDA_LD_PATH" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.90 >/dev/null

  env_pip scaling-crl install \
    -i "$PIP_INDEX" --trusted-host "$PIP_TRUSTED_HOST" \
    "setuptools==70.3.0" "wheel==0.45.1" \
    "numpy==1.26.4" "scipy==1.12.0" "ml-dtypes==0.5.4" "opt-einsum==3.4.0"

  env_pip scaling-crl install \
    -i "$PIP_INDEX" --trusted-host "$PIP_TRUSTED_HOST" \
    --find-links "$JAX_CUDA_LINKS" \
    "jax==0.4.23" "jaxlib==0.4.23+cuda12.cudnn89" "numpy==1.26.4"

  env_pip scaling-crl install \
    -i "$PIP_INDEX" --trusted-host "$PIP_TRUSTED_HOST" \
    "chex==0.1.86" "flax==0.8.3" "optax==0.2.2" "distrax==0.1.5" "jaxopt==0.8.3" \
    "brax==0.12.1" "matplotlib==3.8.4" "mujoco==3.2.7" "mujoco-mjx==3.2.7" \
    "dm-control==1.0.20" "gymnasium==1.1.1" "tyro>=0.9.16" \
    wandb wandb-osh "ogbench==1.1.5" "numpy==1.26.4"

  env_pip scaling-crl install \
    -i "$PIP_INDEX" --trusted-host "$PIP_TRUSTED_HOST" \
    --no-deps "jaxgcrl==0.2.8"

  install_external_runtime_pth scaling-crl
}

verify_envs() {
  env_python gcrlo - <<'PY'
import torch
print("gcrlo", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
PY
  JAX_PLATFORMS=cpu env_python scaling-crl - <<'PY'
import jax, jaxlib, jaxgcrl, flax, optax, brax, ogbench
print("scaling-crl", jax.__version__, jaxlib.__version__, jax.devices())
PY
}

case "${1:-all}" in
  gcrlo) install_gcrlo ;;
  scaling-crl) install_scaling_crl ;;
  verify) verify_envs ;;
  all)
    install_gcrlo
    install_scaling_crl
    ;;
  *)
    echo "Usage: $0 [gcrlo|scaling-crl|all|verify]" >&2
    exit 2
    ;;
esac
