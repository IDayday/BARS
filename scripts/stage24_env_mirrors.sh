#!/usr/bin/env bash
# Optional Stage24 network defaults for mainland-China friendly artifact/dependency setup.
# Source this file before running prepare/install commands; explicit user env vars win.

export BARS_HF_ENDPOINTS="${BARS_HF_ENDPOINTS:-https://hf-mirror.com,https://huggingface.co}"
export BARS_SHARED_DATASET_ROOT="${BARS_SHARED_DATASET_ROOT:-/root/remote/datasets}"
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$BARS_SHARED_DATASET_ROOT/ogbench}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-$BARS_SHARED_DATASET_ROOT/d4rl}"
export OGBENCH_DATASET_SHARED_DIRS="${OGBENCH_DATASET_SHARED_DIRS:-$HOME/.ogbench/data,$HOME/.cache/bars_datasets/ogbench,$BARS_SHARED_DATASET_ROOT/ogbench}"

# Put site-local / mainland-China OGBench mirrors here when available, for example:
#   export BARS_OGBENCH_CN_ENDPOINTS=https://your-mirror.example/ogbench
# If no mirror is configured, the downloader falls back to the official OGBench
# endpoints. HTTP(S)_PROXY is honored by curl and urllib for that fallback.
export BARS_OGBENCH_CN_ENDPOINTS="${BARS_OGBENCH_CN_ENDPOINTS:-}"
if [[ -z "${OGBENCH_DATASET_ENDPOINTS:-}" ]]; then
  if [[ -n "$BARS_OGBENCH_CN_ENDPOINTS" ]]; then
    export OGBENCH_DATASET_ENDPOINTS="${BARS_OGBENCH_CN_ENDPOINTS},https://rail.eecs.berkeley.edu/datasets/ogbench,http://rail.eecs.berkeley.edu/datasets/ogbench"
  else
    export OGBENCH_DATASET_ENDPOINTS="https://rail.eecs.berkeley.edu/datasets/ogbench,http://rail.eecs.berkeley.edu/datasets/ogbench"
  fi
fi

# Serial download is slower but more stable through common HTTP proxies than
# multi-range downloads. Users can raise this explicitly on a fast LAN mirror.
export BARS_DOWNLOAD_WITH_ARIA2="${BARS_DOWNLOAD_WITH_ARIA2:-1}"
export BARS_ARIA2_SPLIT="${BARS_ARIA2_SPLIT:-16}"
export BARS_ARIA2_CONNECTIONS="${BARS_ARIA2_CONNECTIONS:-16}"
export BARS_DOWNLOAD_PARALLEL_CHUNKS="${BARS_DOWNLOAD_PARALLEL_CHUNKS:-1}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

# curl/urllib honor these if the shell already provides a proxy.
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}"
