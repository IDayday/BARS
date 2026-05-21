#!/usr/bin/env bash
# Optional Stage24 network defaults for mainland-China friendly artifact/dependency setup.
# Source this file before running prepare/install commands; explicit user env vars win.

export BARS_HF_ENDPOINTS="${BARS_HF_ENDPOINTS:-https://hf-mirror.com,https://huggingface.co}"
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
export OGBENCH_DATASET_ENDPOINTS="${OGBENCH_DATASET_ENDPOINTS:-https://rail.eecs.berkeley.edu/datasets/ogbench,http://rail.eecs.berkeley.edu/datasets/ogbench}"
export BARS_DOWNLOAD_PARALLEL_CHUNKS="${BARS_DOWNLOAD_PARALLEL_CHUNKS:-8}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

# curl/urllib honor these if the shell already provides a proxy.
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}"
