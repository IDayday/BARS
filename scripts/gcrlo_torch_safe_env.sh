#!/usr/bin/env bash

# Source this after activating the shared gcrlo environment when you want
# Torch-centric runs without JAX grabbing most GPU memory up front.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead of executing it:"
    echo "  source scripts/gcrlo_torch_safe_env.sh [GPU_ID]"
    exit 1
fi

gpu_id="${1:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ -n "${gpu_id}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

echo "Configured a Torch-friendly shared environment."
echo "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    echo "CUDA_VISIBLE_DEVICES is unchanged"
fi
