#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source /root/miniconda3/bin/activate gcrlo
export PYTHONPATH=".:external_src/tmd-release/impls:external_src/tmd-release:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export BARS_TMD_TEST_DATASET_ROOT="${BARS_TMD_TEST_DATASET_ROOT:-/mnt/project/offlinerl_datasets/ogbench}"
python -m bars.tmd_test.construct_graph "$@"
