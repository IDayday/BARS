#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source /root/miniconda3/bin/activate gcrlo
export PYTHONPATH=".:external_src/GAS:external_src/tmd-release/impls:external_src/tmd-release:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export BARS_TMD_TEST_DATASET_ROOT="${BARS_TMD_TEST_DATASET_ROOT:-/mnt/project/offlinerl_datasets/ogbench}"
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$BARS_TMD_TEST_DATASET_ROOT}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/root/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${MUJOCO_PY_MUJOCO_PATH}/bin"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
python -m bars.tmd_test.evaluate_tmd_graph "$@"
