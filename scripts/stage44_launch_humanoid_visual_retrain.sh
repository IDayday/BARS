#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/project/BARS}"
cd "${ROOT}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gcrlo/bin/python}"
REGISTRY="${REGISTRY:-configs/stage32_official_gas_protocol_registry.json}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
DATASET_DIR="${DATASET_DIR:-/mnt/project/offlinerl_datasets/ogbench}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/gas_ogbench_stage44_humanoid_visual_retrain_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-runs_stage44_humanoid_visual_retrain/${STAMP}}"
ENVS="${ENVS:-humanoidmaze-large-navigate-v0,humanoidmaze-large-stitch-v0,visual-antmaze-large-explore-v0,visual-scene-play-v0}"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS:-2}"
MAX_JOBS="${MAX_JOBS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-49}"
EVAL_VIDEO_EPISODES="${EVAL_VIDEO_EPISODES:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${GPUS}" == *,* ]]; then
  echo "Stage44 humanoid/visual retrain is constrained to one GPU; set GPUS to a single id, e.g. GPUS=2." >&2
  exit 2
fi

export OGBENCH_DATASET_DIR="${DATASET_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export PYTHONPATH="${ROOT}:${ROOT}/external_src/tmd-release:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/logs"
ln -sfn "${ROOT}/${RUN_ROOT}" "runs_stage44_humanoid_visual_retrain/latest"

"${PYTHON_BIN}" - <<'PY' "${ENVS}" "${DATASET_DIR}"
from pathlib import Path
import sys

from scripts.stage30_official_gas_common import ensure_ogbench_default_symlinks

envs = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
dataset_dir = Path(sys.argv[2])
missing = []
for env_name in envs:
    for suffix in (".npz", "-val.npz"):
        path = dataset_dir / f"{env_name}{suffix}"
        if not path.exists():
            missing.append(str(path))
    rows = ensure_ogbench_default_symlinks(env_name, dataset_dir=dataset_dir)
    for row in rows:
        print(row)
if missing:
    raise SystemExit("Missing local datasets; refusing to download: " + ", ".join(missing))
PY

cmd=(
  "${PYTHON_BIN}" scripts/stage35_launch_full_gas_training.py
  --registry "${REGISTRY}"
  --gas-repo "${GAS_REPO_PATH}"
  --artifact-root "${ARTIFACT_ROOT}"
  --run-root "${RUN_ROOT}"
  --dataset-dir "${DATASET_DIR}"
  --python-bin "${PYTHON_BIN}"
  --envs "${ENVS}"
  --seeds "${SEEDS}"
  --gpus "${GPUS}"
  --max-jobs "${MAX_JOBS}"
  --missing-only 0
  --download-datasets "${DOWNLOAD_DATASETS}"
  --repair-metadata 1
  --generate-dataset-embeddings 1
  --eval-episodes "${EVAL_EPISODES}"
  --eval-video-episodes "${EVAL_VIDEO_EPISODES}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  printf 'DATASET_DIR=%q ARTIFACT_ROOT=%q RUN_ROOT=%q ENVS=%q SEEDS=%q GPUS=%q MAX_JOBS=%q DOWNLOAD_DATASETS=%q\n' \
    "${DATASET_DIR}" "${ARTIFACT_ROOT}" "${RUN_ROOT}" "${ENVS}" "${SEEDS}" "${GPUS}" "${MAX_JOBS}" "${DOWNLOAD_DATASETS}"
} >> "${RUN_ROOT}/commands.log"

if [[ "${DETACH}" == "1" ]]; then
  setsid env \
    OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR}" \
    MUJOCO_GL="${MUJOCO_GL}" \
    XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_DISABLED="${WANDB_DISABLED}" \
    D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR}" \
    PYTHONPATH="${PYTHONPATH}" \
    "${cmd[@]}" > "${RUN_ROOT}/logs/launcher.outer.log" 2>&1 < /dev/null &
  echo "$!" > "${RUN_ROOT}/launcher.pid"
  echo "launched ${RUN_ROOT} pid=$(cat "${RUN_ROOT}/launcher.pid")"
else
  "${cmd[@]}" 2>&1 | tee "${RUN_ROOT}/logs/launcher.outer.log"
fi
