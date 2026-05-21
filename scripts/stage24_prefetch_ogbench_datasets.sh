#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

ENVS="${ENVS:-scene-play-v0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ -f scripts/stage24_env_mirrors.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/stage24_env_mirrors.sh
fi

python - <<'PY'
import os
from bars.data.ogbench_dataset import ensure_ogbench_dataset_files

envs = [x.strip() for x in os.environ.get("ENVS", "scene-play-v0").split(",") if x.strip()]
dataset_dir = os.environ.get("OGBENCH_DATASET_DIR")
for env in envs:
    train, val = ensure_ogbench_dataset_files(env, dataset_dir)
    print(f"[stage24_prefetch_ogbench] {env}: {train} | {val}", flush=True)
PY
