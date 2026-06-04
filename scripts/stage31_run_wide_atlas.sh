#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/gas_ogbench_offline_full_20260522_165138}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/project/offlinerl_datasets}"
INVENTORY_ROOT="${INVENTORY_ROOT:-runs_stage31_official_gas/wide_artifact_inventory_active}"
INVENTORY_CSV="${INVENTORY_CSV:-${INVENTORY_ROOT}/official_gas_artifact_inventory.csv}"
OUT_ROOT="${OUT_ROOT:-runs_stage31_official_gas/wide_atlas_$(date +%Y%m%d_%H%M%S)}"
ENVS="${ENVS:-auto}"
SEEDS="${SEEDS:-44,45,46}"
TASK_IDS="${TASK_IDS:-auto}"
MAX_TASK_ID="${MAX_TASK_ID:-5}"
EPISODES="${EPISODES:-50}"
GPU="${GPU:-0}"
GPUS="${GPUS:-${GPU}}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
FALLBACK_MODE="${FALLBACK_MODE:-none}"
RECOVER_DATASET_INDICES="${RECOVER_DATASET_INDICES:-0}"
NODE_MAP_BATCH_SIZE="${NODE_MAP_BATCH_SIZE:-4096}"
NODE_MAP_TOLERANCE="${NODE_MAP_TOLERANCE:-1e-5}"
MAX_JOBS="${MAX_JOBS:-999}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${OUT_ROOT}" "${INVENTORY_ROOT}"

if [ ! -s "${INVENTORY_CSV}" ]; then
  python scripts/stage31_official_gas_artifact_inventory.py \
    --artifact-root "${ARTIFACT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --seeds "${SEEDS}" \
    --out-root "${INVENTORY_ROOT}"
fi

if [ "${ENVS}" = "auto" ]; then
  ENVS="$(python - "${INVENTORY_CSV}" <<'PY'
import csv
import sys

path = sys.argv[1]
envs = []
with open(path, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        if row.get("artifact_status") != "READY_OFFICIAL_GAS":
            continue
        env = row.get("env_name", "")
        if env and env not in envs:
            envs.append(env)
print(",".join(envs))
PY
)"
fi

IFS=',' read -r -a env_arr <<< "${ENVS}"
IFS=',' read -r -a seed_arr <<< "${SEEDS}"
IFS=',' read -r -a gpu_arr <<< "${GPUS}"

gpu_for_job() {
  local job_index="$1"
  if [ "${#gpu_arr[@]}" -eq 0 ]; then
    echo "${GPU}"
  else
    echo "${gpu_arr[$((job_index % ${#gpu_arr[@]}))]}"
  fi
}

phase_done() {
  local env_name="$1"
  local seed="$2"
  local output="${OUT_ROOT}/${env_name}/seed${seed}/instrumentation/official_gas_episode_traces.csv"
  [ -s "${output}" ]
}

run_instrumentation() {
  local env_name="$1"
  local seed="$2"
  local gpu_id="$3"
  local run_dir="${OUT_ROOT}/${env_name}/seed${seed}"
  mkdir -p "${run_dir}/instrumentation"

  python scripts/stage30_official_gas_instrument.py \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --out-root "${run_dir}/instrumentation" \
    --envs "${env_name}" \
    --seeds "${seed}" \
    --task-ids "${TASK_IDS}" \
    --max-task-id "${MAX_TASK_ID}" \
    --episodes "${EPISODES}" \
    --eval-on-cpu "${EVAL_ON_CPU}" \
    --gpu "${gpu_id}" \
    --fallback-mode "${FALLBACK_MODE}" \
    --recover-dataset-indices "${RECOVER_DATASET_INDICES}" \
    --node-map-batch-size "${NODE_MAP_BATCH_SIZE}" \
    --node-map-tolerance "${NODE_MAP_TOLERANCE}"
}

failures=0
active=0
job_index=0
for env_name in "${env_arr[@]}"; do
  if [ -z "${env_name}" ]; then
    continue
  fi
  for seed in "${seed_arr[@]}"; do
    if [ "${SKIP_EXISTING}" = "1" ] && phase_done "${env_name}" "${seed}"; then
      echo "[stage31] skip existing instrumentation ${env_name} seed${seed}"
      continue
    fi
    gpu_id="$(gpu_for_job "${job_index}")"
    mkdir -p "${OUT_ROOT}"
    run_instrumentation "${env_name}" "${seed}" "${gpu_id}" > "${OUT_ROOT}/${env_name}_seed${seed}.instrumentation.log" 2>&1 &
    active=$((active + 1))
    job_index=$((job_index + 1))
    if [ "${active}" -ge "${MAX_JOBS}" ]; then
      set +e
      wait -n
      code="$?"
      set -e
      if [ "${code}" -ne 0 ]; then
        failures=$((failures + 1))
      fi
      active=$((active - 1))
    fi
  done
done

while [ "${active}" -gt 0 ]; do
  set +e
  wait -n
  code="$?"
  set -e
  if [ "${code}" -ne 0 ]; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done

python scripts/stage31_collect_wide_atlas.py \
  --run-root "${OUT_ROOT}" \
  --inventory-csv "${INVENTORY_CSV}" \
  --out-root "${OUT_ROOT}/global" > "${OUT_ROOT}/collector.log" 2>&1

python - "${OUT_ROOT}/stage31_run_status.csv" <<PY
import csv
import sys

path = sys.argv[1]
row = {
    "stage": "stage31_wide_official_gas_behavior_atlas",
    "evidence_class": "OFFICIAL_GAS_WIDE_RUN_STATUS",
    "out_root": "${OUT_ROOT}",
    "inventory_csv": "${INVENTORY_CSV}",
    "envs": "${ENVS}",
    "seeds": "${SEEDS}",
    "task_ids": "${TASK_IDS}",
    "max_task_id": "${MAX_TASK_ID}",
    "episodes": "${EPISODES}",
    "eval_on_cpu": "${EVAL_ON_CPU}",
    "gpus": "${GPUS}",
    "max_jobs": "${MAX_JOBS}",
    "skip_existing": "${SKIP_EXISTING}",
    "instrumentation_failures": "${failures}",
}
with open(path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
PY

echo "${OUT_ROOT}"
