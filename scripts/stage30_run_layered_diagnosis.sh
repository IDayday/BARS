#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/gas_ogbench_offline_full_20260522_165138}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
OUT_ROOT="${OUT_ROOT:-runs_stage30_official_gas/layered_$(date +%Y%m%d_%H%M%S)}"
ENVS="${ENVS:-antmaze-medium-navigate-v0,antmaze-medium-stitch-v0,antmaze-giant-navigate-v0,antmaze-giant-stitch-v0}"
SEEDS="${SEEDS:-44,45,46}"
TASK_IDS="${TASK_IDS:-1}"
EPISODES="${EPISODES:-49}"
EDGES_PER_CATEGORY="${EDGES_PER_CATEGORY:-200}"
GPU="${GPU:-cpu}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
RECOVER_DATASET_INDICES="${RECOVER_DATASET_INDICES:-1}"
NODE_MAP_TOLERANCE="${NODE_MAP_TOLERANCE:-1e-5}"
NODE_MAP_BATCH_SIZE="${NODE_MAP_BATCH_SIZE:-4096}"
MAX_JOBS="${MAX_JOBS:-1}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local env_name="$1"
  local seed="$2"
  local run_dir="${OUT_ROOT}/${env_name}/seed${seed}"
  mkdir -p "${run_dir}"

  python scripts/stage30_official_gas_instrument.py \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --out-root "${run_dir}/instrumentation" \
    --envs "${env_name}" \
    --seeds "${seed}" \
    --task-ids "${TASK_IDS}" \
    --episodes "${EPISODES}" \
    --eval-on-cpu "${EVAL_ON_CPU}" \
    --gpu "${GPU}" \
    --recover-dataset-indices "${RECOVER_DATASET_INDICES}" \
    --node-map-batch-size "${NODE_MAP_BATCH_SIZE}" \
    --node-map-tolerance "${NODE_MAP_TOLERANCE}"

  python scripts/stage30_official_gas_keygraph_audit.py \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --out-root "${run_dir}/keygraph_audit" \
    --envs "${env_name}" \
    --seeds "${seed}" \
    --path-edge-csv "${run_dir}/instrumentation/official_gas_path_edges.csv" \
    --recover-dataset-indices "${RECOVER_DATASET_INDICES}" \
    --node-map-batch-size "${NODE_MAP_BATCH_SIZE}" \
    --node-map-tolerance "${NODE_MAP_TOLERANCE}" \
    --gpu "${GPU}"

  python scripts/stage30_official_gas_edge_probe.py \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --out-root "${run_dir}/edge_probe" \
    --envs "${env_name}" \
    --seeds "${seed}" \
    --path-edge-csv "${run_dir}/instrumentation/official_gas_path_edges.csv" \
    --edges-per-category "${EDGES_PER_CATEGORY}" \
    --eval-on-cpu "${EVAL_ON_CPU}" \
    --gpu "${GPU}" \
    --node-map-batch-size "${NODE_MAP_BATCH_SIZE}" \
    --node-map-tolerance "${NODE_MAP_TOLERANCE}"

  python scripts/stage30_official_gas_analyze.py \
    --episode-csv "${run_dir}/instrumentation/official_gas_episode_traces.csv" \
    --path-edge-csv "${run_dir}/instrumentation/official_gas_path_edges.csv" \
    --edge-probe-csv "${run_dir}/edge_probe/official_gas_edge_probe.csv" \
    --out-root "${run_dir}/taxonomy" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --envs "${env_name}" \
    --seeds "${seed}"
}

IFS=',' read -r -a env_arr <<< "${ENVS}"
IFS=',' read -r -a seed_arr <<< "${SEEDS}"

active=0
for env_name in "${env_arr[@]}"; do
  for seed in "${seed_arr[@]}"; do
    run_one "${env_name}" "${seed}" > "${OUT_ROOT}/${env_name}_seed${seed}.log" 2>&1 &
    active=$((active + 1))
    if [ "${active}" -ge "${MAX_JOBS}" ]; then
      wait -n
      active=$((active - 1))
    fi
  done
done

wait
echo "${OUT_ROOT}"
