#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/gas_ogbench_offline_full_20260522_165138}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
OUT_ROOT="${OUT_ROOT:-runs_stage30_official_gas/layered_$(date +%Y%m%d_%H%M%S)}"
ENVS="${ENVS:-antmaze-medium-navigate-v0,antmaze-medium-stitch-v0,antmaze-giant-navigate-v0,antmaze-giant-stitch-v0}"
SEEDS="${SEEDS:-44,45,46}"
TASK_IDS="${TASK_IDS:-auto}"
EPISODES="${EPISODES:-100}"
EDGES_PER_CATEGORY="${EDGES_PER_CATEGORY:-200}"
GPU="${GPU:-cpu}"
GPUS="${GPUS:-${GPU}}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
FALLBACK_MODE="${FALLBACK_MODE:-none}"
RECOVER_DATASET_INDICES="${RECOVER_DATASET_INDICES:-1}"
NODE_MAP_TOLERANCE="${NODE_MAP_TOLERANCE:-1e-5}"
EXACT_NODE_MAP_TOLERANCE="${EXACT_NODE_MAP_TOLERANCE:-1e-5}"
NEAREST_NODE_MAP_TOLERANCE="${NEAREST_NODE_MAP_TOLERANCE:-1.0}"
NODE_MAP_BATCH_SIZE="${NODE_MAP_BATCH_SIZE:-4096}"
MAX_JOBS="${MAX_JOBS:-999}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "${OUT_ROOT}"

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

run_instrumentation() {
  local env_name="$1"
  local seed="$2"
  local gpu_id="$3"
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
    --gpu "${gpu_id}" \
    --fallback-mode "${FALLBACK_MODE}" \
    --recover-dataset-indices "${RECOVER_DATASET_INDICES}" \
    --node-map-batch-size "${NODE_MAP_BATCH_SIZE}" \
    --node-map-tolerance "${NODE_MAP_TOLERANCE}"
}

run_keygraph_audit() {
  local env_name="$1"
  local seed="$2"
  local gpu_id="$3"
  local run_dir="${OUT_ROOT}/${env_name}/seed${seed}"
  mkdir -p "${run_dir}"

  python scripts/stage30_official_gas_keygraph_audit.py \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --out-root "${run_dir}/keygraph_audit" \
    --envs "${env_name}" \
    --seeds "${seed}" \
    --path-edge-csv "${run_dir}/instrumentation/official_gas_path_edges.csv" \
    --recover-dataset-indices "${RECOVER_DATASET_INDICES}" \
    --node-map-batch-size "${NODE_MAP_BATCH_SIZE}" \
    --node-map-tolerance "${EXACT_NODE_MAP_TOLERANCE}" \
    --gpu "${gpu_id}"
}

run_edge_probe() {
  local env_name="$1"
  local seed="$2"
  local gpu_id="$3"
  local probe_mode="$4"
  local tolerance="$5"
  local out_name="$6"
  local run_dir="${OUT_ROOT}/${env_name}/seed${seed}"
  mkdir -p "${run_dir}"

  python scripts/stage30_official_gas_edge_probe.py \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --out-root "${run_dir}/${out_name}" \
    --envs "${env_name}" \
    --seeds "${seed}" \
    --path-edge-csv "${run_dir}/instrumentation/official_gas_path_edges.csv" \
    --edges-per-category "${EDGES_PER_CATEGORY}" \
    --eval-on-cpu "${EVAL_ON_CPU}" \
    --gpu "${gpu_id}" \
    --probe-mode "${probe_mode}" \
    --fallback-mode "${FALLBACK_MODE}" \
    --node-map-batch-size "${NODE_MAP_BATCH_SIZE}" \
    --node-map-tolerance "${tolerance}"
}

run_taxonomy() {
  local env_name="$1"
  local seed="$2"
  local gpu_id="$3"
  local run_dir="${OUT_ROOT}/${env_name}/seed${seed}"
  mkdir -p "${run_dir}"

  python scripts/stage30_official_gas_analyze.py \
    --episode-csv "${run_dir}/instrumentation/official_gas_episode_traces.csv" \
    --path-edge-csv "${run_dir}/instrumentation/official_gas_path_edges.csv" \
    --edge-probe-csv "${run_dir}/exact_semantic_probe/official_gas_edge_probe.csv,${run_dir}/nearest_execution_probe/official_gas_edge_probe.csv" \
    --out-root "${run_dir}/taxonomy" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --envs "${env_name}" \
    --seeds "${seed}"
}

phase_done() {
  local phase_name="$1"
  local env_name="$2"
  local seed="$3"
  local run_dir="${OUT_ROOT}/${env_name}/seed${seed}"
  local output=""
  case "${phase_name}" in
    instrumentation)
      output="${run_dir}/instrumentation/official_gas_episode_traces.csv"
      ;;
    keygraph_audit)
      output="${run_dir}/keygraph_audit/official_gas_keygraph_edges.csv"
      ;;
    exact_semantic_probe)
      output="${run_dir}/exact_semantic_probe/official_gas_edge_probe.csv"
      ;;
    nearest_execution_probe)
      output="${run_dir}/nearest_execution_probe/official_gas_edge_probe.csv"
      ;;
    taxonomy)
      output="${run_dir}/taxonomy/official_gas_failure_taxonomy.csv"
      ;;
    *)
      return 1
      ;;
  esac
  [ -s "${output}" ]
}

run_phase() {
  local phase_name="$1"
  local active=0
  local job_index=0
  for env_name in "${env_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      if [ "${SKIP_EXISTING}" = "1" ] && phase_done "${phase_name}" "${env_name}" "${seed}"; then
        echo "[stage30] skip existing ${phase_name} ${env_name} seed${seed}"
        continue
      fi
      local gpu_id
      gpu_id="$(gpu_for_job "${job_index}")"
      case "${phase_name}" in
        instrumentation)
          run_instrumentation "${env_name}" "${seed}" "${gpu_id}" > "${OUT_ROOT}/${env_name}_seed${seed}.instrumentation.log" 2>&1 &
          ;;
        keygraph_audit)
          run_keygraph_audit "${env_name}" "${seed}" "${gpu_id}" > "${OUT_ROOT}/${env_name}_seed${seed}.keygraph_audit.log" 2>&1 &
          ;;
        exact_semantic_probe)
          run_edge_probe "${env_name}" "${seed}" "${gpu_id}" exact_semantic_probe "${EXACT_NODE_MAP_TOLERANCE}" exact_semantic_probe > "${OUT_ROOT}/${env_name}_seed${seed}.exact_semantic_probe.log" 2>&1 &
          ;;
        nearest_execution_probe)
          run_edge_probe "${env_name}" "${seed}" "${gpu_id}" nearest_execution_probe "${NEAREST_NODE_MAP_TOLERANCE}" nearest_execution_probe > "${OUT_ROOT}/${env_name}_seed${seed}.nearest_execution_probe.log" 2>&1 &
          ;;
        taxonomy)
          run_taxonomy "${env_name}" "${seed}" "${gpu_id}" > "${OUT_ROOT}/${env_name}_seed${seed}.taxonomy.log" 2>&1 &
          ;;
        *)
          echo "Unknown phase: ${phase_name}" >&2
          exit 2
          ;;
      esac
      active=$((active + 1))
      job_index=$((job_index + 1))
      if [ "${active}" -ge "${MAX_JOBS}" ]; then
        wait -n
        active=$((active - 1))
      fi
    done
  done
  wait
}

run_phase instrumentation
run_phase keygraph_audit
run_phase exact_semantic_probe
run_phase nearest_execution_probe
run_phase taxonomy

python scripts/stage30_collect_layered_outputs.py \
  --run-root "${OUT_ROOT}" \
  --out-root "${OUT_ROOT}/global" > "${OUT_ROOT}/collector.log" 2>&1

echo "${OUT_ROOT}"
