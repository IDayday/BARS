#!/usr/bin/env bash
set -euo pipefail

# Reuse existing BARS run directories and cached weights/artifacts.
# Expected run layout: ${LOG_ROOT}/${ENV}/${VARIANT}/seed${SEED}_*/cache/{embeddings.npy,graph.npz}

CONFIG=${CONFIG:-configs/stage28_graph_method_audit.json}
LOG_ROOT=${LOG_ROOT:-runs}
OUT_ROOT=${OUT_ROOT:-runs_stage28_graph_audit}
VARIANT=${VARIANT:-full_bars}
ENVS=${ENVS:-antmaze-medium-navigate-v0,antmaze-medium-stitch-v0,antmaze-giant-navigate-v0,antmaze-giant-stitch-v0}
SEEDS=${SEEDS:-44,45,46}
NUM_PAIRS=${NUM_PAIRS:-256}
NUM_CROSS_PAIRS=${NUM_CROSS_PAIRS:-128}
DEVICE=${DEVICE:-cuda}
GRAPH_VARIANTS=${GRAPH_VARIANTS:-base_cached,projection_temporal,dense_knn,xy_knn,endpoint_aug,bottleneck_aug}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "${OUT_ROOT}"

IFS=',' read -ra ENV_ARR <<< "${ENVS}"
IFS=',' read -ra SEED_ARR <<< "${SEEDS}"

find_run_dir() {
  local env=$1
  local seed=$2
  local pattern="${LOG_ROOT}/${env}/${VARIANT}/seed${seed}_*"
  local candidate
  candidate=$(ls -td ${pattern} 2>/dev/null | head -n 1 || true)
  if [[ -z "${candidate}" ]]; then
    pattern="${LOG_ROOT}/${env}""/**/seed${seed}_*"
    candidate=$(find "${LOG_ROOT}/${env}" -path "*seed${seed}_*" -type d 2>/dev/null | sort | tail -n 1 || true)
  fi
  if [[ -z "${candidate}" || ! -f "${candidate}/cache/graph.npz" || ! -f "${candidate}/cache/embeddings.npy" ]]; then
    return 1
  fi
  echo "${candidate}"
}

for env in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    run_dir=$(find_run_dir "${env}" "${seed}" || true)
    if [[ -z "${run_dir}" ]]; then
      echo "[stage28] missing cached run for env=${env} seed=${seed} under LOG_ROOT=${LOG_ROOT}; skipping" >&2
      continue
    fi
    out_dir="${OUT_ROOT}/${env}/${VARIANT}/seed${seed}"
    mkdir -p "${out_dir}"
    echo "[stage28] audit env=${env} seed=${seed} run_dir=${run_dir}"
    python scripts/stage28_graph_audit.py \
      --config "${CONFIG}" \
      --run-dir "${run_dir}" \
      --out "${out_dir}/stage28_graph_audit.csv" \
      --env "${env}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --num-pairs "${NUM_PAIRS}" \
      --num-cross-pairs "${NUM_CROSS_PAIRS}" \
      --graph-variants "${GRAPH_VARIANTS}" \
      --clear \
      ${EXTRA_ARGS}
  done
done

python scripts/stage28_analyze_audit.py \
  --roots "${OUT_ROOT}" \
  --out-dir "${OUT_ROOT}/_analysis"

echo "[stage28] analysis written to ${OUT_ROOT}/_analysis"
