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
MAX_JOBS=${MAX_JOBS:-1}
AUDIT_WORKERS=${AUDIT_WORKERS:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}
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

is_complete() {
  local out_file=$1
  local expected_pairs=$((NUM_PAIRS + NUM_CROSS_PAIRS))
  python - "${out_file}" "${expected_pairs}" <<'PY'
import csv
import os
import sys

path = sys.argv[1]
expected_pairs = int(sys.argv[2])
if not os.path.exists(path) or os.path.getsize(path) == 0:
    raise SystemExit(1)
proxy = 0
summary = 0
try:
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            phase = row.get("phase", "")
            if phase == "stage28_failure_taxonomy_proxy":
                proxy += 1
            elif phase == "stage28_failure_taxonomy_summary":
                summary += 1
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if proxy >= expected_pairs and summary > 0 else 1)
PY
}

run_one() {
  local env=$1
  local seed=$2
  local run_dir
  run_dir=$(find_run_dir "${env}" "${seed}" || true)
  if [[ -z "${run_dir}" ]]; then
    echo "[stage28] missing cached run for env=${env} seed=${seed} under LOG_ROOT=${LOG_ROOT}; skipping" >&2
    return 0
  fi
  local out_dir="${OUT_ROOT}/${env}/${VARIANT}/seed${seed}"
  local out_file="${out_dir}/stage28_graph_audit.csv"
  mkdir -p "${out_dir}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && is_complete "${out_file}"; then
    echo "[stage28] complete env=${env} seed=${seed}; skipping"
    return 0
  fi
  echo "[stage28] audit env=${env} seed=${seed} workers=${AUDIT_WORKERS} run_dir=${run_dir}"
  python scripts/stage28_graph_audit.py \
    --config "${CONFIG}" \
    --run-dir "${run_dir}" \
    --out "${out_file}" \
    --env "${env}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --num-pairs "${NUM_PAIRS}" \
    --num-cross-pairs "${NUM_CROSS_PAIRS}" \
    --num-workers "${AUDIT_WORKERS}" \
    --graph-variants "${GRAPH_VARIANTS}" \
    --clear \
    ${EXTRA_ARGS}
}

wait_for_slot() {
  local failed_ref=$1
  while [[ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]]; do
    if ! wait -n; then
      printf -v "${failed_ref}" '%s' "1"
    fi
  done
}

failed=0
for env in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    if [[ "${MAX_JOBS}" -le 1 ]]; then
      if ! run_one "${env}" "${seed}"; then
        failed=1
      fi
    else
      wait_for_slot failed
      run_one "${env}" "${seed}" &
    fi
  done
done

if [[ "${MAX_JOBS}" -gt 1 ]]; then
  while [[ $(jobs -rp | wc -l) -gt 0 ]]; do
    if ! wait -n; then
      failed=1
    fi
  done
fi

if [[ "${failed}" != "0" ]]; then
  echo "[stage28] one or more audit jobs failed; not running analysis" >&2
  exit 1
fi

python scripts/stage28_analyze_audit.py \
  --roots "${OUT_ROOT}" \
  --out-dir "${OUT_ROOT}/_analysis"

echo "[stage28] analysis written to ${OUT_ROOT}/_analysis"
