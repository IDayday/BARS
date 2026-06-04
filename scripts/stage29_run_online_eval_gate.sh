#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/stage29_support_calibrated_graph.json}
LOG_ROOT=${LOG_ROOT:-runs_stage28_cached_gas}
OUT_ROOT=${OUT_ROOT:-runs_stage29_online_eval_gate}
VARIANT=${VARIANT:-full_bars}
ENVS=${ENVS:-antmaze-medium-navigate-v0,antmaze-medium-stitch-v0,antmaze-giant-navigate-v0,antmaze-giant-stitch-v0}
SEEDS=${SEEDS:-44,45,46}
EPISODES=${EPISODES:-20}
PLANNERS=${PLANNERS:-BARS_BASE,STAGE29_LEXICOGRAPHIC,SUPPORT_BUDGET_K0,SUPPORT_BUDGET_K1,SUPPORT_BUDGET_K2}
SUPPORT_RISK_BUDGET=${SUPPORT_RISK_BUDGET:-1.0}
DEVICE=${DEVICE:-cpu}
GPU=${GPU:-0}
MAX_JOBS=${MAX_JOBS:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}
PYTHON=${PYTHON:-python}
GAS_ARTIFACT_ROOT=${GAS_ARTIFACT_ROOT:-artifacts/gas_ogbench_offline_full_20260522_165138}
GAS_REPO_PATH=${GAS_REPO_PATH:-external_src/GAS}
STAGE29B_FULL_CALIBRATION_CSV=${STAGE29B_FULL_CALIBRATION_CSV:-runs_stage29b_edge_execution/antmaze-medium-stitch-v0/seed44/stage29b_full_calibration.csv}
STAGE29B_TEMPORAL_SANITY_CSV=${STAGE29B_TEMPORAL_SANITY_CSV:-runs_stage29b_edge_execution/antmaze-medium-stitch-v0/seed44/stage29b_temporal_sanity.csv}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ "${EPISODES}" != "20" ]]; then
  echo "[stage29-online] refusing EPISODES=${EPISODES}; this gate is 20ep-only" >&2
  exit 2
fi

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
    candidate=$(find "${LOG_ROOT}/${env}" -path "*seed${seed}_*" -type d 2>/dev/null | sort | tail -n 1 || true)
  fi
  if [[ -z "${candidate}" || ! -f "${candidate}/cache/graph.npz" || ! -f "${candidate}/cache/embeddings.npy" ]]; then
    return 1
  fi
  echo "${candidate}"
}

is_complete() {
  local out_file=$1
  "${PYTHON}" - "${out_file}" "${EPISODES}" "${PLANNERS}" <<'PY'
import csv
import os
import sys

path = sys.argv[1]
episodes = int(sys.argv[2])
planners = [p for p in sys.argv[3].split(",") if p and p != "SUPPORT_PLUS_ENDPOINT"]
if not os.path.exists(path) or os.path.getsize(path) == 0:
    raise SystemExit(1)
counts = {p: 0 for p in planners}
summary = set()
with open(path, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row.get("phase") == "stage29_online_eval_episode" and row.get("planner_id") in counts:
            counts[row["planner_id"]] += 1
        elif row.get("phase") == "stage29_online_eval_summary":
            summary.add(row.get("planner_id"))
ok = all(counts[p] >= episodes for p in planners) and all(p in summary for p in planners)
raise SystemExit(0 if ok else 1)
PY
}

run_one() {
  local env=$1
  local seed=$2
  local run_dir
  run_dir=$(find_run_dir "${env}" "${seed}" || true)
  if [[ -z "${run_dir}" ]]; then
    echo "[stage29-online] missing cached GAS run for env=${env} seed=${seed} under LOG_ROOT=${LOG_ROOT}; skipping" >&2
    return 0
  fi
  local out_dir="${OUT_ROOT}/${env}/${VARIANT}/seed${seed}"
  local out_file="${out_dir}/stage29_online_eval_gate.csv"
  local report_file="${out_dir}/stage29_online_eval_gate.md"
  mkdir -p "${out_dir}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && is_complete "${out_file}"; then
    echo "[stage29-online] complete env=${env} seed=${seed}; skipping"
    return 0
  fi
  echo "[stage29-online] 20ep env=${env} seed=${seed} planners=${PLANNERS} run_dir=${run_dir}"
  OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/mnt/project/offlinerl_datasets/ogbench}" \
  "${PYTHON}" scripts/stage29_online_eval_gate.py \
    --config "${CONFIG}" \
    --run-dir "${run_dir}" \
    --out "${out_file}" \
    --report "${report_file}" \
    --env "${env}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --episodes "${EPISODES}" \
    --planners "${PLANNERS}" \
    --support-risk-budget "${SUPPORT_RISK_BUDGET}" \
    --gpu "${GPU}" \
    --gas-artifact-root "${GAS_ARTIFACT_ROOT}" \
    --gas-repo-path "${GAS_REPO_PATH}" \
    --stage29b-full-calibration-csv "${STAGE29B_FULL_CALIBRATION_CSV}" \
    --stage29b-temporal-sanity-csv "${STAGE29B_TEMPORAL_SANITY_CSV}" \
    --set data.dataset_dir="${OGBENCH_DATASET_DIR:-/mnt/project/offlinerl_datasets/ogbench}" \
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
  echo "[stage29-online] one or more online eval jobs failed; not running analysis" >&2
  exit 1
fi

"${PYTHON}" scripts/stage29_analyze_online_eval_gate.py \
  --roots "${OUT_ROOT}" \
  --out-dir "${OUT_ROOT}/_analysis"

echo "[stage29-online] analysis written to ${OUT_ROOT}/_analysis"
