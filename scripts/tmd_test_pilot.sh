#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for arg in "$@"; do
  case "$arg" in
    *=*) export "$arg" ;;
  esac
done
ENVS="${ENVS:-antmaze-medium-stitch-v0}"
SEEDS="${SEEDS:-0}"
EPISODES="${EPISODES:-2}"
QUICK="${QUICK:-1}"
MODES="${MODES:-tmd_graph_tmd_actor}"
FALLBACK="${FALLBACK:-none}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/project/offlinerl_datasets/ogbench}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas_ogbench_offline_full_20260522_165138}"
GAS_REPO="${GAS_REPO:-external_src/GAS}"
GAS_SEED="${GAS_SEED:-42}"
DEFAULT_CKPT="${TMD_CHECKPOINT:-artifacts/tmd_gas_script_smoke/exp_tmd/smoke_tmd_actor/antmaze-medium-stitch-v0_sd000__2026-05-21_17-53-08/params_1000.pkl}"
mkdir -p artifacts/tmd_test runs_tmd_test reports
bash scripts/tmd_test_audit.sh
IFS=',' read -ra ENV_ARR <<< "$ENVS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra MODE_ARR <<< "$MODES"
for env_name in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    ckpt="$DEFAULT_CKPT"
    out="artifacts/tmd_test/${env_name}/${seed}"
    quick_args=()
    if [[ "$QUICK" == "1" ]]; then
      quick_args+=(--quick --max-observations 2048 --sample-size 512 --max-nodes 64 --topk-l2-candidates 16)
    fi
    bash scripts/tmd_test_construct_graph.sh \
      --env "$env_name" \
      --seed "$seed" \
      --tmd-checkpoint "$ckpt" \
      --dataset-root "$DATASET_ROOT" \
      --output-dir "$out" \
      --temporal-horizon-steps 8 \
      --pairwise-batch-size 256 \
      "${quick_args[@]}"
    for mode in "${MODE_ARR[@]}"; do
      run_dir="runs_tmd_test/${env_name}/seed${seed}/${mode}"
      bash scripts/tmd_test_eval.sh \
        --mode "$mode" \
        --env "$env_name" \
        --seed "$seed" \
        --episodes "$EPISODES" \
        --tasks "1" \
        --keygraph-path "$out/keygraph_tmd.pkl" \
        --tmd-checkpoint "$ckpt" \
        --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
        --gas-repo "$GAS_REPO" \
        --gas-seed "$GAS_SEED" \
        --fallback "$FALLBACK" \
        --dataset-root "$DATASET_ROOT" \
        --output-dir "$run_dir"
    done
  done
done
source /root/miniconda3/bin/activate gcrlo
PYTHONPATH=".:external_src/GAS:external_src/tmd-release/impls:external_src/tmd-release:${PYTHONPATH:-}" python scripts/tmd_test_analyze.py
