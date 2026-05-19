#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

MODE="${MODE:-all_adaptive}"
ENVS="${ENVS:-antmaze-medium-navigate-v0,antmaze-medium-stitch-v0}"
HARD_ENVS="${HARD_ENVS:-antmaze-large-stitch-v0,antmaze-giant-stitch-v0,antmaze-giant-navigate-v0}"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS:-cpu}"
GPU_FIRST="${GPUS%%,*}"
WAIT="${WAIT:-1}"
EPISODES_REPRO="${EPISODES_REPRO:-100}"
EPISODES_ATLAS="${EPISODES_ATLAS:-100}"
EPISODES_INTEGRATED="${EPISODES_INTEGRATED:-100}"
EDGE_EXEC_PILOT="${EDGE_EXEC_PILOT:-1}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage23}"
STAGE22_ARTIFACT_ROOT="${STAGE22_ARTIFACT_ROOT:-artifacts/stage22}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-1}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs_stage23_tensorboard}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "$REPORTS_ROOT" "$ARTIFACT_ROOT"

run_repro() {
  python scripts/stage23_protocol_audit.py \
    --envs "$ENVS" --seeds "$SEEDS" --gpus "$GPUS" \
    --gas-artifact-root "$GAS_ARTIFACT_ROOT" --gas-repo-path "$GAS_REPO_PATH" \
    --out-csv "$REPORTS_ROOT/stage23_protocol_audit.csv" \
    --out-md "$REPORTS_ROOT/stage23_protocol_audit.md"
  python scripts/stage23_gas_repro.py \
    --envs "$ENVS" --seeds "$SEEDS" --gpus "$GPUS" \
    --episodes "$EPISODES_REPRO" \
    --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
    --stage22-artifact-root "$STAGE22_ARTIFACT_ROOT" \
    --gas-repo-path "$GAS_REPO_PATH" \
    --run-root runs_stage23_repro \
    --reports-root "$REPORTS_ROOT" \
    --prefer-pretrained "$PREFER_PRETRAINED" \
    --train-if-missing "$TRAIN_IF_MISSING"
}

run_atlas() {
  IFS=',' read -r -a ENV_ARR <<< "$ENVS"
  IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
  for env in "${ENV_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      python -m bars.gas_bars.evaluate \
        --env "$env" --seed "$seed" --episodes "$EPISODES_ATLAS" \
        --variant gas_shortest --budget 999 --fallback-mode none \
        --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
        --artifact-root "$STAGE22_ARTIFACT_ROOT" \
        --stage22-root runs_stage23_atlas \
        --gas-repo-path "$GAS_REPO_PATH" \
        --gpu "$GPU_FIRST" --eval-on-cpu 1 \
        --prefer-pretrained "$PREFER_PRETRAINED" \
        --train-if-missing "$TRAIN_IF_MISSING" \
        --quick 1 || true
    done
  done
  python scripts/stage23_failure_atlas.py \
    --eval-roots runs_stage23_atlas,runs_stage23_repro,runs_stage23_integrated \
    --artifact-root "$ARTIFACT_ROOT" \
    --reports-root "$REPORTS_ROOT"
}

run_bridge() {
  python scripts/stage23_build_bridge_graphs.py \
    --envs "$ENVS" --seeds "$SEEDS" \
    --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --reports-root "$REPORTS_ROOT"
}

run_edge_exec() {
  python scripts/stage23_edge_execution.py \
    --envs "$ENVS" --seeds "$SEEDS" --graph-id G3 \
    --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --gas-repo-path "$GAS_REPO_PATH" \
    --reports-root "$REPORTS_ROOT" \
    --gpu "$GPU_FIRST" \
    --pilot "$EDGE_EXEC_PILOT" \
    --prefer-pretrained "$PREFER_PRETRAINED" \
    --train-if-missing "$TRAIN_IF_MISSING"
}

run_oracle() {
  python scripts/stage23_oracle_bridge.py \
    --envs "$ENVS" --seeds "$SEEDS" --graph-id G3 \
    --artifact-root "$ARTIFACT_ROOT" \
    --reports-root "$REPORTS_ROOT"
}

run_train_p_bridge() {
  python scripts/stage23_train_p_bridge.py \
    --envs "$ENVS" --seeds "$SEEDS" \
    --artifact-root "$ARTIFACT_ROOT" \
    --reports-root "$REPORTS_ROOT" \
    --epochs "${P_BRIDGE_EPOCHS:-25}" \
    --device "${P_BRIDGE_DEVICE:-cpu}"
  python scripts/stage23_score_bridges.py \
    --envs "$ENVS" --seeds "$SEEDS" \
    --artifact-root "$ARTIFACT_ROOT" \
    --device "${P_BRIDGE_DEVICE:-cpu}"
}

run_boundary() {
  python scripts/stage23_boundary_junction_diagnostic.py \
    --envs "$ENVS" --seeds "$SEEDS" \
    --artifact-root "$ARTIFACT_ROOT" \
    --stage22-artifact-root "$STAGE22_ARTIFACT_ROOT" \
    --reports-root "$REPORTS_ROOT"
}

run_integrated() {
  IFS=',' read -r -a ENV_ARR <<< "$ENVS"
  IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
  VARIANTS="${VARIANTS:-official_gas_shortest_G0,aggressive_shortest_G3,oracle_bridge_shortest,p_bridge_budget}"
  IFS=',' read -r -a VAR_ARR <<< "$VARIANTS"
  for env in "${ENV_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      for variant in "${VAR_ARR[@]}"; do
        python scripts/stage23_integrated_eval.py \
          --env "$env" --seed "$seed" --episodes "$EPISODES_INTEGRATED" \
          --variant "$variant" \
          --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
          --artifact-root "$ARTIFACT_ROOT" \
          --gas-repo-path "$GAS_REPO_PATH" \
          --run-root runs_stage23_integrated \
          --gpu "$GPU_FIRST" \
          --prefer-pretrained "$PREFER_PRETRAINED" \
          --train-if-missing "$TRAIN_IF_MISSING" || true
      done
    done
  done
}

run_analyze() {
  python scripts/stage23_monitor.py \
    --roots runs_stage23_repro,runs_stage23_atlas,runs_stage23_integrated \
    --artifact-root "$ARTIFACT_ROOT" \
    --summary-md "$REPORTS_ROOT/stage23_live_summary.md" \
    --failed-csv "$REPORTS_ROOT/stage23_failed_jobs.csv" \
    --gate-json "$REPORTS_ROOT/stage23_gate_status.json"
  python scripts/analyze_stage23.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --runs-root . \
    --reports-root "$REPORTS_ROOT" \
    --out "$REPORTS_ROOT/stage23_summary.md"
}

case "$MODE" in
  repro) run_repro ;;
  atlas) run_atlas ;;
  bridge) run_bridge ;;
  edge_exec) run_edge_exec ;;
  oracle) run_oracle ;;
  train_p_bridge) run_train_p_bridge ;;
  boundary) run_boundary ;;
  integrated) run_integrated ;;
  fallback) python scripts/stage23_fallback_causal_ablation.py --reports-root "$REPORTS_ROOT" ;;
  all_adaptive)
    run_repro || true
    run_atlas || true
    OLD_ENVS="$ENVS"
    ENVS="$OLD_ENVS"
    run_bridge || true
    if [[ -n "$HARD_ENVS" ]]; then
      ENVS="$HARD_ENVS"
      run_bridge || true
      run_edge_exec || true
      run_oracle || true
      run_train_p_bridge || true
      run_boundary || true
      if [[ -f "$REPORTS_ROOT/stage23_gas_reproduction_matrix.csv" ]] && ! grep -Eq ',(failed|skipped),' "$REPORTS_ROOT/stage23_gas_reproduction_matrix.csv"; then
        run_integrated || true
      else
        echo "[stage23_pipeline] HOLD_REPRO: skipping integrated BARS-v3."
      fi
    fi
    ENVS="$OLD_ENVS"
    python scripts/stage23_fallback_causal_ablation.py --reports-root "$REPORTS_ROOT" || true
    run_analyze
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac

run_analyze || true
