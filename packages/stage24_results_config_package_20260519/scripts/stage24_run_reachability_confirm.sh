#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

CONFIG="${CONFIG:-configs/stage24_reachability_confirm.json}"
GPUS="${GPUS:-0}"
WAIT="${WAIT:-1}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage24}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
STAGE24_ROOT="${STAGE24_ROOT:-runs_stage24_reachability_confirm}"
LOG_ROOT="${LOG_ROOT:-runs_stage24_reachability_confirm_logs}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
QUICK="${QUICK:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
REQUIRE_RECOMMENDED_BUDGET="${REQUIRE_RECOMMENDED_BUDGET:-1}"
MAX_PLAN_EDGES="${MAX_PLAN_EDGES:-20}"
INCLUDE_REPAIR="${INCLUDE_REPAIR:-0}"
DEBUG_JSONL="${DEBUG_JSONL:-1}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-1}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs_stage24_tensorboard}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "$LOG_ROOT" "$STAGE24_ROOT" "$REPORTS_ROOT"
FAILED="$LOG_ROOT/failed_jobs.csv"
if [[ ! -f "$FAILED" ]]; then
  echo "env,seed,variant,budget,fallback,log,error_class" > "$FAILED"
fi

classify_error() {
  local log="$1"
  if grep -Eiq "ModuleNotFoundError|ImportError|No module named" "$log"; then echo "dependency/import";
  elif grep -Eiq "CUDA|cudnn|XLA|GPU|jaxlib|torch" "$log"; then echo "CUDA/JAX/MuJoCo";
  elif grep -Eiq "MuJoCo|EGL|GLFW|mujoco" "$log"; then echo "CUDA/JAX/MuJoCo";
  elif grep -Eiq "checkpoint|params_|keygraph|artifact|pickle" "$log"; then echo "checkpoint/artifact missing";
  elif grep -Eiq "goal|task_id|reset" "$log"; then echo "goal protocol";
  elif grep -Eiq "planner|budget_infeasible|no_start_connection" "$log"; then echo "planner infeasible";
  elif grep -Eiq "actor|sample_actions|skill" "$log"; then echo "GAS adapter mismatch";
  else echo "unknown"; fi
}

eval "$(python - "$CONFIG" "$INCLUDE_REPAIR" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
include_repair = sys.argv[2] == "1"
variants = list(c.get("variants", []))
if include_repair:
    variants.extend(c.get("repair_variants", []))
print("ENVS_CONF="+",".join(c.get("envs", [])))
print("SEEDS_CONF="+",".join(map(str,c.get("seeds", [0]))))
print("EPISODES_CONF="+str(c.get("episodes",100)))
print("MAX_PARALLEL_CONF="+str(c.get("max_parallel_eval",4)))
print("VARIANTS_CONF="+",".join(variants))
print("FALLBACK_CONF="+",".join(c.get("fallback_modes", ["none"])))
PY
)"

ENVS="${ENVS:-$ENVS_CONF}"
SEEDS="${SEEDS:-$SEEDS_CONF}"
EPISODES="${EPISODES:-$EPISODES_CONF}"
MAX_PARALLEL_EVAL="${MAX_PARALLEL_EVAL:-$MAX_PARALLEL_CONF}"
VARIANTS="${VARIANTS:-$VARIANTS_CONF}"
FALLBACK_MODES="${FALLBACK_MODES:-$FALLBACK_CONF}"

ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" ENVS="$ENVS" SEEDS="$SEEDS" GPUS="$GPUS" QUICK="$QUICK" PREFER_PRETRAINED="$PREFER_PRETRAINED" TRAIN_IF_MISSING="$TRAIN_IF_MISSING" LOG_ROOT="runs_stage24_prepare" bash scripts/stage22_prepare_gas_backbone.sh
ENVS="$ENVS" SEEDS="$SEEDS" GPUS="$GPUS" QUICK="$QUICK" GAS_ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" ARTIFACT_ROOT="$ARTIFACT_ROOT" GAS_REPO_PATH="$GAS_REPO_PATH" PREFER_PRETRAINED="$PREFER_PRETRAINED" TRAIN_IF_MISSING="$TRAIN_IF_MISSING" LOG_ROOT="runs_stage24_reachability_train" bash scripts/stage22_train_reachability.sh

jobs_tsv="$LOG_ROOT/jobs.tsv"
python - "$CONFIG" "$ENVS" "$SEEDS" "$EPISODES" "$VARIANTS" "$FALLBACK_MODES" "$REQUIRE_RECOMMENDED_BUDGET" > "$jobs_tsv" <<'PY'
import json, sys
from pathlib import Path

config, envs_s, seeds_s, episodes_s, variants_s, fallback_s, require_s = sys.argv[1:8]
c=json.load(open(config))
envs = [x for x in envs_s.split(",") if x]
seeds = [int(x) for x in seeds_s.split(",") if x]
episodes = int(episodes_s)
variants = [x for x in variants_s.split(",") if x]
fallback_modes = [x for x in fallback_s.split(",") if x]
require_recommended = require_s == "1"
rec_path=Path(c.get("budget_source","reports/stage22r_recommended_budgets.json"))
rec=json.load(open(rec_path)) if rec_path.exists() else {"envs":{}}
for env in envs:
  for seed in seeds:
    key=f"{env}/seed{seed}"
    info=rec.get("envs",{}).get(key,{})
    budget=info.get("recommended_reachability_budget")
    if budget is None:
      env_infos=[v for k,v in rec.get("envs",{}).items() if k.startswith(f"{env}/")]
      for cand in env_infos:
        if cand.get("recommended_reachability_budget") is not None:
          budget=cand.get("recommended_reachability_budget")
          break
    for variant in variants:
      is_reach = "reachability" in variant
      if budget is None and is_reach:
        if require_recommended:
          continue
        budget = 3.0
      variant_budget = budget if budget is not None else 999.0
      for fallback in fallback_modes:
        if fallback != "none":
          continue
        print("\t".join(map(str,[env,seed,variant,variant_budget,fallback,episodes])))
PY

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
idx=0
active=0
overall=0
while IFS=$'\t' read -r env seed variant budget fallback episodes; do
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  idx=$((idx + 1))
  run_dir="$LOG_ROOT/$env/seed$seed/$variant/budget$budget/fallback_$fallback"
  out_dir="$STAGE24_ROOT/$env/seed$seed/$variant/budget$budget/fallback_$fallback"
  mkdir -p "$run_dir" "$out_dir"
  log="$run_dir/evaluate.log"
  if [[ -f "$out_dir/eval.csv" ]]; then
    count="$(($(wc -l < "$out_dir/eval.csv") - 1))"
    if [[ "$count" -ge "$episodes" ]]; then
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"skipped\":true}" > "$run_dir/status.json"
      continue
    fi
  fi
  (
    echo "{\"status\":\"running\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"gpu\":\"$gpu\"}" > "$run_dir/status.json"
    python -m bars.gas_bars.evaluate \
      --env "$env" --seed "$seed" --task-ids all --episodes "$episodes" \
      --variant "$variant" --budget "$budget" \
      --gas-artifact-root "$GAS_ARTIFACT_ROOT" --artifact-root "$ARTIFACT_ROOT" --stage22-root "$STAGE24_ROOT" \
      --gas-repo-path "$GAS_REPO_PATH" --gpu "$gpu" --eval-on-cpu "$EVAL_ON_CPU" \
      --fallback-mode "$fallback" --prefer-pretrained "$PREFER_PRETRAINED" --train-if-missing "$TRAIN_IF_MISSING" --quick "$QUICK" \
      --max-plan-edges "$MAX_PLAN_EDGES" --debug-jsonl "$DEBUG_JSONL" \
      > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"gpu\":\"$gpu\"}" > "$run_dir/status.json"
    else
      err="$(classify_error "$log")"
      echo "$env,$seed,$variant,$budget,$fallback,$log,$err" >> "$FAILED"
      echo "{\"status\":\"failed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"budget\":\"$budget\",\"fallback\":\"$fallback\",\"gpu\":\"$gpu\",\"error_class\":\"$err\"}" > "$run_dir/status.json"
    fi
    exit "$rc"
  ) &
  active=$((active + 1))
  if [[ "$active" -ge "$MAX_PARALLEL_EVAL" ]]; then
    wait -n || overall=1
    active=$((active - 1))
    python scripts/stage24_analyze.py --reachability-roots "$STAGE24_ROOT" --local-drift-roots "$STAGE24_ROOT" || true
  fi
done < "$jobs_tsv"

while [[ "$active" -gt 0 ]]; do
  wait -n || overall=1
  active=$((active - 1))
done

python scripts/stage24_local_drift_diagnostic.py --eval-roots "$STAGE24_ROOT" --out "$REPORTS_ROOT/stage24_local_drift.csv" || true
python scripts/stage24_analyze.py --reachability-roots "$STAGE24_ROOT" --local-drift-roots "$STAGE24_ROOT" --reports-root "$REPORTS_ROOT" || true

if [[ "$WAIT" == "1" ]]; then
  echo "[stage24_reachability] reports written under $REPORTS_ROOT; root=$STAGE24_ROOT"
fi
exit "$overall"
