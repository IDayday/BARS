#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

CONFIG="${CONFIG:-configs/stage25_local_drift_v2.json}"
ENVS="${ENVS:-}"
SEEDS="${SEEDS:-}"
EPISODES="${EPISODES:-}"
VARIANTS="${VARIANTS:-}"
FALLBACK_MODES="${FALLBACK_MODES:-}"
GPUS="${GPUS:-0}"
WAIT="${WAIT:-1}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage25}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
STAGE25_ROOT="${STAGE25_ROOT:-runs_stage25_local_drift_v2}"
LOG_ROOT="${LOG_ROOT:-runs_stage25_local_drift_v2_logs}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
QUICK="${QUICK:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
MAX_PARALLEL_EVAL="${MAX_PARALLEL_EVAL:-}"
MAX_PLAN_EDGES="${MAX_PLAN_EDGES:-20}"
MAX_STEPS="${MAX_STEPS:-1000}"
DEBUG_JSONL="${DEBUG_JSONL:-1}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_REACHABILITY_TRAIN="${SKIP_REACHABILITY_TRAIN:-1}"
ANALYZE_MIN_EPISODES="${ANALYZE_MIN_EPISODES:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export BARS_USE_TENSORBOARD="${BARS_USE_TENSORBOARD:-1}"
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-runs_stage25_tensorboard}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ -f scripts/stage24_env_mirrors.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/stage24_env_mirrors.sh
fi

mkdir -p "$LOG_ROOT" "$STAGE25_ROOT" "$REPORTS_ROOT"
FAILED="$LOG_ROOT/failed_jobs.csv"
if [[ ! -f "$FAILED" ]]; then
  echo "env,seed,variant,budget,fallback,log,error_class" > "$FAILED"
fi

classify_error() {
  local log="$1"
  if grep -Eiq "ModuleNotFoundError|ImportError|No module named" "$log"; then echo "dependency/import";
  elif grep -Eiq "CUDA|cudnn|XLA|GPU|jaxlib|torch|MuJoCo|EGL|GLFW|mujoco" "$log"; then echo "CUDA/JAX/MuJoCo";
  elif grep -Eiq "checkpoint|params_|keygraph|artifact|pickle" "$log"; then echo "checkpoint/artifact";
  elif grep -Eiq "ogbench|dataset|No registered env|Environment" "$log"; then echo "env/dataset";
  elif grep -Eiq "goal|task_id|reset" "$log"; then echo "goal protocol";
  else echo "unknown"; fi
}

eval "$(python - "$CONFIG" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
variants=list(c.get("variants", []))
print("ENVS_CONF="+",".join(c.get("envs", [])))
print("SEEDS_CONF="+",".join(map(str,c.get("seeds", [0]))))
print("EPISODES_CONF="+str(c.get("episodes",100)))
print("MAX_PARALLEL_CONF="+str(c.get("max_parallel_eval",4)))
print("VARIANTS_CONF="+",".join(variants))
print("FALLBACK_CONF="+",".join(c.get("fallback_modes", ["none"])))
print("MIN_EPISODES_CONF="+str(c.get("min_episodes_for_gate", c.get("episodes",100))))
PY
)"

ENVS="${ENVS:-$ENVS_CONF}"
SEEDS="${SEEDS:-$SEEDS_CONF}"
EPISODES="${EPISODES:-$EPISODES_CONF}"
VARIANTS="${VARIANTS:-$VARIANTS_CONF}"
FALLBACK_MODES="${FALLBACK_MODES:-$FALLBACK_CONF}"
MAX_PARALLEL_EVAL="${MAX_PARALLEL_EVAL:-$MAX_PARALLEL_CONF}"
ANALYZE_MIN_EPISODES="${ANALYZE_MIN_EPISODES:-$MIN_EPISODES_CONF}"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" ENVS="$ENVS" SEEDS="$SEEDS" GPUS="$GPUS" QUICK="$QUICK" PREFER_PRETRAINED="$PREFER_PRETRAINED" TRAIN_IF_MISSING="$TRAIN_IF_MISSING" LOG_ROOT="runs_stage25_prepare" bash scripts/stage22_prepare_gas_backbone.sh
fi
if [[ "$SKIP_REACHABILITY_TRAIN" != "1" ]]; then
  ENVS="$ENVS" SEEDS="$SEEDS" GPUS="$GPUS" QUICK="$QUICK" GAS_ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" ARTIFACT_ROOT="$ARTIFACT_ROOT" GAS_REPO_PATH="$GAS_REPO_PATH" PREFER_PRETRAINED="$PREFER_PRETRAINED" TRAIN_IF_MISSING="$TRAIN_IF_MISSING" LOG_ROOT="runs_stage25_reachability_train" bash scripts/stage22_train_reachability.sh
fi

jobs_tsv="$LOG_ROOT/jobs.tsv"
python - "$CONFIG" "$ENVS" "$SEEDS" "$EPISODES" "$VARIANTS" "$FALLBACK_MODES" > "$jobs_tsv" <<'PY'
import json, sys
from pathlib import Path
config, envs_s, seeds_s, episodes_s, variants_s, fallback_s = sys.argv[1:7]
c=json.load(open(config))
envs=[x for x in envs_s.split(",") if x]
seeds=[int(x) for x in seeds_s.split(",") if x]
episodes=int(episodes_s)
variants=[x for x in variants_s.split(",") if x]
fallback_modes=[x for x in fallback_s.split(",") if x]
rec_path=Path(c.get("budget_source","reports/stage22r_recommended_budgets.json"))
rec=json.load(open(rec_path)) if rec_path.exists() else {"envs":{}}
for env in envs:
  for seed in seeds:
    budget=rec.get("envs",{}).get(f"{env}/seed{seed}",{}).get("recommended_reachability_budget")
    if budget is None:
      for k,v in rec.get("envs",{}).items():
        if k.startswith(f"{env}/") and v.get("recommended_reachability_budget") is not None:
          budget=v["recommended_reachability_budget"]; break
    if budget is None:
      budget=999.0
    for variant in variants:
      for fallback in fallback_modes:
        if fallback == "none":
          print("\t".join(map(str,[env,seed,variant,budget,fallback,episodes])))
PY

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
idx=0
active=0
overall=0
while IFS=$'\t' read -r env seed variant budget fallback episodes; do
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  idx=$((idx + 1))
  run_dir="$LOG_ROOT/$env/seed$seed/$variant/budget$budget/fallback_$fallback"
  out_dir="$STAGE25_ROOT/$env/seed$seed/$variant/budget$budget/fallback_$fallback"
  mkdir -p "$run_dir" "$out_dir"
  log="$run_dir/evaluate.log"
  if [[ -f "$out_dir/eval.csv" ]]; then
    count="$(($(wc -l < "$out_dir/eval.csv") - 1))"
    if [[ "$count" -ge "$episodes" ]]; then
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"skipped\":true}" > "$run_dir/status.json"
      continue
    fi
  fi
  (
    echo "{\"status\":\"running\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"gpu\":\"$gpu\"}" > "$run_dir/status.json"
    python -m bars.gas_bars.evaluate \
      --env "$env" --seed "$seed" --task-ids all --episodes "$episodes" \
      --variant "$variant" --budget "$budget" \
      --gas-artifact-root "$GAS_ARTIFACT_ROOT" --artifact-root "$ARTIFACT_ROOT" --stage22-root "$STAGE25_ROOT" \
      --gas-repo-path "$GAS_REPO_PATH" --gpu "$gpu" --eval-on-cpu "$EVAL_ON_CPU" \
      --fallback-mode "$fallback" --prefer-pretrained "$PREFER_PRETRAINED" --train-if-missing "$TRAIN_IF_MISSING" --quick "$QUICK" \
      --max-steps "$MAX_STEPS" --max-plan-edges "$MAX_PLAN_EDGES" --debug-jsonl "$DEBUG_JSONL" \
      > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "{\"status\":\"completed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"gpu\":\"$gpu\"}" > "$run_dir/status.json"
    else
      err="$(classify_error "$log")"
      echo "$env,$seed,$variant,$budget,$fallback,$log,$err" >> "$FAILED"
      echo "{\"status\":\"failed\",\"env\":\"$env\",\"seed\":$seed,\"variant\":\"$variant\",\"gpu\":\"$gpu\",\"error_class\":\"$err\"}" > "$run_dir/status.json"
    fi
    exit "$rc"
  ) &
  active=$((active + 1))
  if [[ "$active" -ge "$MAX_PARALLEL_EVAL" ]]; then
    wait -n || overall=1
    active=$((active - 1))
  fi
done < "$jobs_tsv"

while [[ "$active" -gt 0 ]]; do
  wait -n || overall=1
  active=$((active - 1))
done

python scripts/stage25_enrich_failure_atlas_all_variants.py \
  --eval-roots "$STAGE25_ROOT" \
  --out "$REPORTS_ROOT/stage25_failure_atlas_all_variants.csv" \
  --summary-out "$REPORTS_ROOT/stage25_failure_atlas_summary.csv" \
  --integrity-out "$REPORTS_ROOT/stage25_label_integrity.json" \
  --min-episodes "$ANALYZE_MIN_EPISODES" || true

python scripts/stage25_analyze.py \
  --reports-root "$REPORTS_ROOT" \
  --local-drift-roots "$STAGE25_ROOT" \
  --failure-atlas "$REPORTS_ROOT/stage25_failure_atlas_all_variants.csv" \
  --min-episodes "$ANALYZE_MIN_EPISODES" || true

if [[ "$WAIT" == "1" ]]; then
  echo "[stage25_local_drift_v2] reports written under $REPORTS_ROOT; root=$STAGE25_ROOT"
fi
exit "$overall"
