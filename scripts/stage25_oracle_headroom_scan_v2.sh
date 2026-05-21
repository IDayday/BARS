#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

CONFIG="${CONFIG:-configs/stage25_oracle_scan_matrix.json}"
ENVS="${ENVS:-}"
SEEDS="${SEEDS:-}"
GRAPH_IDS="${GRAPH_IDS:-G3}"
GPUS="${GPUS:-0}"
EDGE_EXEC_PILOT="${EDGE_EXEC_PILOT:-1}"
TOP_K_BRIDGE="${TOP_K_BRIDGE:-}"
MAX_SOURCES="${MAX_SOURCES:-}"
GAS_ARTIFACT_ROOT="${GAS_ARTIFACT_ROOT:-artifacts/gas}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/stage25}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
ORACLE_REPORTS_ROOT="${ORACLE_REPORTS_ROOT:-reports/stage25_oracle_scan_tmp}"
REPORTS_ROOT="${REPORTS_ROOT:-reports}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
PREFER_PRETRAINED="${PREFER_PRETRAINED:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-0}"
QUICK="${QUICK:-0}"
WAIT="${WAIT:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ -f scripts/stage24_env_mirrors.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/stage24_env_mirrors.sh
fi

mkdir -p "$ORACLE_REPORTS_ROOT" "$ARTIFACT_ROOT" "$REPORTS_ROOT"
FAILED="$ORACLE_REPORTS_ROOT/failed_jobs.csv"
if [[ ! -f "$FAILED" ]]; then
  echo "env,seed,stage,graph_id,log,error_class" > "$FAILED"
fi

eval "$(python - "$CONFIG" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
envs=[]
for xs in c.get("env_groups",{}).values():
    envs.extend(xs)
print("ENVS_CONF="+",".join(envs))
print("SEEDS_CONF="+",".join(map(str,c.get("seeds",[0]))))
print("TOP_K_CONF="+str(c.get("top_k_bridge",4)))
print("MAX_SOURCES_CONF="+str(c.get("max_sources",200)))
PY
)"
ENVS="${ENVS:-$ENVS_CONF}"
SEEDS="${SEEDS:-$SEEDS_CONF}"
TOP_K_BRIDGE="${TOP_K_BRIDGE:-$TOP_K_CONF}"
MAX_SOURCES="${MAX_SOURCES:-$MAX_SOURCES_CONF}"

classify_error() {
  local log="$1"
  if grep -Eiq "No registered env|Environment|ogbench|dataset" "$log"; then echo "env/dataset";
  elif grep -Eiq "checkpoint|params_|keygraph|artifact|pickle|Missing GAS keygraph|missing keygraph" "$log"; then echo "checkpoint/artifact";
  elif grep -Eiq "ModuleNotFoundError|ImportError|No module named" "$log"; then echo "dependency/import";
  elif grep -Eiq "CUDA|jaxlib|XLA|MuJoCo|mujoco|EGL|GLFW" "$log"; then echo "CUDA/JAX/MuJoCo";
  else echo "unknown"; fi
}

IFS=',' read -r -a ENV_ARR <<< "$ENVS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GRAPH_ARR <<< "$GRAPH_IDS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
overall=0
idx=0

for env in "${ENV_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    idx=$((idx + 1))
    prep_log="$ORACLE_REPORTS_ROOT/${env}_seed${seed}_prepare.log"
    if [[ "$SKIP_PREPARE" != "1" ]]; then
      ARTIFACT_ROOT="$GAS_ARTIFACT_ROOT" ENVS="$env" SEEDS="$seed" GPUS="$gpu" QUICK="$QUICK" PREFER_PRETRAINED="$PREFER_PRETRAINED" TRAIN_IF_MISSING="$TRAIN_IF_MISSING" LOG_ROOT="runs_stage25_prepare_oracle" bash scripts/stage22_prepare_gas_backbone.sh > "$prep_log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        err="$(classify_error "$prep_log")"
        echo "$env,$seed,prepare,,$prep_log,$err" >> "$FAILED"
        overall=1
        continue
      fi
    fi
    build_log="$ORACLE_REPORTS_ROOT/${env}_seed${seed}_build.log"
    python scripts/stage23_build_bridge_graphs.py \
      --envs "$env" --seeds "$seed" \
      --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
      --artifact-root "$ARTIFACT_ROOT" \
      --reports-root "$ORACLE_REPORTS_ROOT" \
      --top-k-bridge "$TOP_K_BRIDGE" \
      --max-sources "$MAX_SOURCES" \
      > "$build_log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      err="$(classify_error "$build_log")"
      echo "$env,$seed,build,,${build_log},$err" >> "$FAILED"
      overall=1
      continue
    fi
    for graph_id in "${GRAPH_ARR[@]}"; do
      if [[ "$graph_id" == "G0" ]]; then
        continue
      fi
      edge_log="$ORACLE_REPORTS_ROOT/${env}_seed${seed}_${graph_id}_edge.log"
      python scripts/stage23_edge_execution.py \
        --envs "$env" --seeds "$seed" --graph-id "$graph_id" \
        --gas-artifact-root "$GAS_ARTIFACT_ROOT" \
        --artifact-root "$ARTIFACT_ROOT" \
        --gas-repo-path "$GAS_REPO_PATH" \
        --reports-root "$ORACLE_REPORTS_ROOT" \
        --gpu "$gpu" \
        --pilot "$EDGE_EXEC_PILOT" \
        --prefer-pretrained "$PREFER_PRETRAINED" \
        --train-if-missing "$TRAIN_IF_MISSING" \
        > "$edge_log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        err="$(classify_error "$edge_log")"
        echo "$env,$seed,edge_execution,$graph_id,$edge_log,$err" >> "$FAILED"
        overall=1
        continue
      fi
      oracle_log="$ORACLE_REPORTS_ROOT/${env}_seed${seed}_${graph_id}_oracle.log"
      python scripts/stage23_oracle_bridge.py \
        --envs "$env" --seeds "$seed" --graph-id "$graph_id" \
        --artifact-root "$ARTIFACT_ROOT" \
        --reports-root "$ORACLE_REPORTS_ROOT" \
        > "$oracle_log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        err="$(classify_error "$oracle_log")"
        echo "$env,$seed,oracle_bridge,$graph_id,$oracle_log,$err" >> "$FAILED"
        overall=1
      fi
    done
  done
done

python scripts/stage25_rank_oracle_envs.py \
  --reports-root "$ORACLE_REPORTS_ROOT" \
  --out "$ORACLE_REPORTS_ROOT/stage25_oracle_env_ranking.csv" \
  --out-md "$ORACLE_REPORTS_ROOT/stage25_oracle_env_ranking.md" || overall=1

python scripts/stage25_analyze.py \
  --reports-root "$REPORTS_ROOT" \
  --oracle-reports-root "$ORACLE_REPORTS_ROOT" \
  --oracle-artifact-root "$ARTIFACT_ROOT" || true

if [[ "$WAIT" == "1" ]]; then
  echo "[stage25_oracle] reports written under $REPORTS_ROOT and $ORACLE_REPORTS_ROOT"
fi
exit "$overall"
