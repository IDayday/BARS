#!/usr/bin/env bash
set -euo pipefail

# Round004 GAS self-training launcher.
# Runs full-budget GAS from scratch without official checkpoints:
#   pretrain_tdr.py 1M -> construct_graph.py -> train_policy.py 1M -> evaluate_gas.py.

ENVS="${ENVS:-antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0}"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS:-0,1,2}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
TE_THRESHOLD="${TE_THRESHOLD:-0.99}"
OUT_ROOT="${OUT_ROOT:-artifacts/gas_selftrain_round004}"
RUN_ROOT="${RUN_ROOT:-runs_round004_gas_selftrain}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
ROUND="${ROUND:-004}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-}"
MODE="${MODE:-launch}"
ONE_ENV="${ONE_ENV:-}"
ONE_SEED="${ONE_SEED:-}"
ONE_GPU="${ONE_GPU:-}"
SCRIPT_PATH="$(readlink -f "$0")"

for arg in "$@"; do
  case "$arg" in
    ENVS=*) ENVS="${arg#ENVS=}" ;;
    SEEDS=*) SEEDS="${arg#SEEDS=}" ;;
    GPUS=*) GPUS="${arg#GPUS=}" ;;
    TRAIN_STEPS=*) TRAIN_STEPS="${arg#TRAIN_STEPS=}" ;;
    LOG_INTERVAL=*) LOG_INTERVAL="${arg#LOG_INTERVAL=}" ;;
    SAVE_INTERVAL=*) SAVE_INTERVAL="${arg#SAVE_INTERVAL=}" ;;
    TE_THRESHOLD=*) TE_THRESHOLD="${arg#TE_THRESHOLD=}" ;;
    OUT_ROOT=*) OUT_ROOT="${arg#OUT_ROOT=}" ;;
    RUN_ROOT=*) RUN_ROOT="${arg#RUN_ROOT=}" ;;
    GAS_REPO_PATH=*) GAS_REPO_PATH="${arg#GAS_REPO_PATH=}" ;;
    ROUND=*) ROUND="${arg#ROUND=}" ;;
    WAIT_FOR_GPU_FREE=*) WAIT_FOR_GPU_FREE="${arg#WAIT_FOR_GPU_FREE=}" ;;
    MIN_FREE_MB=*) MIN_FREE_MB="${arg#MIN_FREE_MB=}" ;;
    MAX_GPU_UTIL=*) MAX_GPU_UTIL="${arg#MAX_GPU_UTIL=}" ;;
    POLL_SECONDS=*) POLL_SECONDS="${arg#POLL_SECONDS=}" ;;
    FORCE=*) FORCE="${arg#FORCE=}" ;;
    PYTHON=*) PYTHON="${arg#PYTHON=}" ;;
  esac
done

if [[ -f scripts/stage24_env_mirrors.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/stage24_env_mirrors.sh
fi

if [[ -z "$PYTHON" ]]; then
  if [[ -x /root/anaconda3/envs/gcrlo/bin/python ]]; then
    PYTHON=/root/anaconda3/envs/gcrlo/bin/python
  else
    PYTHON=python
  fi
fi

export PYTHONPATH="${PYTHONPATH:-$PWD}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export D4RL_SUPPRESS_IMPORT_ERROR=1
export BARS_OGBENCH_OFFLINE="${BARS_OGBENCH_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/root/remote/datasets/ogbench}"

mkdir -p "$RUN_ROOT" "$OUT_ROOT" reports commands rounds/round_"$ROUND"

IFS=',' read -r -a ENV_LIST <<< "$ENVS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs specified" >&2
  exit 2
fi

slug_for_env() {
  local env="$1"
  echo "${env%-v0}"
}

way_steps_for_env() {
  local env="$1"
  if [[ "$env" == *scene* || "$env" == *kitchen* ]]; then
    echo 48
  else
    echo 8
  fi
}

discount_for_env() {
  local env="$1"
  if [[ "$env" == *giant* ]]; then
    echo 0.995
  else
    echo 0.99
  fi
}

alpha_for_env() {
  local env="$1"
  if [[ "$env" == *explore* ]]; then
    echo 0.01
  elif [[ "$env" == *kitchen* ]]; then
    echo 10.0
  else
    echo 1.0
  fi
}

expectile_for_env() {
  local env="$1"
  if [[ "$env" == *kitchen* ]]; then
    echo 0.95
  else
    echo 0.999
  fi
}

checkpoint_path() {
  local root="$1"
  find "$root" -type f -name "params_${TRAIN_STEPS}.pkl" 2>/dev/null | sort | tail -n 1
}

latest_checkpoint_path() {
  local root="$1"
  find "$root" -type f -name "params_*.pkl" 2>/dev/null |
    sed -E 's#(.*/params_)([0-9]+)(\.pkl)$#\2 & #' |
    sort -n |
    tail -n 1 |
    cut -d' ' -f2-
}

keygraph_path() {
  local root="$1"
  find "$root" -type f -name "keygraph.pkl" 2>/dev/null | sort | tail -n 1
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

wait_for_gpu() {
  local gpu="$1"
  local log="$2"
  if [[ "$WAIT_FOR_GPU_FREE" != "1" ]]; then
    return 0
  fi
  while true; do
    local free util
    free="$(gpu_free_mb "$gpu" || echo 0)"
    util="$(gpu_util "$gpu" || echo 100)"
    free="${free:-0}"
    util="${util:-100}"
    if [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$free" -ge "$MIN_FREE_MB" && "$util" -le "$MAX_GPU_UTIL" ]]; then
      echo "$(date -Is) gpu=$gpu free_mb=$free util=$util -> start" >> "$log"
      return 0
    fi
    echo "$(date -Is) waiting gpu=$gpu free_mb=$free/${MIN_FREE_MB} util=$util/${MAX_GPU_UTIL}" >> "$log"
    sleep "$POLL_SECONDS"
  done
}

run_phase() {
  local phase="$1"
  local log="$2"
  shift 2
  echo "$(date -Is) START $phase" >> "$log"
  echo "$*" >> "$log"
  "$@" >> "$log" 2>&1
  echo "$(date -Is) DONE $phase" >> "$log"
}

run_env_seed() {
  local env="$1"
  local seed="$2"
  local gpu="$3"
  local slug run_group env_out run_dir tdr_root graph_root policy_root eval_root status pipeline_log
  slug="$(slug_for_env "$env")"
  run_group="round${ROUND}_selftrain_${slug}_seed${seed}"
  env_out="$OUT_ROOT/$env/seed${seed}"
  run_dir="$RUN_ROOT/$env/seed${seed}"
  tdr_root="$env_out/tdr"
  graph_root="$env_out/graph"
  policy_root="$env_out/policy"
  eval_root="$run_dir/evaluate_gas"
  status="$run_dir/status.json"
  pipeline_log="$run_dir/pipeline.log"
  mkdir -p "$run_dir" "$tdr_root" "$graph_root" "$policy_root" "$eval_root"

  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$tdr_root" "$graph_root" "$policy_root" "$eval_root"
    mkdir -p "$tdr_root" "$graph_root" "$policy_root" "$eval_root"
  fi

  {
    echo "{"
    echo "  \"env\": \"$env\","
    echo "  \"seed\": $seed,"
    echo "  \"gpu\": \"$gpu\","
    echo "  \"train_steps\": $TRAIN_STEPS,"
    echo "  \"artifact_source\": \"full_budget_train_from_scratch\","
    echo "  \"official_weights_used\": false,"
    echo "  \"status\": \"launched\","
    echo "  \"started_at\": \"$(date -Is)\""
    echo "}"
  } > "$status"

  local flags
  flags=(
    --env_name "$env"
    --seed "$seed"
    --gpu "$gpu"
    --agent_config.encoder not_used
    --agent_config.discount "$(discount_for_env "$env")"
    --agent_config.tdr_expectile "$(expectile_for_env "$env")"
    --agent_config.alpha "$(alpha_for_env "$env")"
    --agent_config.batch_size 1024
    --agent_config.p_aug 0.0
    --agent_config.way_steps "$(way_steps_for_env "$env")"
  )

  wait_for_gpu "$gpu" "$pipeline_log"
  cd "$GAS_REPO_PATH"

  local tdr_path graph_path policy_path resume_tdr_path resume_policy_path
  tdr_path="$(checkpoint_path "../../$tdr_root" || true)"
  if [[ -z "$tdr_path" ]]; then
    resume_tdr_path="$(latest_checkpoint_path "../../$tdr_root" || true)"
    resume_args=()
    if [[ -n "$resume_tdr_path" ]]; then
      resume_args=(--resume_tdr_path "$resume_tdr_path")
    fi
    run_phase pretrain_tdr "../../$run_dir/pretrain_tdr.log" \
      "$PYTHON" pretrain_tdr.py \
      --run_tdr_project "Round${ROUND}_GAS_SelfTrain" \
      --run_group "$run_group" \
      --save_tdr_dir "../../$tdr_root" \
      --train_steps "$TRAIN_STEPS" \
      --log_interval "$LOG_INTERVAL" \
      --save_interval "$SAVE_INTERVAL" \
      "${resume_args[@]}" \
      "${flags[@]}"
    tdr_path="$(checkpoint_path "../../$tdr_root")"
  fi

  graph_path="$(keygraph_path "../../$graph_root" || true)"
  if [[ -z "$graph_path" ]]; then
    run_phase construct_graph "../../$run_dir/construct_graph.log" \
      "$PYTHON" construct_graph.py \
      --run_group "$run_group" \
      --save_graph_dir "../../$graph_root" \
      --te_threshold "$TE_THRESHOLD" \
      --tdr_path "$tdr_path" \
      "${flags[@]}"
    graph_path="$(keygraph_path "../../$graph_root")"
  fi

  policy_path="$(checkpoint_path "../../$policy_root" || true)"
  if [[ -z "$policy_path" ]]; then
    resume_policy_path="$(latest_checkpoint_path "../../$policy_root" || true)"
    resume_args=()
    if [[ -n "$resume_policy_path" ]]; then
      resume_args=(--resume_policy_path "$resume_policy_path")
    fi
    run_phase train_policy "../../$run_dir/train_policy.log" \
      "$PYTHON" train_policy.py \
      --run_policy_project "Round${ROUND}_GAS_SelfTrain" \
      --run_group "$run_group" \
      --save_policy_dir "../../$policy_root" \
      --train_steps "$TRAIN_STEPS" \
      --log_interval "$LOG_INTERVAL" \
      --save_interval "$SAVE_INTERVAL" \
      --tdr_path "$tdr_path" \
      "${resume_args[@]}" \
      "${flags[@]}"
    policy_path="$(checkpoint_path "../../$policy_root")"
  fi

  run_phase evaluate_gas "../../$run_dir/evaluate_gas.log" \
    "$PYTHON" evaluate_gas.py \
    --run_eval_project "Round${ROUND}_GAS_SelfTrain_Eval" \
    --run_group "$run_group" \
    --save_eval_dir "../../$eval_root" \
    --eval_on_cpu 1 \
    --eval_episodes 49 \
    --eval_video_episodes 1 \
    --eval_final_goal_threshold 2 \
    --keygraph_path "$graph_path" \
    --policy_path "$policy_path" \
    "${flags[@]}"

  cd - >/dev/null
  {
    echo "{"
    echo "  \"env\": \"$env\","
    echo "  \"seed\": $seed,"
    echo "  \"gpu\": \"$gpu\","
    echo "  \"train_steps\": $TRAIN_STEPS,"
    echo "  \"artifact_source\": \"full_budget_train_from_scratch\","
    echo "  \"official_weights_used\": false,"
    echo "  \"tdr_checkpoint\": \"$tdr_path\","
    echo "  \"graph_checkpoint\": \"$graph_path\","
    echo "  \"policy_checkpoint\": \"$policy_path\","
    echo "  \"status\": \"completed\","
    echo "  \"completed_at\": \"$(date -Is)\""
    echo "}"
  } > "$status"
}

if [[ "$MODE" == "run_one" ]]; then
  if [[ -z "$ONE_ENV" || -z "$ONE_SEED" || -z "$ONE_GPU" ]]; then
    echo "MODE=run_one requires ONE_ENV, ONE_SEED, and ONE_GPU" >&2
    exit 2
  fi
  set +e
  run_env_seed "$ONE_ENV" "$ONE_SEED" "$ONE_GPU"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    run_dir="$RUN_ROOT/$ONE_ENV/seed${ONE_SEED}"
    mkdir -p "$run_dir"
    echo "{\"env\":\"$ONE_ENV\",\"seed\":$ONE_SEED,\"gpu\":\"$ONE_GPU\",\"status\":\"failed\",\"returncode\":$rc,\"failed_at\":\"$(date -Is)\"}" > "$run_dir/status.json"
  fi
  exit $rc
fi

jobs_tsv="$RUN_ROOT/jobs.tsv"
echo -e "env\tseed\tgpu\tpid\tstatus\tlog" > "$jobs_tsv"
idx=0
for env in "${ENV_LIST[@]}"; do
  env="${env//[[:space:]]/}"
  [[ -z "$env" ]] && continue
  for seed in "${SEED_LIST[@]}"; do
    seed="${seed//[[:space:]]/}"
    gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
    run_dir="$RUN_ROOT/$env/seed${seed}"
    mkdir -p "$run_dir"
    setsid env \
      MODE=run_one \
      ONE_ENV="$env" \
      ONE_SEED="$seed" \
      ONE_GPU="$gpu" \
      ENVS="$ENVS" \
      SEEDS="$SEEDS" \
      GPUS="$GPUS" \
      TRAIN_STEPS="$TRAIN_STEPS" \
      LOG_INTERVAL="$LOG_INTERVAL" \
      SAVE_INTERVAL="$SAVE_INTERVAL" \
      TE_THRESHOLD="$TE_THRESHOLD" \
      OUT_ROOT="$OUT_ROOT" \
      RUN_ROOT="$RUN_ROOT" \
      GAS_REPO_PATH="$GAS_REPO_PATH" \
      ROUND="$ROUND" \
      WAIT_FOR_GPU_FREE="$WAIT_FOR_GPU_FREE" \
      MIN_FREE_MB="$MIN_FREE_MB" \
      MAX_GPU_UTIL="$MAX_GPU_UTIL" \
      POLL_SECONDS="$POLL_SECONDS" \
      FORCE="$FORCE" \
      PYTHON="$PYTHON" \
      BARS_OGBENCH_OFFLINE="$BARS_OGBENCH_OFFLINE" \
      OGBENCH_DATASET_DIR="$OGBENCH_DATASET_DIR" \
      MUJOCO_GL="$MUJOCO_GL" \
      PYTHONPATH="$PYTHONPATH" \
      bash "$SCRIPT_PATH" < /dev/null >> "$run_dir/launcher.log" 2>&1 &
    pid=$!
    echo -e "$env\t$seed\t$gpu\t$pid\tlaunched\t$run_dir/launcher.log" >> "$jobs_tsv"
    idx=$((idx + 1))
  done
done

cp "$jobs_tsv" "rounds/round_${ROUND}/gas_selftrain_jobs.tsv"
cp "$jobs_tsv" "reports/round_${ROUND}_gas_selftrain_jobs.tsv"
cat > "reports/round_${ROUND}_gas_selftrain_launch.md" <<EOF
# Round ${ROUND} GAS Self-Training Launch

- Evidence class while running: E4_FULL_BUDGET_TRAINED_METHOD pending completion.
- Official weights used: false.
- Train steps: ${TRAIN_STEPS} for TDR and policy.
- Dataset root: ${OGBENCH_DATASET_DIR}
- Artifact root: ${OUT_ROOT}
- Run root: ${RUN_ROOT}
- GPU wait: WAIT_FOR_GPU_FREE=${WAIT_FOR_GPU_FREE}, MIN_FREE_MB=${MIN_FREE_MB}, MAX_GPU_UTIL=${MAX_GPU_UTIL}
- Jobs: ${jobs_tsv}

No p_bridge, integrated BARS-v3, oracle-headroom, or failure taxonomy interpretation is run by this launcher.
EOF

echo "$jobs_tsv"
