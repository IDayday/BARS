#!/usr/bin/env bash
set -euo pipefail

# Round005 GAS self-training queue launcher.
# Runs 3 envs x 3 seeds from scratch, with at most one internal job per GPU.

ROUND="${ROUND:-005}"
ENVS="${ENVS:-antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0}"
SEEDS="${SEEDS:-0,1,2}"
GPUS="${GPUS:-0,1,2,3,4,5}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
TE_THRESHOLD="${TE_THRESHOLD:-0.99}"
OUT_ROOT="${OUT_ROOT:-artifacts/gas_selftrain_round005}"
RUN_ROOT="${RUN_ROOT:-runs_round005_gas_selftrain}"
GAS_REPO_PATH="${GAS_REPO_PATH:-external_src/GAS}"
LAUNCHER="${LAUNCHER:-scripts/round004_train_gas_from_scratch_multigpu.sh}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FORCE="${FORCE:-1}"
PYTHON="${PYTHON:-}"
MODE="${MODE:-launch}"
WORKER_INDEX="${WORKER_INDEX:-}"
WORKER_GPU="${WORKER_GPU:-}"
SCRIPT_PATH="$(readlink -f "$0")"

for arg in "$@"; do
  case "$arg" in
    ROUND=*) ROUND="${arg#ROUND=}" ;;
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
    LAUNCHER=*) LAUNCHER="${arg#LAUNCHER=}" ;;
    WAIT_FOR_GPU_FREE=*) WAIT_FOR_GPU_FREE="${arg#WAIT_FOR_GPU_FREE=}" ;;
    MIN_FREE_MB=*) MIN_FREE_MB="${arg#MIN_FREE_MB=}" ;;
    MAX_GPU_UTIL=*) MAX_GPU_UTIL="${arg#MAX_GPU_UTIL=}" ;;
    POLL_SECONDS=*) POLL_SECONDS="${arg#POLL_SECONDS=}" ;;
    FORCE=*) FORCE="${arg#FORCE=}" ;;
    PYTHON=*) PYTHON="${arg#PYTHON=}" ;;
  esac
done

IFS=',' read -r -a ENV_LIST <<< "$ENVS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs specified" >&2
  exit 2
fi

if [[ -z "$PYTHON" ]]; then
  if [[ -x /root/anaconda3/envs/gcrlo/bin/python ]]; then
    PYTHON=/root/anaconda3/envs/gcrlo/bin/python
  else
    PYTHON=python
  fi
fi

mkdir -p "$RUN_ROOT" "$OUT_ROOT" reports commands rounds/round_"$ROUND" "$RUN_ROOT/_workers"

build_jobs() {
  JOB_ENVS=()
  JOB_SEEDS=()
  local env seed
  for env in "${ENV_LIST[@]}"; do
    env="${env//[[:space:]]/}"
    [[ -z "$env" ]] && continue
    for seed in "${SEED_LIST[@]}"; do
      seed="${seed//[[:space:]]/}"
      [[ -z "$seed" ]] && continue
      JOB_ENVS+=("$env")
      JOB_SEEDS+=("$seed")
    done
  done
}

run_worker() {
  if [[ -z "$WORKER_INDEX" || -z "$WORKER_GPU" ]]; then
    echo "MODE=worker requires WORKER_INDEX and WORKER_GPU" >&2
    exit 2
  fi
  build_jobs
  local idx env seed run_dir job_log rc
  for idx in "${!JOB_ENVS[@]}"; do
    if (( idx % ${#GPU_LIST[@]} != WORKER_INDEX )); then
      continue
    fi
    env="${JOB_ENVS[$idx]}"
    seed="${JOB_SEEDS[$idx]}"
    run_dir="$RUN_ROOT/$env/seed${seed}"
    job_log="$run_dir/launcher.log"
    mkdir -p "$run_dir"
    {
      echo "$(date -Is) ROUND=$ROUND job_index=$idx env=$env seed=$seed gpu=$WORKER_GPU START"
      echo "FORCE=$FORCE WAIT_FOR_GPU_FREE=$WAIT_FOR_GPU_FREE OUT_ROOT=$OUT_ROOT RUN_ROOT=$RUN_ROOT"
    } >> "$job_log"
    set +e
    env \
      MODE=run_one \
      ONE_ENV="$env" \
      ONE_SEED="$seed" \
      ONE_GPU="$WORKER_GPU" \
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
      BARS_OGBENCH_OFFLINE="${BARS_OGBENCH_OFFLINE:-1}" \
      OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/root/remote/datasets/ogbench}" \
      MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
      PYTHONPATH="${PYTHONPATH:-$PWD}" \
      bash "$LAUNCHER" >> "$job_log" 2>&1
    rc=$?
    set -e
    echo "$(date -Is) ROUND=$ROUND job_index=$idx env=$env seed=$seed gpu=$WORKER_GPU DONE rc=$rc" >> "$job_log"
  done
}

write_assignment() {
  build_jobs
  local jobs_tsv="$RUN_ROOT/jobs.tsv"
  echo -e "job_index\tenv\tseed\tgpu\tworker_index\tstatus\tlog" > "$jobs_tsv"
  local idx env seed worker_idx gpu run_dir
  for idx in "${!JOB_ENVS[@]}"; do
    env="${JOB_ENVS[$idx]}"
    seed="${JOB_SEEDS[$idx]}"
    worker_idx=$((idx % ${#GPU_LIST[@]}))
    gpu="${GPU_LIST[$worker_idx]}"
    run_dir="$RUN_ROOT/$env/seed${seed}"
    echo -e "$idx\t$env\t$seed\t$gpu\t$worker_idx\tassigned\t$run_dir/launcher.log" >> "$jobs_tsv"
  done
  cp "$jobs_tsv" "rounds/round_${ROUND}/gas_selftrain_3seed_jobs.tsv"
  cp "$jobs_tsv" "reports/round_${ROUND}_gas_selftrain_3seed_jobs.tsv"
}

launch_workers() {
  write_assignment
  local worker_tsv="$RUN_ROOT/workers.tsv"
  echo -e "worker_index\tgpu\tpid\tlog" > "$worker_tsv"
  local i gpu worker_log pid
  for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    worker_log="$RUN_ROOT/_workers/gpu${gpu}.log"
    setsid env \
      MODE=worker \
      WORKER_INDEX="$i" \
      WORKER_GPU="$gpu" \
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
      LAUNCHER="$LAUNCHER" \
      ROUND="$ROUND" \
      WAIT_FOR_GPU_FREE="$WAIT_FOR_GPU_FREE" \
      MIN_FREE_MB="$MIN_FREE_MB" \
      MAX_GPU_UTIL="$MAX_GPU_UTIL" \
      POLL_SECONDS="$POLL_SECONDS" \
      FORCE="$FORCE" \
      PYTHON="$PYTHON" \
      BARS_OGBENCH_OFFLINE="${BARS_OGBENCH_OFFLINE:-1}" \
      OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/root/remote/datasets/ogbench}" \
      MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
      PYTHONPATH="${PYTHONPATH:-$PWD}" \
      bash "$SCRIPT_PATH" < /dev/null >> "$worker_log" 2>&1 &
    pid=$!
    echo -e "$i\t$gpu\t$pid\t$worker_log" >> "$worker_tsv"
  done
  cp "$worker_tsv" "rounds/round_${ROUND}/gas_selftrain_3seed_workers.tsv"
  cp "$worker_tsv" "reports/round_${ROUND}_gas_selftrain_3seed_workers.tsv"

  cat > "commands/round_${ROUND}_gas_selftrain_3seed_launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash scripts/round005_train_gas_from_scratch_3seed_queue.sh \\
  ROUND=$ROUND \\
  ENVS=$ENVS \\
  SEEDS=$SEEDS \\
  GPUS=$GPUS \\
  TRAIN_STEPS=$TRAIN_STEPS \\
  LOG_INTERVAL=$LOG_INTERVAL \\
  SAVE_INTERVAL=$SAVE_INTERVAL \\
  TE_THRESHOLD=$TE_THRESHOLD \\
  OUT_ROOT=$OUT_ROOT \\
  RUN_ROOT=$RUN_ROOT \\
  GAS_REPO_PATH=$GAS_REPO_PATH \\
  WAIT_FOR_GPU_FREE=$WAIT_FOR_GPU_FREE \\
  FORCE=$FORCE
EOF
  chmod +x "commands/round_${ROUND}_gas_selftrain_3seed_launch.sh"

  cat > "reports/round_${ROUND}_gas_selftrain_3seed_launch.md" <<EOF
# Round ${ROUND} GAS 3-Seed Self-Training Launch

- Evidence class while running: E4_FULL_BUDGET_TRAINED_METHOD pending completion.
- Official weights used: false.
- From scratch: FORCE=${FORCE}.
- GPU wait: WAIT_FOR_GPU_FREE=${WAIT_FOR_GPU_FREE}.
- Environments: ${ENVS}.
- Seeds: ${SEEDS}.
- GPUs: ${GPUS}.
- Train steps: ${TRAIN_STEPS} for TDR and policy.
- Dataset root: ${OGBENCH_DATASET_DIR:-/root/remote/datasets/ogbench}.
- Artifact root: ${OUT_ROOT}.
- Run root: ${RUN_ROOT}.
- Job assignment: reports/round_${ROUND}_gas_selftrain_3seed_jobs.tsv.
- Worker table: reports/round_${ROUND}_gas_selftrain_3seed_workers.tsv.
- Launch command: commands/round_${ROUND}_gas_selftrain_3seed_launch.sh.

Scheduling note: at most one Round ${ROUND} internal job is assigned to each GPU at a time; jobs beyond the GPU count are queued serially on their assigned worker GPU. External GPU occupancy is not waited on.

No p_bridge, integrated BARS-v3, oracle-headroom, or failure taxonomy interpretation is run by this launcher.
EOF
}

case "$MODE" in
  launch)
    launch_workers
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
