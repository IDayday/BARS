#!/usr/bin/env bash
set -Eeuo pipefail
export D4RL_SUPPRESS_IMPORT_ERROR=1
SWEEP="${SWEEP:-configs/sweeps/d4rl_stage16_sanity4.json}"
LOG_ROOT="${LOG_ROOT:-runs_stage16_sanity4}"
GPUS="${GPUS:-0,1,2,3}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-4}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LAUNCH_BURST="${LAUNCH_BURST:-4}"
python -m bars.sched.jobctl launch \
  --sweep "$SWEEP" \
  --log-root "$LOG_ROOT" \
  --gpus "$GPUS" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds "$POLL_SECONDS" \
  --launch-burst "$LAUNCH_BURST" \
  --background
