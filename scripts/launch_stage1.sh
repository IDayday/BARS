#!/usr/bin/env bash
set -euo pipefail
LOG_ROOT="${LOG_ROOT:-runs}"
GPUS="${GPUS:-auto}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
SWEEP="${SWEEP:-configs/sweeps/d4rl_stage1.json}"
python -m bars.sched.jobctl launch --sweep "${SWEEP}" --log-root "${LOG_ROOT}" --gpus "${GPUS}" --max-jobs-per-gpu "${MAX_JOBS_PER_GPU}" --poll-seconds 10 --launch-burst 1 --background
