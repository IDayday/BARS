#!/usr/bin/env bash
set -euo pipefail
LOG_ROOT="${LOG_ROOT:-runs}"
GPUS="${GPUS:-auto}"
python -m bars.sched.jobctl status --log-root "${LOG_ROOT}" --gpus "${GPUS}"
