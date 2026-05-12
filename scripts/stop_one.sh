#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "Usage: bash scripts/stop_one.sh <run_id>"; exit 1; fi
LOG_ROOT="${LOG_ROOT:-runs}"
python -m bars.sched.jobctl stop --log-root "${LOG_ROOT}" --run-id "$1"
