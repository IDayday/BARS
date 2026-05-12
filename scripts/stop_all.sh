#!/usr/bin/env bash
set -euo pipefail
LOG_ROOT="${LOG_ROOT:-runs}"
python -m bars.sched.jobctl stop --log-root "${LOG_ROOT}" --all
