#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${1:-antmaze-medium-play-v2}"
SEED="${2:-0}"
VARIANT="${3:-full_bars}"
NODE_METHOD="${4:-bars}"
CONFIG="${CONFIG:-configs/d4rl_antmaze_quick.json}"
LOG_ROOT="${LOG_ROOT:-runs}"
python -m bars.cli run --config "${CONFIG}" --log-root "${LOG_ROOT}" --env "${ENV_NAME}" --seed "${SEED}" --variant "${VARIANT}" --node-method "${NODE_METHOD}"
