#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ENV_NAME="${ENV_NAME:-antmaze-medium-stitch-v0}"
FOUND="$(find artifacts -path "*tmd*${ENV_NAME}*" -name 'params_*.pkl' | sort -V | tail -n 1 || true)"
if [[ -n "$FOUND" ]]; then
  echo "$FOUND"
  exit 0
fi
cat >&2 <<MSG
No existing TMD checkpoint found for ${ENV_NAME}.
Train a quick checkpoint with official tmd-release main.py, or pass --tmd-checkpoint explicitly.
MSG
exit 2
