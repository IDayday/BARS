#!/usr/bin/env bash
set -u
for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

CONFIG="${CONFIG:-configs/stage23_boundary_reentry.json}"
if ! python scripts/check_gate.py --gate GO_BOUNDARY_REENTRY --decision-report reports/stage22r_decisions.md; then
  mkdir -p reports
  cat > reports/stage23_boundary_status.md <<'EOF'
# Stage23 Boundary Status

HOLD_BOUNDARY

Boundary re-entry gate did not pass. Boundary eval was not launched.
EOF
  exit 0
fi

echo "[stage23_boundary] Gate passed; launching boundary re-entry through key-claim runner."
CONFIG="$CONFIG" STAGE23_ROOT="${STAGE23_ROOT:-runs_stage23_boundary}" LOG_ROOT="${LOG_ROOT:-runs_stage23_boundary_logs}" bash scripts/stage23_run_key_claim.sh
