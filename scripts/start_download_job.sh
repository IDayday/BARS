#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <job-name> <command> [args...]" >&2
  exit 2
fi

job_name="$1"
shift

DATA_ROOT="${DATA_ROOT:-/mnt/project/offlinerl_datasets}"
LOG_DIR="${LOG_DIR:-$DATA_ROOT/logs}"
RUN_DIR="${RUN_DIR:-$DATA_ROOT/run}"
RETRY_WAIT_SECONDS="${RETRY_WAIT_SECONDS:-300}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

log="$LOG_DIR/${job_name}.log"
pidfile="$RUN_DIR/${job_name}.pid"
lockfile="$RUN_DIR/${job_name}.lock"

if [[ -s "$pidfile" ]]; then
  old_pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Job already running: $job_name"
    echo "PID: $old_pid"
    echo "Log: $log"
    exit 0
  fi
fi

cmd_q=""
for arg in "$@"; do
  printf -v quoted '%q' "$arg"
  cmd_q+="$quoted "
done

setsid nohup bash -c "
  exec 9>'$lockfile'
  if ! flock -n 9; then
    echo \"[\$(date -Is)] Another $job_name worker is already running.\" >> '$log'
    exit 0
  fi
  echo \$\$ > '$pidfile'
  trap 'rm -f \"$pidfile\"' EXIT
  while true; do
    echo \"[\$(date -Is)] starting: $cmd_q\" >> '$log'
    $cmd_q >> '$log' 2>&1
    rc=\$?
    if [ \$rc -eq 0 ]; then
      echo \"[\$(date -Is)] completed successfully\" >> '$log'
      exit 0
    fi
    echo \"[\$(date -Is)] failed with rc=\$rc; retrying in $RETRY_WAIT_SECONDS seconds\" >> '$log'
    sleep '$RETRY_WAIT_SECONDS'
  done
" >/dev/null 2>&1 < /dev/null &

echo "Started resilient download job: $job_name"
echo "PID: $!"
echo "Log: $log"
echo "PID file: $pidfile"
