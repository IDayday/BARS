#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-runs}"
GPUS="${GPUS:-auto}"
INTERVAL_MIN="${INTERVAL_MIN:-30}"
FORCE_STALE="${FORCE_STALE:-0}"
MONITOR_DIR="${LOG_ROOT}/_monitor"
MONITOR_LOG="${MONITOR_DIR}/monitor.log"
STALE_JSON="${MONITOR_DIR}/stale_counts.json"
mkdir -p "${MONITOR_DIR}"

if [ ! -f "${STALE_JSON}" ]; then
  printf '{}\n' > "${STALE_JSON}"
fi

while true; do
  {
    echo "===== $(date '+%F %T %Z') ====="
    echo "[nvidia-smi]"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    echo
    echo "[jobctl status]"
    python -m bars.sched.jobctl status --log-root "${LOG_ROOT}" --gpus "${GPUS}" || true
    echo
    echo "[recent failed runs]"
    python - <<'PY' "${LOG_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in root.glob('**/manifest.json'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue
    if data.get('status') == 'failed':
        rows.append((data.get('created_at', 0), data.get('run_id', ''), data.get('run_dir', '')))
for _, run_id, run_dir in sorted(rows)[-10:]:
    print(f'{run_id} {run_dir}')
PY
    echo
    echo "[stale runs]"
  } >> "${MONITOR_LOG}"

  python - <<'PY' "${LOG_ROOT}" "${STALE_JSON}" "${FORCE_STALE}" "${MONITOR_LOG}"
import json
import os
import subprocess
import sys
import time
from pathlib import Path

log_root = Path(sys.argv[1])
stale_json = Path(sys.argv[2])
force_stale = str(sys.argv[3]) == '1'
monitor_log = Path(sys.argv[4])
now = time.time()

try:
    stale_counts = json.loads(stale_json.read_text(encoding='utf-8'))
except Exception:
    stale_counts = {}

def last_update(run_dir: Path):
    paths = [
        run_dir / 'stdout.log',
        run_dir / 'stderr.log',
        run_dir / 'logs' / 'summary.csv',
        run_dir / 'logs' / 'diagnostics.csv',
        run_dir / 'logs' / 'graph.csv',
    ]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(p.stat().st_mtime for p in existing)

states = []
jobs_dir = log_root / '_jobs'
for path in jobs_dir.glob('*.json'):
    try:
        states.append(json.loads(path.read_text(encoding='utf-8')))
    except Exception:
        continue

new_counts = {}
for st in states:
    if st.get('status') != 'running':
        continue
    run_dir = Path(st.get('run_dir', ''))
    ts = last_update(run_dir) if run_dir else None
    age_min = None if ts is None else (now - ts) / 60.0
    stale = age_min is not None and age_min >= 90.0
    run_id = st.get('run_id', '')
    count = stale_counts.get(run_id, 0)
    if stale:
        count += 1
        with monitor_log.open('a', encoding='utf-8') as f:
            f.write(f'{run_id} stale age_min={age_min:.1f} consecutive={count} run_dir={run_dir}\n')
        if count >= 2 and st.get('status') == 'running':
            subprocess.run([sys.executable, '-m', 'bars.sched.jobctl', 'stop', '--log-root', str(log_root), '--run-id', run_id], check=False)
            with monitor_log.open('a', encoding='utf-8') as f:
                f.write(f'{run_id} graceful stop requested after consecutive stale detections\n')
        if force_stale and count >= 3:
            subprocess.run([sys.executable, '-m', 'bars.sched.jobctl', 'stop', '--log-root', str(log_root), '--run-id', run_id, '--force'], check=False)
            with monitor_log.open('a', encoding='utf-8') as f:
                f.write(f'{run_id} force stop requested after repeated stale detections\n')
    else:
        count = 0
    new_counts[run_id] = count

stale_json.write_text(json.dumps(new_counts, indent=2, sort_keys=True), encoding='utf-8')
PY

  {
    echo
    echo "[archive count]"
    find "${LOG_ROOT}" -path '*/archives/*.tar.gz' | wc -l
    echo
    echo "[latest summary tails]"
    find "${LOG_ROOT}" -path '*/logs/summary.csv' | sort | tail -n 3 | while read -r f; do
      echo "----- ${f}"
      tail -n 5 "${f}" || true
    done
    echo
    echo "[scheduler.out tail]"
    tail -n 80 "${LOG_ROOT}/_jobs/scheduler.out" 2>/dev/null || true
    echo
  } >> "${MONITOR_LOG}"

  sleep "$(( INTERVAL_MIN * 60 ))"
done
