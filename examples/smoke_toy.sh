#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
python -m bars.cli run --config configs/toy_smoke.json --run-dir runs/toy/smoke
python scripts/collect_csv.py --log-root runs
