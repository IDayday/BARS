#!/usr/bin/env bash
set -euo pipefail
python -m bars.cli run --config configs/toy_smoke.json --run-dir runs/toy/smoke
python scripts/collect_csv.py --log-root runs
