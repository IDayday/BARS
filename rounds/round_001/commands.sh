#!/usr/bin/env bash
set -euo pipefail
python -m compileall bars scripts
python scripts/stage25_enrich_failure_atlas_all_variants.py --eval-roots runs_stage24_reachability_confirm,runs_stage24_local_drift --out reports/round_001_failure_atlas_all_variants.csv --summary-out reports/round_001_failure_atlas_summary.csv --integrity-out reports/round_001_label_integrity.json --min-episodes 100
python scripts/stage25_analyze.py --reports-root reports --failure-atlas reports/round_001_failure_atlas_all_variants.csv --min-episodes 100
