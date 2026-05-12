#!/usr/bin/env bash
set -Eeuo pipefail
LOG_ROOT="${1:-runs_stage16_sanity4}"
python scripts/collect_csv.py --log-root "$LOG_ROOT"
python scripts/analyze_bars_results.py --log-root "$LOG_ROOT" --stage stage1 --out "reports/$(basename "$LOG_ROOT")_stage1_report.md"
python - <<'PY' "$LOG_ROOT"
from pathlib import Path
import sys
import pandas as pd
root = Path(sys.argv[1]) / '_analysis'
for name in ['profile_all.csv','graph_all.csv','diagnostics_all.csv']:
    p = root / name
    print('\n===', name, '===')
    if p.exists():
        df = pd.read_csv(p)
        print(df.tail(20).to_string(index=False))
    else:
        print('missing')
PY
