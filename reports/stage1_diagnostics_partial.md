# Stage1 Report

## Run Completion
- total runs: 5
- completed: 0
- failed: 5
- terminated: 0
- archived: 5

## Core Metrics
_No data._

## Diagnostics
_No data._

## Failure Modes
| failure_mode | count |
| --- | --- |
| dataset_truncated | 5 |

## Decision
- decision: RETRY_STAGE

## Next Actions
- python scripts/collect_csv.py --log-root runs_stage1_diag
- python scripts/analyze_bars_results.py --log-root runs_stage1_diag --stage stage1
- Rerun only failed or incomplete runs after the current code fixes are validated.
