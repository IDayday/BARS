# Round 006 GAS Completed Eval Summary

Generated: 2026-05-27T15:22:53+08:00.

Baseline-only GAS full-budget self-training summary. This file reports completed eval rows only and does not interpret BARS mechanisms.

- Evidence class for completed eval rows: `E4_FULL_BUDGET_TRAINED_METHOD`.
- Total jobs: `120`.
- Job table status counts: `{'completed': 64, 'launched': 14, 'queued': 42}`.
- Completed eval rows: `64`.
- Completed-row aggregate mean/std/min/max: `86.7` / `12.6` / `45.2` / `99.6` pp.
- Source jobs table: `reports/round_006_gas_dynamic_jobs.tsv`.

| env | n | mean | std | min | max | seeds | scores |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| antmaze-giant-navigate-v0 | 5 | 74.3 | 8.7 | 64.0 | 86.0 | 42,43,44,45,46 | 71.6,70.0,80.0,64.0,86.0 |
| antmaze-giant-stitch-v0 | 5 | 87.0 | 2.5 | 84.4 | 90.8 | 42,43,44,45,46 | 84.4,87.6,90.8,86.8,85.2 |
| antmaze-large-explore-v0 | 5 | 92.0 | 3.0 | 86.8 | 94.4 | 42,43,44,45,46 | 92.8,94.4,86.8,93.2,92.8 |
| antmaze-large-navigate-v0 | 5 | 93.8 | 1.5 | 92.0 | 95.6 | 42,43,44,45,46 | 94.8,94.0,95.6,92.4,92.0 |
| antmaze-large-stitch-v0 | 5 | 95.3 | 1.0 | 94.4 | 96.8 | 42,43,44,45,46 | 96.8,95.6,95.2,94.4,94.4 |
| antmaze-medium-explore-v0 | 5 | 96.6 | 1.8 | 94.8 | 99.6 | 42,43,44,45,46 | 95.6,96.4,96.8,99.6,94.8 |
| antmaze-medium-navigate-v0 | 5 | 97.0 | 1.2 | 95.6 | 98.4 | 42,43,44,45,46 | 95.6,98.0,96.4,96.8,98.4 |
| antmaze-medium-stitch-v0 | 5 | 97.3 | 0.7 | 96.4 | 98.4 | 42,43,44,45,46 | 97.2,97.2,96.4,98.4,97.2 |
| humanoidmaze-giant-navigate-v0 | 5 | 79.8 | 8.6 | 65.6 | 86.8 | 42,43,44,45,46 | 65.6,78.0,85.2,83.2,86.8 |
| humanoidmaze-large-navigate-v0 | 5 | 76.6 | 8.7 | 66.8 | 83.6 | 42,43,44,45,46 | 82.8,83.6,67.2,66.8,82.4 |
| humanoidmaze-large-stitch-v0 | 4 | 83.1 | 4.4 | 76.8 | 87.2 | 43,44,45,46 | 84.4,84.0,76.8,87.2 |
| humanoidmaze-medium-navigate-v0 | 5 | 95.8 | 1.4 | 94.0 | 97.2 | 42,43,44,45,46 | 96.8,97.2,94.8,94.0,96.4 |
| scene-play-v0 | 5 | 57.7 | 10.4 | 45.2 | 72.4 | 42,43,44,45,46 | 53.2,63.2,72.4,54.4,45.2 |

Weights/checkpoints are intentionally excluded from git-tracked Round006 artifacts.
