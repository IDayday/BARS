# Round 006 Interim Training Status and Results

Generated: 2026-05-24T07:52:00+08:00.

Baseline-only GAS run. This is an interim training operations report, not a BARS mechanism interpretation.

## Scheduler Status

- Total jobs: `120`.
- Completed/evaluated: `12`.
- Currently launched: `12`.
- Queued/not started: `96`.
- Retry pending: `0`.
- Failed: `0`.
- Datasets: `24/24` ready.
- The old orchestrator was stuck on zombie worker PIDs after 12 completions; it was replaced with PID `171747` and resumed from checkpoints.

## Completed Eval Summary

| env | n | mean | std | min | max | seeds | scores |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| antmaze-large-explore-v0 | 3 | 90.9 | 3.6 | 86.8 | 93.2 | 44,45,46 | 86.8, 93.2, 92.8 |
| antmaze-large-navigate-v0 | 4 | 94.2 | 1.4 | 92.4 | 95.6 | 42,43,44,45 | 94.8, 94.0, 95.6, 92.4 |
| scene-play-v0 | 5 | 57.7 | 10.4 | 45.2 | 72.4 | 42,43,44,45,46 | 53.2, 63.2, 72.4, 54.4, 45.2 |

## Aggregate Over Completed Rows

- Completed-row mean: `78.2` pp over `12` evaluated jobs.

## Active Launched Jobs

| env | seed | phase/status |
| --- | ---: | --- |
| antmaze-giant-navigate-v0 | 42 | launched |
| antmaze-giant-navigate-v0 | 43 | launched |
| antmaze-giant-navigate-v0 | 44 | launched |
| antmaze-giant-navigate-v0 | 45 | launched |
| antmaze-giant-navigate-v0 | 46 | launched |
| antmaze-giant-stitch-v0 | 42 | launched |
| antmaze-giant-stitch-v0 | 43 | launched |
| antmaze-giant-stitch-v0 | 44 | launched |
| antmaze-giant-stitch-v0 | 45 | launched |
| antmaze-giant-stitch-v0 | 46 | launched |
| antmaze-large-explore-v0 | 42 | launched |
| antmaze-large-explore-v0 | 43 | launched |

## Completed Eval Files

| env | seed | score_pp | eval_csv |
| --- | ---: | ---: | --- |
| antmaze-large-explore-v0 | 44 | 86.8 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed44/policy/round006_selftrain_antmaze-large-explore_seed44/antmaze-large-explore-v0_sd044__2026-05-24_02-57-12/eval.csv` |
| antmaze-large-explore-v0 | 45 | 93.2 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed45/policy/round006_selftrain_antmaze-large-explore_seed45/antmaze-large-explore-v0_sd045__2026-05-24_02-57-07/eval.csv` |
| antmaze-large-explore-v0 | 46 | 92.8 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed46/policy/round006_selftrain_antmaze-large-explore_seed46/antmaze-large-explore-v0_sd046__2026-05-24_02-52-32/eval.csv` |
| antmaze-large-navigate-v0 | 42 | 94.8 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed42/policy/round006_selftrain_antmaze-large-navigate_seed42/antmaze-large-navigate-v0_sd042__2026-05-24_02-54-01/eval.csv` |
| antmaze-large-navigate-v0 | 43 | 94.0 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed43/policy/round006_selftrain_antmaze-large-navigate_seed43/antmaze-large-navigate-v0_sd043__2026-05-24_02-53-57/eval.csv` |
| antmaze-large-navigate-v0 | 44 | 95.6 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed44/policy/round006_selftrain_antmaze-large-navigate_seed44/antmaze-large-navigate-v0_sd044__2026-05-24_03-00-13/eval.csv` |
| antmaze-large-navigate-v0 | 45 | 92.4 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed45/policy/round006_selftrain_antmaze-large-navigate_seed45/antmaze-large-navigate-v0_sd045__2026-05-24_03-06-04/eval.csv` |
| scene-play-v0 | 42 | 53.2 | `artifacts/gas_selftrain_round006/scene-play-v0/seed42/policy/round006_selftrain_scene-play_seed42/scene-play-v0_sd042__2026-05-24_03-00-19/eval.csv` |
| scene-play-v0 | 43 | 63.2 | `artifacts/gas_selftrain_round006/scene-play-v0/seed43/policy/round006_selftrain_scene-play_seed43/scene-play-v0_sd043__2026-05-24_02-54-51/eval.csv` |
| scene-play-v0 | 44 | 72.4 | `artifacts/gas_selftrain_round006/scene-play-v0/seed44/policy/round006_selftrain_scene-play_seed44/scene-play-v0_sd044__2026-05-24_02-54-50/eval.csv` |
| scene-play-v0 | 45 | 54.4 | `artifacts/gas_selftrain_round006/scene-play-v0/seed45/policy/round006_selftrain_scene-play_seed45/scene-play-v0_sd045__2026-05-24_02-54-56/eval.csv` |
| scene-play-v0 | 46 | 45.2 | `artifacts/gas_selftrain_round006/scene-play-v0/seed46/policy/round006_selftrain_scene-play_seed46/scene-play-v0_sd046__2026-05-24_02-54-54/eval.csv` |
