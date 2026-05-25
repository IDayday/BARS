# Round 006 GAS Dynamic Download/Training Launch

Generated: 2026-05-25T09:40:43+08:00.

- Evidence class while running: `E4_FULL_BUDGET_TRAINED_METHOD` pending completion.
- Baseline-only run: no p_bridge, integrated BARS, oracle-headroom, boundary, or failure-taxonomy interpretation.
- Seeds: 42,43,44,45,46.
- Target OGBench envs: antmaze-giant-navigate-v0,antmaze-giant-stitch-v0,antmaze-large-explore-v0,scene-play-v0,antmaze-large-navigate-v0,antmaze-medium-navigate-v0,antmaze-large-stitch-v0,antmaze-medium-stitch-v0,antmaze-medium-explore-v0,humanoidmaze-giant-navigate-v0,humanoidmaze-large-navigate-v0,humanoidmaze-medium-navigate-v0,humanoidmaze-giant-stitch-v0,humanoidmaze-large-stitch-v0,humanoidmaze-medium-stitch-v0,visual-antmaze-giant-navigate-v0,visual-antmaze-large-navigate-v0,visual-antmaze-medium-navigate-v0,visual-antmaze-giant-stitch-v0,visual-antmaze-large-stitch-v0,visual-antmaze-medium-stitch-v0,visual-antmaze-large-explore-v0,visual-antmaze-medium-explore-v0,visual-scene-play-v0.
- Dataset root: `/root/remote/datasets/ogbench`.
- Artifact root: `artifacts/gas_selftrain_round006`.
- Run root: `runs_round006_gas_dynamic`.
- GPUs: `0,1,2,3,4,5` with slot capacity `2` per GPU.
- Checkpoint policy: full completed stage checkpoints may feed the next stage; interrupted intermediate checkpoints are never resumed.
- Download uses proxy-aware HTTP(S) environment variables inherited by curl/urllib/aria2.
- Common datasets are prioritized before additional antmaze, humanoidmaze, and visual datasets.
- Skipped: kitchen-partial-v0 (D4RL, not OGBench; skipped by this OGBench queue).

## Config Summary

| env | steps | encoder | batch | discount | expectile | alpha | p_aug | way_steps | te | eval_on_cpu | priority | slots |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-giant-navigate-v0 | 1000000 | not_used | 1024 | 0.995 | 0.999 | 1.0 | 0.0 | 8 | 0.99 | 1 | 0 | 1 |
| antmaze-giant-stitch-v0 | 1000000 | not_used | 1024 | 0.995 | 0.999 | 1.0 | 0.0 | 8 | 0.99 | 1 | 1 | 1 |
| antmaze-large-explore-v0 | 1000000 | not_used | 1024 | 0.99 | 0.999 | 0.01 | 0.0 | 8 | 0.99 | 1 | 2 | 1 |
| scene-play-v0 | 1000000 | not_used | 1024 | 0.99 | 0.999 | 1.0 | 0.0 | 48 | 0.99 | 1 | 3 | 1 |
| antmaze-large-navigate-v0 | 1000000 | not_used | 1024 | 0.99 | 0.999 | 1.0 | 0.0 | 8 | 0.99 | 1 | 4 | 1 |
| antmaze-medium-navigate-v0 | 1000000 | not_used | 1024 | 0.99 | 0.999 | 1.0 | 0.0 | 8 | 0.99 | 1 | 5 | 1 |
| antmaze-large-stitch-v0 | 1000000 | not_used | 1024 | 0.99 | 0.999 | 1.0 | 0.0 | 8 | 0.99 | 1 | 6 | 1 |
| antmaze-medium-stitch-v0 | 1000000 | not_used | 1024 | 0.99 | 0.999 | 1.0 | 0.0 | 8 | 0.99 | 1 | 7 | 1 |
| antmaze-medium-explore-v0 | 1000000 | not_used | 1024 | 0.99 | 0.999 | 0.01 | 0.0 | 8 | 0.99 | 1 | 8 | 1 |
| humanoidmaze-giant-navigate-v0 | 1000000 | not_used | 1024 | 0.995 | 0.95 | 0.1 | 0.0 | 32 | 0.99 | 1 | 9 | 1 |
| humanoidmaze-large-navigate-v0 | 1000000 | not_used | 1024 | 0.99 | 0.95 | 0.1 | 0.0 | 32 | 0.99 | 1 | 10 | 1 |
| humanoidmaze-medium-navigate-v0 | 1000000 | not_used | 1024 | 0.99 | 0.95 | 0.1 | 0.0 | 32 | 0.99 | 1 | 11 | 1 |
| humanoidmaze-giant-stitch-v0 | 1000000 | not_used | 1024 | 0.995 | 0.95 | 0.1 | 0.0 | 32 | 0.99 | 1 | 12 | 1 |
| humanoidmaze-large-stitch-v0 | 1000000 | not_used | 1024 | 0.99 | 0.95 | 0.1 | 0.0 | 32 | 0.99 | 1 | 13 | 1 |
| humanoidmaze-medium-stitch-v0 | 1000000 | not_used | 1024 | 0.99 | 0.95 | 0.1 | 0.0 | 32 | 0.99 | 1 | 14 | 1 |
| visual-antmaze-giant-navigate-v0 | 500000 | impala_small | 256 | 0.995 | 0.95 | 1.0 | 0.5 | 8 | 0.9 | 0 | 15 | 2 |
| visual-antmaze-large-navigate-v0 | 500000 | impala_small | 256 | 0.99 | 0.95 | 1.0 | 0.5 | 8 | 0.9 | 0 | 16 | 2 |
| visual-antmaze-medium-navigate-v0 | 500000 | impala_small | 256 | 0.99 | 0.95 | 1.0 | 0.5 | 8 | 0.9 | 0 | 17 | 2 |
| visual-antmaze-giant-stitch-v0 | 500000 | impala_small | 256 | 0.995 | 0.95 | 1.0 | 0.5 | 8 | 0.9 | 0 | 18 | 2 |
| visual-antmaze-large-stitch-v0 | 500000 | impala_small | 256 | 0.99 | 0.95 | 1.0 | 0.5 | 8 | 0.9 | 0 | 19 | 2 |
| visual-antmaze-medium-stitch-v0 | 500000 | impala_small | 256 | 0.99 | 0.95 | 1.0 | 0.5 | 8 | 0.9 | 0 | 20 | 2 |
| visual-antmaze-large-explore-v0 | 500000 | impala_small | 256 | 0.99 | 0.95 | 0.01 | 0.5 | 8 | 0.9 | 0 | 21 | 2 |
| visual-antmaze-medium-explore-v0 | 500000 | impala_small | 256 | 0.99 | 0.95 | 0.01 | 0.5 | 8 | 0.9 | 0 | 22 | 2 |
| visual-scene-play-v0 | 500000 | impala_small | 256 | 0.99 | 0.95 | 1.0 | 0.5 | 24 | 0.9 | 0 | 23 | 2 |

## Live Files

- Jobs: `reports/round_006_gas_dynamic_jobs.tsv`
- Dataset status: `reports/round_006_ogbench_download_status.tsv`
- Events: `runs_round006_gas_dynamic/_orchestrator/events.jsonl`
