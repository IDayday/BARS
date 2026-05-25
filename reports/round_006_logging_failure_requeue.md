# Round 006 Logging Failure Requeue

Generated: 2026-05-24T01:57:01+08:00.
Dry run: `False`.

Only failures with both `FileNotFoundError` and `events.out.tfevents` in the phase logs are eligible.
These are logging-path failures, not algorithm-result evidence.

| env | seed | action | requeues_before | latest_ckpt_step | latest_ckpt | status_file |
| --- | ---: | --- | ---: | ---: | --- | --- |
| antmaze-giant-navigate-v0 | 42 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed42/tdr/round006_selftrain_antmaze-giant-navigate_seed42/antmaze-giant-navigate-v0_sd042__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-navigate-v0/seed42/status.json |
| antmaze-giant-navigate-v0 | 43 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed43/tdr/round006_selftrain_antmaze-giant-navigate_seed43/antmaze-giant-navigate-v0_sd043__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-navigate-v0/seed43/status.json |
| antmaze-giant-navigate-v0 | 44 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed44/tdr/round006_selftrain_antmaze-giant-navigate_seed44/antmaze-giant-navigate-v0_sd044__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-navigate-v0/seed44/status.json |
| antmaze-giant-navigate-v0 | 45 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed45/tdr/round006_selftrain_antmaze-giant-navigate_seed45/antmaze-giant-navigate-v0_sd045__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-navigate-v0/seed45/status.json |
| antmaze-giant-navigate-v0 | 46 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed46/tdr/round006_selftrain_antmaze-giant-navigate_seed46/antmaze-giant-navigate-v0_sd046__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-navigate-v0/seed46/status.json |
| antmaze-giant-stitch-v0 | 42 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed42/tdr/round006_selftrain_antmaze-giant-stitch_seed42/antmaze-giant-stitch-v0_sd042__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-stitch-v0/seed42/status.json |
| antmaze-giant-stitch-v0 | 43 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed43/tdr/round006_selftrain_antmaze-giant-stitch_seed43/antmaze-giant-stitch-v0_sd043__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-stitch-v0/seed43/status.json |
| antmaze-giant-stitch-v0 | 44 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed44/tdr/round006_selftrain_antmaze-giant-stitch_seed44/antmaze-giant-stitch-v0_sd044__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-stitch-v0/seed44/status.json |
| antmaze-giant-stitch-v0 | 45 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed45/tdr/round006_selftrain_antmaze-giant-stitch_seed45/antmaze-giant-stitch-v0_sd045__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-stitch-v0/seed45/status.json |
| antmaze-giant-stitch-v0 | 46 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed46/tdr/round006_selftrain_antmaze-giant-stitch_seed46/antmaze-giant-stitch-v0_sd046__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-giant-stitch-v0/seed46/status.json |
| antmaze-large-explore-v0 | 42 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed42/tdr/round006_selftrain_antmaze-large-explore_seed42/antmaze-large-explore-v0_sd042__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-large-explore-v0/seed42/status.json |
| antmaze-large-explore-v0 | 43 | requeued | 0 | 100000 | artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed43/tdr/round006_selftrain_antmaze-large-explore_seed43/antmaze-large-explore-v0_sd043__2026-05-24_01-28-31/params_100000.pkl | runs_round006_gas_dynamic/antmaze-large-explore-v0/seed43/status.json |
