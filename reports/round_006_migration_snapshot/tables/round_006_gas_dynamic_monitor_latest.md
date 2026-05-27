# Round 006 GAS Monitor Snapshot

- Generated: 2026-05-26T13:42:13+08:00
- Evidence class: `E4_FULL_BUDGET_TRAINED_METHOD` pending completed eval rows.
- Baseline-only monitor: no BARS mechanism interpretation.
- Orchestrator PID: `11492` (`alive`)
- Jobs table: `reports/round_006_gas_dynamic_jobs.tsv`
- Dataset table: `reports/round_006_ogbench_download_status.tsv`
- Artifact root: `artifacts/gas_selftrain_round006`

## Queue Status

| status | count |
| --- | ---: |
| completed | 64 |
| launched | 14 |
| queued | 42 |

## Phase Status

| phase | count |
| --- | ---: |
| await_eval | 6 |
| evaluated | 64 |
| not_started | 42 |
| policy | 2 |
| tdr | 6 |

## Datasets

- Ready: `24/24`
- Pending: `none`

## GPU

| gpu | used_mb | free_mb | util_pct | name |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1077 | 23181 | 0 | NVIDIA GeForce RTX 3090 |
| 1 | 2657 | 21602 | 69 | NVIDIA GeForce RTX 3090 |
| 2 | 1862 | 22396 | 6 | NVIDIA GeForce RTX 3090 |
| 3 | 1057 | 23201 | 7 | NVIDIA GeForce RTX 3090 |
| 4 | 1 | 24258 | 0 | NVIDIA GeForce RTX 3090 |
| 5 | 1 | 24258 | 0 | NVIDIA GeForce RTX 3090 |

## Active Jobs

| env | seed | gpu | status | phase | tdr | policy | pct | pid | alive |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| humanoidmaze-giant-stitch-v0 | 42 | 1 | launched | await_eval | 1000000/1000000 | 1000000/1000000 | 100.0 | 11502 | True |
| humanoidmaze-giant-stitch-v0 | 43 | 1 | launched | await_eval | 1000000/1000000 | 1000000/1000000 | 100.0 | 11503 | True |
| humanoidmaze-giant-stitch-v0 | 44 | 2 | launched | await_eval | 1000000/1000000 | 1000000/1000000 | 100.0 | 11504 | True |
| humanoidmaze-giant-stitch-v0 | 45 | 2 | launched | await_eval | 1000000/1000000 | 1000000/1000000 | 100.0 | 11505 | True |
| humanoidmaze-giant-stitch-v0 | 46 | 3 | launched | await_eval | 1000000/1000000 | 1000000/1000000 | 100.0 | 11506 | True |
| humanoidmaze-large-stitch-v0 | 42 | 3 | launched | await_eval | 1000000/1000000 | 1000000/1000000 | 100.0 | 11507 | True |
| humanoidmaze-medium-stitch-v0 | 42 | 0 | launched | policy | 1000000/1000000 | 30000/1000000 | 51.5 | 11508 | True |
| humanoidmaze-medium-stitch-v0 | 43 | 0 | launched | policy | 1000000/1000000 | 30000/1000000 | 51.5 | 11509 | True |
| humanoidmaze-medium-stitch-v0 | 44 | 1 | launched | tdr | 10000/1000000 | 0/1000000 | 0.5 | 11510 | True |
| humanoidmaze-medium-stitch-v0 | 45 | 1 | launched | tdr | 10000/1000000 | 0/1000000 | 0.5 | 11511 | True |
| humanoidmaze-medium-stitch-v0 | 46 | 2 | launched | tdr | 10000/1000000 | 0/1000000 | 0.5 | 11512 | True |
| visual-antmaze-giant-navigate-v0 | 42 | 3 | launched | tdr | 5000/500000 | 0/500000 | 0.5 | 11513 | True |
| visual-antmaze-giant-navigate-v0 | 43 | 4 | launched | tdr | 5000/500000 | 0/500000 | 0.5 | 11514 | True |
| visual-antmaze-giant-navigate-v0 | 44 | 5 | launched | tdr | 5000/500000 | 0/500000 | 0.5 | 11515 | True |

## Failures

| env | seed | phase | tdr | policy | error |
| --- | ---: | --- | ---: | ---: | --- |
| none |  |  |  |  |  |

## Completed Eval Results

| env | seed | score_pp | eval_csv |
| --- | ---: | ---: | --- |
| antmaze-giant-navigate-v0 | 42 | 71.6 | `artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed42/policy/round006_selftrain_antmaze-giant-navigate_seed42/antmaze-giant-navigate-v0_sd042__2026-05-24_21-53-53/eval.csv` |
| antmaze-giant-navigate-v0 | 43 | 70.0 | `artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed43/policy/round006_selftrain_antmaze-giant-navigate_seed43/antmaze-giant-navigate-v0_sd043__2026-05-24_21-54-11/eval.csv` |
| antmaze-giant-navigate-v0 | 44 | 80.0 | `artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed44/policy/round006_selftrain_antmaze-giant-navigate_seed44/antmaze-giant-navigate-v0_sd044__2026-05-24_21-55-14/eval.csv` |
| antmaze-giant-navigate-v0 | 45 | 64.0 | `artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed45/policy/round006_selftrain_antmaze-giant-navigate_seed45/antmaze-giant-navigate-v0_sd045__2026-05-24_21-55-07/eval.csv` |
| antmaze-giant-navigate-v0 | 46 | 86.0 | `artifacts/gas_selftrain_round006/antmaze-giant-navigate-v0/seed46/policy/round006_selftrain_antmaze-giant-navigate_seed46/antmaze-giant-navigate-v0_sd046__2026-05-24_21-52-44/eval.csv` |
| antmaze-giant-stitch-v0 | 42 | 84.4 | `artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed42/policy/round006_selftrain_antmaze-giant-stitch_seed42/antmaze-giant-stitch-v0_sd042__2026-05-24_21-51-54/eval.csv` |
| antmaze-giant-stitch-v0 | 43 | 87.6 | `artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed43/policy/round006_selftrain_antmaze-giant-stitch_seed43/antmaze-giant-stitch-v0_sd043__2026-05-24_21-47-43/eval.csv` |
| antmaze-giant-stitch-v0 | 44 | 90.8 | `artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed44/policy/round006_selftrain_antmaze-giant-stitch_seed44/antmaze-giant-stitch-v0_sd044__2026-05-24_21-57-16/eval.csv` |
| antmaze-giant-stitch-v0 | 45 | 86.8 | `artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed45/policy/round006_selftrain_antmaze-giant-stitch_seed45/antmaze-giant-stitch-v0_sd045__2026-05-24_21-50-29/eval.csv` |
| antmaze-giant-stitch-v0 | 46 | 85.2 | `artifacts/gas_selftrain_round006/antmaze-giant-stitch-v0/seed46/policy/round006_selftrain_antmaze-giant-stitch_seed46/antmaze-giant-stitch-v0_sd046__2026-05-24_21-50-33/eval.csv` |
| antmaze-large-explore-v0 | 42 | 92.8 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed42/policy/round006_selftrain_antmaze-large-explore_seed42/antmaze-large-explore-v0_sd042__2026-05-24_22-02-00/eval.csv` |
| antmaze-large-explore-v0 | 43 | 94.4 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed43/policy/round006_selftrain_antmaze-large-explore_seed43/antmaze-large-explore-v0_sd043__2026-05-24_21-59-59/eval.csv` |
| antmaze-large-explore-v0 | 44 | 86.8 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed44/policy/round006_selftrain_antmaze-large-explore_seed44/antmaze-large-explore-v0_sd044__2026-05-24_02-57-12/eval.csv` |
| antmaze-large-explore-v0 | 45 | 93.2 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed45/policy/round006_selftrain_antmaze-large-explore_seed45/antmaze-large-explore-v0_sd045__2026-05-24_02-57-07/eval.csv` |
| antmaze-large-explore-v0 | 46 | 92.8 | `artifacts/gas_selftrain_round006/antmaze-large-explore-v0/seed46/policy/round006_selftrain_antmaze-large-explore_seed46/antmaze-large-explore-v0_sd046__2026-05-24_02-52-32/eval.csv` |
| antmaze-large-navigate-v0 | 42 | 94.8 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed42/policy/round006_selftrain_antmaze-large-navigate_seed42/antmaze-large-navigate-v0_sd042__2026-05-24_02-54-01/eval.csv` |
| antmaze-large-navigate-v0 | 43 | 94.0 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed43/policy/round006_selftrain_antmaze-large-navigate_seed43/antmaze-large-navigate-v0_sd043__2026-05-24_02-53-57/eval.csv` |
| antmaze-large-navigate-v0 | 44 | 95.6 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed44/policy/round006_selftrain_antmaze-large-navigate_seed44/antmaze-large-navigate-v0_sd044__2026-05-24_03-00-13/eval.csv` |
| antmaze-large-navigate-v0 | 45 | 92.4 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed45/policy/round006_selftrain_antmaze-large-navigate_seed45/antmaze-large-navigate-v0_sd045__2026-05-24_03-06-04/eval.csv` |
| antmaze-large-navigate-v0 | 46 | 92.0 | `artifacts/gas_selftrain_round006/antmaze-large-navigate-v0/seed46/policy/round006_selftrain_antmaze-large-navigate_seed46/antmaze-large-navigate-v0_sd046__2026-05-24_12-46-10/eval.csv` |
| antmaze-large-stitch-v0 | 42 | 96.8 | `artifacts/gas_selftrain_round006/antmaze-large-stitch-v0/seed42/policy/round006_selftrain_antmaze-large-stitch_seed42/antmaze-large-stitch-v0_sd042__2026-05-24_12-55-03/eval.csv` |
| antmaze-large-stitch-v0 | 43 | 95.6 | `artifacts/gas_selftrain_round006/antmaze-large-stitch-v0/seed43/policy/round006_selftrain_antmaze-large-stitch_seed43/antmaze-large-stitch-v0_sd043__2026-05-24_12-53-05/eval.csv` |
| antmaze-large-stitch-v0 | 44 | 95.2 | `artifacts/gas_selftrain_round006/antmaze-large-stitch-v0/seed44/policy/round006_selftrain_antmaze-large-stitch_seed44/antmaze-large-stitch-v0_sd044__2026-05-24_12-57-08/eval.csv` |
| antmaze-large-stitch-v0 | 45 | 94.4 | `artifacts/gas_selftrain_round006/antmaze-large-stitch-v0/seed45/policy/round006_selftrain_antmaze-large-stitch_seed45/antmaze-large-stitch-v0_sd045__2026-05-24_12-57-05/eval.csv` |
| antmaze-large-stitch-v0 | 46 | 94.4 | `artifacts/gas_selftrain_round006/antmaze-large-stitch-v0/seed46/policy/round006_selftrain_antmaze-large-stitch_seed46/antmaze-large-stitch-v0_sd046__2026-05-24_12-58-29/eval.csv` |
| antmaze-medium-explore-v0 | 42 | 95.6 | `artifacts/gas_selftrain_round006/antmaze-medium-explore-v0/seed42/policy/round006_selftrain_antmaze-medium-explore_seed42/antmaze-medium-explore-v0_sd042__2026-05-24_15-30-48/eval.csv` |
| antmaze-medium-explore-v0 | 43 | 96.4 | `artifacts/gas_selftrain_round006/antmaze-medium-explore-v0/seed43/policy/round006_selftrain_antmaze-medium-explore_seed43/antmaze-medium-explore-v0_sd043__2026-05-24_15-32-27/eval.csv` |
| antmaze-medium-explore-v0 | 44 | 96.8 | `artifacts/gas_selftrain_round006/antmaze-medium-explore-v0/seed44/policy/round006_selftrain_antmaze-medium-explore_seed44/antmaze-medium-explore-v0_sd044__2026-05-24_15-35-49/eval.csv` |
| antmaze-medium-explore-v0 | 45 | 99.6 | `artifacts/gas_selftrain_round006/antmaze-medium-explore-v0/seed45/policy/round006_selftrain_antmaze-medium-explore_seed45/antmaze-medium-explore-v0_sd045__2026-05-24_15-39-38/eval.csv` |
| antmaze-medium-explore-v0 | 46 | 94.8 | `artifacts/gas_selftrain_round006/antmaze-medium-explore-v0/seed46/policy/round006_selftrain_antmaze-medium-explore_seed46/antmaze-medium-explore-v0_sd046__2026-05-24_15-39-39/eval.csv` |
| antmaze-medium-navigate-v0 | 42 | 95.6 | `artifacts/gas_selftrain_round006/antmaze-medium-navigate-v0/seed42/policy/round006_selftrain_antmaze-medium-navigate_seed42/antmaze-medium-navigate-v0_sd042__2026-05-24_12-49-23/eval.csv` |
| antmaze-medium-navigate-v0 | 43 | 98.0 | `artifacts/gas_selftrain_round006/antmaze-medium-navigate-v0/seed43/policy/round006_selftrain_antmaze-medium-navigate_seed43/antmaze-medium-navigate-v0_sd043__2026-05-24_12-49-18/eval.csv` |
| antmaze-medium-navigate-v0 | 44 | 96.4 | `artifacts/gas_selftrain_round006/antmaze-medium-navigate-v0/seed44/policy/round006_selftrain_antmaze-medium-navigate_seed44/antmaze-medium-navigate-v0_sd044__2026-05-24_12-48-52/eval.csv` |
| antmaze-medium-navigate-v0 | 45 | 96.8 | `artifacts/gas_selftrain_round006/antmaze-medium-navigate-v0/seed45/policy/round006_selftrain_antmaze-medium-navigate_seed45/antmaze-medium-navigate-v0_sd045__2026-05-24_12-48-54/eval.csv` |
| antmaze-medium-navigate-v0 | 46 | 98.4 | `artifacts/gas_selftrain_round006/antmaze-medium-navigate-v0/seed46/policy/round006_selftrain_antmaze-medium-navigate_seed46/antmaze-medium-navigate-v0_sd046__2026-05-24_12-54-15/eval.csv` |
| antmaze-medium-stitch-v0 | 42 | 97.2 | `artifacts/gas_selftrain_round006/antmaze-medium-stitch-v0/seed42/policy/round006_selftrain_antmaze-medium-stitch_seed42/antmaze-medium-stitch-v0_sd042__2026-05-24_12-58-32/eval.csv` |
| antmaze-medium-stitch-v0 | 43 | 97.2 | `artifacts/gas_selftrain_round006/antmaze-medium-stitch-v0/seed43/policy/round006_selftrain_antmaze-medium-stitch_seed43/antmaze-medium-stitch-v0_sd043__2026-05-24_15-22-50/eval.csv` |
| antmaze-medium-stitch-v0 | 44 | 96.4 | `artifacts/gas_selftrain_round006/antmaze-medium-stitch-v0/seed44/policy/round006_selftrain_antmaze-medium-stitch_seed44/antmaze-medium-stitch-v0_sd044__2026-05-24_15-25-42/eval.csv` |
| antmaze-medium-stitch-v0 | 45 | 98.4 | `artifacts/gas_selftrain_round006/antmaze-medium-stitch-v0/seed45/policy/round006_selftrain_antmaze-medium-stitch_seed45/antmaze-medium-stitch-v0_sd045__2026-05-24_15-25-41/eval.csv` |
| antmaze-medium-stitch-v0 | 46 | 97.2 | `artifacts/gas_selftrain_round006/antmaze-medium-stitch-v0/seed46/policy/round006_selftrain_antmaze-medium-stitch_seed46/antmaze-medium-stitch-v0_sd046__2026-05-24_15-27-58/eval.csv` |

## Download Log Tail

```text
2026-05-24T01:49:34+08:00 START download visual-antmaze-large-stitch-v0
2026-05-24T01:50:52+08:00 DONE download visual-antmaze-large-stitch-v0: /root/remote/datasets/ogbench/visual-antmaze-large-stitch-v0.npz | /root/remote/datasets/ogbench/visual-antmaze-large-stitch-v0-val.npz
2026-05-24T01:51:52+08:00 START download visual-antmaze-medium-stitch-v0
2026-05-24T01:53:14+08:00 DONE download visual-antmaze-medium-stitch-v0: /root/remote/datasets/ogbench/visual-antmaze-medium-stitch-v0.npz | /root/remote/datasets/ogbench/visual-antmaze-medium-stitch-v0-val.npz
2026-05-24T01:54:14+08:00 START download visual-antmaze-large-explore-v0
2026-05-24T01:59:26+08:00 DONE download visual-antmaze-large-explore-v0: /root/remote/datasets/ogbench/visual-antmaze-large-explore-v0.npz | /root/remote/datasets/ogbench/visual-antmaze-large-explore-v0-val.npz
2026-05-24T02:00:26+08:00 START download visual-antmaze-medium-explore-v0
2026-05-24T02:06:08+08:00 DONE download visual-antmaze-medium-explore-v0: /root/remote/datasets/ogbench/visual-antmaze-medium-explore-v0.npz | /root/remote/datasets/ogbench/visual-antmaze-medium-explore-v0-val.npz
2026-05-24T02:07:08+08:00 START download visual-scene-play-v0
2026-05-24T02:08:19+08:00 DONE download visual-scene-play-v0: /root/remote/datasets/ogbench/visual-scene-play-v0.npz | /root/remote/datasets/ogbench/visual-scene-play-v0-val.npz
```

## Event Tail

```json
{"env": "humanoidmaze-medium-stitch-v0", "event": "job_launched", "gpu": "0", "log": "runs_round006_gas_dynamic/humanoidmaze-medium-stitch-v0/seed42/worker_supervisor.log", "pid": 11508, "seed": 42, "slot_cost": 1, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
{"env": "humanoidmaze-medium-stitch-v0", "event": "job_launched", "gpu": "0", "log": "runs_round006_gas_dynamic/humanoidmaze-medium-stitch-v0/seed43/worker_supervisor.log", "pid": 11509, "seed": 43, "slot_cost": 1, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
{"env": "humanoidmaze-medium-stitch-v0", "event": "job_launched", "gpu": "1", "log": "runs_round006_gas_dynamic/humanoidmaze-medium-stitch-v0/seed44/worker_supervisor.log", "pid": 11510, "seed": 44, "slot_cost": 1, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
{"env": "humanoidmaze-medium-stitch-v0", "event": "job_launched", "gpu": "1", "log": "runs_round006_gas_dynamic/humanoidmaze-medium-stitch-v0/seed45/worker_supervisor.log", "pid": 11511, "seed": 45, "slot_cost": 1, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
{"env": "humanoidmaze-medium-stitch-v0", "event": "job_launched", "gpu": "2", "log": "runs_round006_gas_dynamic/humanoidmaze-medium-stitch-v0/seed46/worker_supervisor.log", "pid": 11512, "seed": 46, "slot_cost": 1, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
{"env": "visual-antmaze-giant-navigate-v0", "event": "job_launched", "gpu": "3", "log": "runs_round006_gas_dynamic/visual-antmaze-giant-navigate-v0/seed42/worker_supervisor.log", "pid": 11513, "seed": 42, "slot_cost": 2, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
{"env": "visual-antmaze-giant-navigate-v0", "event": "job_launched", "gpu": "4", "log": "runs_round006_gas_dynamic/visual-antmaze-giant-navigate-v0/seed43/worker_supervisor.log", "pid": 11514, "seed": 43, "slot_cost": 2, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
{"env": "visual-antmaze-giant-navigate-v0", "event": "job_launched", "gpu": "5", "log": "runs_round006_gas_dynamic/visual-antmaze-giant-navigate-v0/seed44/worker_supervisor.log", "pid": 11515, "seed": 44, "slot_cost": 2, "started_at": "2026-05-26T13:41:42+08:00", "time": "2026-05-26T13:41:42+08:00"}
```

