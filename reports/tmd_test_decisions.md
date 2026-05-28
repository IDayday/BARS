# tmd-test Decisions

## Branch and protocol

Authorized branch: `stage25-protocol-oracle-drift`.
Fallback mode: `none`.

## Implemented components

`tmd_test` wrapper, calibration, key nodes, directed key graph, construct/eval/analyze scripts.

## Smoke result

Evaluation rows: 9102.
Graph rows: 26.

## Graph diagnostics

See `reports/tmd_test_graph_diagnostics.md`.

## Evaluation result

See `reports/tmd_test_eval_summary.md`.

## Mode/task aggregate

| run_name | env | seed | gas_seed | mode | task_id | episodes | success_rate | mean_steps | mean_no_path_rate | mean_switches | mean_final_goal_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gas_direct_goal | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 1 | 2 | 1.000 | 419.0 | 0.000 | 0.0 | 0.0 |
| gas_direct_goal_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 1 | 2 | 1.000 | 292.0 | 0.000 | 0.0 | 0.0 |
| gas_direct_goal_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 2 | 2 | 1.000 | 268.0 | 0.000 | 0.0 | 0.0 |
| gas_direct_goal_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 3 | 2 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| gas_direct_goal_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 4 | 2 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| gas_direct_goal_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_direct_goal | 5 | 2 | 0.500 | 656.0 | 0.000 | 0.0 | 0.0 |
| gas_graph_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 1 | 2 | 1.000 | 285.5 | 0.000 | 0.0 | 37.0 |
| gas_graph_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 2 | 2 | 1.000 | 206.0 | 0.000 | 0.0 | 23.0 |
| gas_graph_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 3 | 2 | 1.000 | 141.0 | 0.000 | 0.0 | 21.5 |
| gas_graph_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 4 | 2 | 0.500 | 600.5 | 0.000 | 0.0 | 417.0 |
| gas_graph_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 5 | 2 | 1.000 | 249.0 | 0.000 | 0.0 | 76.5 |
| gas_graph_policy_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 1 | 10 | 0.900 | 392.7 | 0.000 | 0.0 | 58.1 |
| gas_graph_policy_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 2 | 10 | 1.000 | 215.9 | 0.000 | 0.0 | 25.3 |
| gas_graph_policy_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 3 | 10 | 1.000 | 143.0 | 0.000 | 0.0 | 28.5 |
| gas_graph_policy_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 4 | 10 | 1.000 | 247.7 | 0.000 | 0.0 | 51.9 |
| gas_graph_policy_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 5 | 10 | 0.900 | 296.4 | 0.000 | 0.0 | 119.9 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 0.0 | 31.7 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 10 | 0.900 | 736.5 | 0.000 | 0.0 | 45.2 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 10 | 0.600 | 905.8 | 0.000 | 0.0 | 39.6 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 10 | 0.700 | 772.1 | 0.000 | 0.0 | 23.5 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 5 | 10 | 0.500 | 790.7 | 0.000 | 0.0 | 20.7 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_policy | 1 | 10 | 0.800 | 851.5 | 0.000 | 0.0 | 33.1 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_policy | 2 | 10 | 0.800 | 755.8 | 0.000 | 0.0 | 43.0 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_policy | 3 | 10 | 0.700 | 863.7 | 0.000 | 0.0 | 31.8 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_policy | 4 | 10 | 1.000 | 689.5 | 0.000 | 0.0 | 36.1 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_policy | 5 | 10 | 0.600 | 663.5 | 0.000 | 0.0 | 19.5 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 10 | 0.800 | 532.3 | 0.000 | 0.0 | 95.4 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 10 | 0.900 | 587.3 | 0.000 | 0.0 | 78.7 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 10 | 1.000 | 288.7 | 0.000 | 0.0 | 32.8 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 10 | 1.000 | 392.3 | 0.000 | 0.0 | 42.5 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-navigate-v0 | 0 | 42 | gas_graph_policy | 5 | 10 | 1.000 | 391.3 | 0.000 | 0.0 | 32.2 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-stitch-v0 | 0 | 42 | gas_graph_policy | 1 | 10 | 0.900 | 437.8 | 0.000 | 0.0 | 30.7 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-stitch-v0 | 0 | 42 | gas_graph_policy | 2 | 10 | 0.800 | 595.9 | 0.000 | 0.0 | 22.8 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-stitch-v0 | 0 | 42 | gas_graph_policy | 3 | 10 | 1.000 | 282.3 | 0.000 | 0.0 | 28.4 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-stitch-v0 | 0 | 42 | gas_graph_policy | 4 | 10 | 1.000 | 345.0 | 0.000 | 0.0 | 28.2 |
| gas_graph_policy_alltasks_ep10_gasseed42_screen | antmaze-large-stitch-v0 | 0 | 42 | gas_graph_policy | 5 | 10 | 0.800 | 508.9 | 0.000 | 0.0 | 21.9 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 1 | 10 | 0.200 | 977.5 | 0.000 | 0.0 | 112.5 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 2 | 10 | 0.900 | 749.5 | 0.000 | 0.0 | 92.1 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 3 | 10 | 0.700 | 838.7 | 0.000 | 0.0 | 42.7 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 4 | 10 | 1.000 | 675.6 | 0.000 | 0.0 | 36.0 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 5 | 10 | 0.800 | 537.7 | 0.000 | 0.0 | 44.7 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_policy | 1 | 10 | 0.400 | 955.9 | 0.000 | 0.0 | 40.9 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_policy | 2 | 10 | 0.800 | 781.7 | 0.000 | 0.0 | 83.0 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_policy | 3 | 10 | 0.800 | 807.6 | 0.000 | 0.0 | 20.0 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_policy | 4 | 10 | 0.900 | 686.0 | 0.000 | 0.0 | 28.2 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_policy | 5 | 10 | 1.000 | 448.4 | 0.000 | 0.0 | 39.4 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-large-stitch-v0 | 0 | 43 | gas_graph_policy | 1 | 10 | 0.900 | 487.2 | 0.000 | 0.0 | 97.3 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-large-stitch-v0 | 0 | 43 | gas_graph_policy | 2 | 10 | 0.900 | 543.9 | 0.000 | 0.0 | 78.4 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-large-stitch-v0 | 0 | 43 | gas_graph_policy | 3 | 10 | 1.000 | 295.1 | 0.000 | 0.0 | 39.0 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-large-stitch-v0 | 0 | 43 | gas_graph_policy | 4 | 10 | 1.000 | 352.6 | 0.000 | 0.0 | 20.7 |
| gas_graph_policy_alltasks_ep10_gasseed43_screen | antmaze-large-stitch-v0 | 0 | 43 | gas_graph_policy | 5 | 10 | 1.000 | 417.6 | 0.000 | 0.0 | 19.1 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 20 | 0.000 | 1000.0 | 0.000 | 0.0 | 30.5 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 20 | 0.800 | 783.0 | 0.000 | 0.0 | 49.9 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 20 | 0.500 | 896.0 | 0.000 | 0.0 | 47.2 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 20 | 0.850 | 773.0 | 0.000 | 0.0 | 44.2 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 5 | 20 | 0.550 | 769.2 | 0.000 | 0.0 | 56.3 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_policy | 1 | 20 | 0.050 | 998.6 | 0.000 | 0.0 | 43.8 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_policy | 2 | 20 | 0.850 | 765.1 | 0.000 | 0.0 | 42.0 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_policy | 3 | 20 | 0.850 | 818.5 | 0.000 | 0.0 | 45.1 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_policy | 4 | 20 | 0.750 | 826.2 | 0.000 | 0.0 | 30.1 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_policy | 5 | 20 | 0.450 | 843.4 | 0.000 | 0.0 | 19.9 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 1 | 20 | 0.950 | 323.1 | 0.000 | 0.0 | 49.4 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 2 | 20 | 0.950 | 278.3 | 0.000 | 0.0 | 43.6 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 20 | 0.900 | 249.5 | 0.000 | 0.0 | 31.9 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 4 | 20 | 1.000 | 268.1 | 0.000 | 0.0 | 46.0 |
| gas_graph_policy_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 5 | 20 | 1.000 | 203.2 | 0.000 | 0.0 | 40.0 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 1 | 20 | 0.400 | 957.5 | 0.000 | 0.0 | 77.7 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 2 | 20 | 0.900 | 721.4 | 0.000 | 0.0 | 52.8 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 3 | 20 | 0.800 | 849.6 | 0.000 | 0.0 | 45.0 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 4 | 20 | 0.850 | 724.1 | 0.000 | 0.0 | 29.1 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 5 | 20 | 0.950 | 457.6 | 0.000 | 0.0 | 39.6 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_policy | 1 | 20 | 0.100 | 992.2 | 0.000 | 0.0 | 91.0 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_policy | 2 | 20 | 0.750 | 764.0 | 0.000 | 0.0 | 38.5 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_policy | 3 | 20 | 0.750 | 817.1 | 0.000 | 0.0 | 39.5 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_policy | 4 | 20 | 0.800 | 786.9 | 0.000 | 0.0 | 33.4 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_policy | 5 | 20 | 0.900 | 524.6 | 0.000 | 0.0 | 84.0 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 1 | 20 | 1.000 | 289.9 | 0.000 | 0.0 | 53.1 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 2 | 20 | 1.000 | 234.2 | 0.000 | 0.0 | 37.0 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 3 | 20 | 0.950 | 203.9 | 0.000 | 0.0 | 33.8 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 4 | 20 | 1.000 | 240.6 | 0.000 | 0.0 | 30.1 |
| gas_graph_policy_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 5 | 20 | 1.000 | 229.6 | 0.000 | 0.0 | 61.0 |
| gas_graph_policy_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 1 | 50 | 0.920 | 367.7 | 0.000 | 0.0 | 54.9 |
| gas_graph_policy_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 2 | 50 | 0.980 | 232.3 | 0.000 | 0.0 | 42.9 |
| gas_graph_policy_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 3 | 50 | 0.980 | 164.1 | 0.000 | 0.0 | 28.3 |
| gas_graph_policy_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 4 | 50 | 1.000 | 245.5 | 0.000 | 0.0 | 51.5 |
| gas_graph_policy_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_policy | 5 | 50 | 0.960 | 279.0 | 0.000 | 0.0 | 82.6 |
| gas_graph_policy_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_policy | 1 | 50 | 0.980 | 296.1 | 0.000 | 0.0 | 55.8 |
| gas_graph_policy_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_policy | 2 | 50 | 1.000 | 237.8 | 0.000 | 0.0 | 36.6 |
| gas_graph_policy_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_policy | 3 | 50 | 0.980 | 160.9 | 0.000 | 0.0 | 44.5 |
| gas_graph_policy_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_policy | 4 | 50 | 1.000 | 245.2 | 0.000 | 0.0 | 49.9 |
| gas_graph_policy_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_policy | 5 | 50 | 0.980 | 236.8 | 0.000 | 0.0 | 38.2 |
| gas_graph_policy_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 100 | 0.960 | 195.6 | 0.000 | 0.0 | 38.3 |
| gas_graph_policy_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 1 | 42 | gas_graph_policy | 3 | 100 | 0.950 | 199.3 | 0.000 | 0.0 | 30.3 |
| gas_graph_policy_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 3 | 100 | 0.980 | 172.2 | 0.000 | 0.0 | 42.8 |
| gas_graph_policy_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 1 | 43 | gas_graph_policy | 3 | 100 | 0.970 | 178.4 | 0.000 | 0.0 | 33.6 |
| gas_graph_policy_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 50 | 0.960 | 201.1 | 0.000 | 0.0 | 30.0 |
| gas_graph_policy_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_policy | 3 | 50 | 0.960 | 189.5 | 0.000 | 0.0 | 32.6 |
| gas_graph_policy_tasks3_5_ep50_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 3 | 50 | 0.820 | 828.1 | 0.000 | 0.0 | 42.7 |
| gas_graph_policy_tasks3_5_ep50_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_policy | 5 | 50 | 0.680 | 748.6 | 0.000 | 0.0 | 26.5 |
| gas_graph_policy_tasks3_5_ep50_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 3 | 50 | 0.740 | 819.2 | 0.000 | 0.0 | 53.0 |
| gas_graph_policy_tasks3_5_ep50_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_policy | 5 | 50 | 0.920 | 476.0 | 0.000 | 0.0 | 38.8 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.800 | 851.3 | 0.000 | 0.8 | 31.8 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.700 | 766.4 | 0.000 | 1.9 | 31.4 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.700 | 874.3 | 0.000 | 5.0 | 42.8 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.900 | 721.9 | 0.000 | 1.0 | 31.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 0.700 | 599.5 | 0.000 | 1.1 | 16.4 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.400 | 943.7 | 0.000 | 2.2 | 29.1 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.900 | 744.1 | 0.000 | 2.4 | 46.5 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.800 | 814.8 | 0.000 | 0.9 | 27.6 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.900 | 684.7 | 0.000 | 1.3 | 26.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 1.000 | 444.5 | 0.000 | 1.4 | 35.2 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 196.7 | 0.000 | 1.7 | 56.6 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 217.5 | 0.000 | 2.6 | 55.6 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.940 | 218.2 | 0.000 | 2.3 | 63.3 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.560 | 576.0 | 0.000 | 9.9 | 161.6 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.700 | 414.4 | 0.000 | 7.4 | 93.4 |
| gas_graph_tmd_exec_rescue_s125_p7_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 189.7 | 0.000 | 1.7 | 52.6 |
| gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.970 | 191.2 | 0.000 | 1.3 | 38.4 |
| gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.960 | 186.6 | 0.000 | 1.4 | 51.0 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.970 | 191.4 | 0.000 | 1.4 | 39.7 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 1 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.950 | 199.2 | 0.000 | 1.1 | 29.4 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.990 | 162.3 | 0.000 | 1.4 | 34.1 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 1 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.970 | 179.5 | 0.000 | 1.4 | 33.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale10_cc_sticky25_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.920 | 376.8 | 0.000 | 1.3 | 62.2 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.900 | 387.1 | 0.000 | 1.4 | 53.1 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 1.000 | 220.0 | 0.000 | 2.0 | 28.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 1.000 | 141.2 | 0.000 | 1.2 | 26.3 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 1.000 | 246.7 | 0.000 | 2.3 | 50.7 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 1.000 | 236.8 | 0.000 | 2.9 | 56.1 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.920 | 376.1 | 0.000 | 2.1 | 62.8 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 50 | 1.000 | 216.8 | 0.000 | 1.6 | 26.9 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 166.8 | 0.000 | 1.4 | 31.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 50 | 1.000 | 249.8 | 0.000 | 2.2 | 54.9 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50 | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 50 | 0.980 | 267.9 | 0.000 | 3.0 | 87.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 1.000 | 283.9 | 0.000 | 1.7 | 42.1 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 2 | 50 | 1.000 | 231.5 | 0.000 | 1.8 | 34.6 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 1.000 | 146.2 | 0.000 | 1.3 | 30.3 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 4 | 50 | 1.000 | 244.2 | 0.000 | 1.8 | 50.8 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_gasseed43 | antmaze-medium-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 5 | 50 | 0.980 | 243.3 | 0.000 | 2.0 | 42.8 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.900 | 372.2 | 0.000 | 1.8 | 60.9 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 50 | 1.000 | 218.7 | 0.000 | 1.6 | 29.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 1.000 | 147.5 | 0.000 | 1.5 | 30.9 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 50 | 1.000 | 244.5 | 0.000 | 2.1 | 50.7 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 50 | 0.960 | 294.9 | 0.000 | 3.0 | 92.2 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_n512_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.940 | 354.0 | 0.000 | 1.4 | 55.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_nosticky_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.920 | 384.9 | 0.000 | 2.1 | 70.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_tmdactor_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.920 | 386.0 | 0.000 | 2.2 | 72.2 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_nocc_sticky25_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.920 | 361.2 | 0.000 | 1.7 | 48.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 199.8 | 0.000 | 1.2 | 29.5 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 186.7 | 0.000 | 1.3 | 31.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale20_cc_sticky25_task1_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 50 | 0.920 | 370.0 | 0.000 | 1.6 | 57.0 |
| gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 186.9 | 0.000 | 1.3 | 31.3 |
| gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 187.9 | 0.000 | 1.1 | 32.3 |
| gas_graph_tmd_exec_rescue_s200_p7_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 177.1 | 0.000 | 1.6 | 33.3 |
| gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 188.5 | 0.000 | 1.1 | 33.1 |
| gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 187.8 | 0.000 | 1.2 | 31.5 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.980 | 183.4 | 0.000 | 1.2 | 32.3 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 100 | 0.970 | 182.4 | 0.000 | 1.4 | 45.4 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 188.4 | 0.000 | 1.3 | 31.2 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 169.5 | 0.000 | 1.3 | 31.6 |
| gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.940 | 214.8 | 0.000 | 1.2 | 43.8 |
| gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 172.1 | 0.000 | 1.1 | 33.4 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 18.1 | 0.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 13.8 | 0.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.000 | 1000.0 | 0.000 | 15.9 | 17.5 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 4.7 | 0.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 0.700 | 605.2 | 0.000 | 2.7 | 21.1 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 14.2 | 0.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 19.2 | 7.5 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.100 | 992.0 | 0.000 | 12.8 | 39.5 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 2.9 | 0.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 1.000 | 439.9 | 0.000 | 1.2 | 30.4 |
| gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 203.8 | 0.000 | 1.1 | 32.8 |
| gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 170.0 | 0.000 | 1.1 | 30.6 |
| gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.960 | 199.0 | 0.000 | 1.2 | 29.7 |
| gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.980 | 172.0 | 0.000 | 1.2 | 33.5 |
| gas_graph_tmd_exec_rescue_s50_p7_scale15_cc_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 50 | 0.820 | 409.8 | 0.000 | 3.7 | 135.3 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.800 | 862.0 | 0.000 | 1.4 | 43.8 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.800 | 740.7 | 0.000 | 1.9 | 33.3 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.700 | 849.3 | 0.000 | 1.4 | 14.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 1.000 | 684.0 | 0.000 | 1.2 | 34.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 0.600 | 665.5 | 0.000 | 1.2 | 13.4 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.400 | 931.0 | 0.000 | 1.2 | 18.5 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.800 | 772.9 | 0.000 | 1.2 | 75.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.800 | 815.9 | 0.000 | 1.2 | 27.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.800 | 730.9 | 0.000 | 0.9 | 27.1 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 1.000 | 441.2 | 0.000 | 1.0 | 31.9 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.800 | 860.1 | 0.000 | 1.2 | 41.7 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.700 | 773.2 | 0.000 | 1.4 | 66.5 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.600 | 885.4 | 0.000 | 1.2 | 21.5 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.900 | 728.7 | 0.000 | 1.1 | 31.4 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 0.500 | 725.6 | 0.000 | 0.7 | 14.7 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.500 | 936.2 | 0.000 | 0.5 | 18.6 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.900 | 733.0 | 0.000 | 1.2 | 34.6 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.800 | 807.3 | 0.000 | 0.5 | 19.9 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.800 | 721.3 | 0.000 | 0.8 | 24.4 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 1.000 | 442.7 | 0.000 | 2.3 | 33.8 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.800 | 860.1 | 0.000 | 0.8 | 39.5 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.700 | 778.0 | 0.000 | 2.3 | 34.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.600 | 886.1 | 0.000 | 1.3 | 20.9 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.900 | 716.1 | 0.000 | 1.1 | 28.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 0.700 | 617.4 | 0.000 | 0.7 | 14.6 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 1 | 10 | 0.500 | 933.2 | 0.000 | 0.6 | 18.2 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 2 | 10 | 0.900 | 742.7 | 0.000 | 2.9 | 44.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 3 | 10 | 0.800 | 814.3 | 0.000 | 1.1 | 24.7 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 4 | 10 | 0.900 | 693.4 | 0.000 | 1.1 | 32.1 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | gas_graph_tmd_exec_rescue_policy | 5 | 10 | 1.000 | 448.2 | 0.000 | 1.2 | 38.1 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 1 | 20 | 0.000 | 1000.0 | 0.000 | 0.0 | 30.2 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 2 | 20 | 0.850 | 763.9 | 0.000 | 0.0 | 49.0 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 3 | 20 | 0.900 | 800.8 | 0.000 | 0.0 | 77.4 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 4 | 20 | 0.850 | 743.6 | 0.000 | 0.0 | 34.6 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 5 | 20 | 0.450 | 834.6 | 0.000 | 0.0 | 20.8 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 1 | 20 | 0.100 | 990.6 | 0.000 | 0.0 | 40.5 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 2 | 20 | 0.750 | 812.7 | 0.000 | 0.0 | 69.3 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 3 | 20 | 0.800 | 849.1 | 0.000 | 0.0 | 48.3 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 4 | 20 | 0.800 | 815.7 | 0.000 | 0.0 | 36.0 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 5 | 20 | 0.700 | 765.8 | 0.000 | 0.0 | 52.8 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 1 | 20 | 0.400 | 963.7 | 0.000 | 0.0 | 93.7 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 2 | 20 | 0.950 | 700.6 | 0.000 | 0.0 | 44.2 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 3 | 20 | 0.950 | 799.9 | 0.000 | 0.0 | 54.1 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 4 | 20 | 1.000 | 660.2 | 0.000 | 0.0 | 36.0 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 5 | 20 | 1.000 | 424.5 | 0.000 | 0.0 | 41.5 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 1 | 20 | 0.400 | 963.0 | 0.000 | 0.0 | 75.3 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 2 | 20 | 1.000 | 708.1 | 0.000 | 0.0 | 78.4 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 3 | 20 | 0.850 | 806.5 | 0.000 | 0.0 | 40.0 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 4 | 20 | 1.000 | 658.9 | 0.000 | 0.0 | 32.6 |
| gas_graph_tmdcost_w025_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 5 | 20 | 0.850 | 525.5 | 0.000 | 0.0 | 39.1 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 1 | 10 | 0.100 | 994.6 | 0.000 | 0.0 | 25.6 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 2 | 10 | 0.700 | 812.8 | 0.000 | 0.0 | 26.1 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 3 | 10 | 0.400 | 948.6 | 0.000 | 0.0 | 23.2 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 4 | 10 | 0.900 | 806.3 | 0.000 | 0.0 | 48.9 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 5 | 10 | 0.800 | 693.5 | 0.000 | 0.0 | 33.6 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 1 | 10 | 0.300 | 974.7 | 0.000 | 0.0 | 101.2 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 2 | 10 | 0.900 | 690.5 | 0.000 | 0.0 | 38.8 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 3 | 10 | 0.900 | 836.3 | 0.000 | 0.0 | 75.2 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 4 | 10 | 1.000 | 663.3 | 0.000 | 0.0 | 29.7 |
| gas_graph_tmdcost_w05_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 5 | 10 | 1.000 | 443.1 | 0.000 | 0.0 | 38.0 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 1 | 20 | 0.050 | 996.8 | 0.000 | 0.0 | 28.6 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 2 | 20 | 0.800 | 789.5 | 0.000 | 0.0 | 45.5 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 3 | 20 | 0.300 | 963.1 | 0.000 | 0.0 | 16.9 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 4 | 20 | 0.950 | 761.6 | 0.000 | 0.0 | 39.5 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | gas_graph_tmd_cost_policy | 5 | 20 | 0.650 | 780.0 | 0.000 | 0.0 | 31.2 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 1 | 20 | 0.100 | 996.2 | 0.000 | 0.0 | 27.9 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 2 | 20 | 0.850 | 823.4 | 0.000 | 0.0 | 68.8 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 3 | 20 | 0.250 | 983.4 | 0.000 | 0.0 | 21.0 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 4 | 20 | 0.900 | 785.9 | 0.000 | 0.0 | 41.6 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed42 | antmaze-giant-navigate-v0 | 1 | 42 | gas_graph_tmd_cost_policy | 5 | 20 | 0.650 | 800.8 | 0.000 | 0.0 | 34.9 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 1 | 20 | 0.300 | 977.9 | 0.000 | 0.0 | 70.6 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 2 | 20 | 0.850 | 724.5 | 0.000 | 0.0 | 51.7 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 3 | 20 | 0.900 | 840.4 | 0.000 | 0.0 | 45.0 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 4 | 20 | 0.950 | 683.0 | 0.000 | 0.0 | 32.1 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | gas_graph_tmd_cost_policy | 5 | 20 | 1.000 | 441.4 | 0.000 | 0.0 | 44.6 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 1 | 20 | 0.350 | 953.6 | 0.000 | 0.0 | 70.4 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 2 | 20 | 0.950 | 716.8 | 0.000 | 0.0 | 58.6 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 3 | 20 | 0.700 | 875.5 | 0.000 | 0.0 | 53.5 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 4 | 20 | 0.900 | 721.6 | 0.000 | 0.0 | 33.9 |
| gas_graph_tmdcost_w05_alltasks_ep20_gasseed43 | antmaze-giant-navigate-v0 | 1 | 43 | gas_graph_tmd_cost_policy | 5 | 20 | 1.000 | 440.0 | 0.000 | 0.0 | 40.9 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.940 | 388.5 | 0.000 | 7.6 | 75.0 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.920 | 415.5 | 0.000 | 7.5 | 70.5 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.200 | 920.4 | 0.000 | 1.9 | 16.4 |
| tmd100keff_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.060 | 964.7 | 0.000 | 1.1 | 4.2 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 8.3 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 10 | 0.800 | 819.2 | 0.000 | 86.0 | 116.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 10 | 0.000 | 1000.0 | 0.000 | 46.8 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 5.2 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 10 | 0.700 | 711.6 | 0.000 | 70.6 | 81.9 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 10 | 0.300 | 961.7 | 0.000 | 115.7 | 37.1 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 10 | 0.300 | 987.2 | 0.000 | 121.7 | 49.5 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 119.1 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 10 | 0.400 | 954.8 | 0.000 | 136.8 | 32.1 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 7.2 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 42.1 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 10 | 0.000 | 1000.0 | 0.000 | 68.7 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 5 | 10 | 0.500 | 852.0 | 0.000 | 80.4 | 55.4 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 2 | 10 | 0.400 | 898.1 | 0.000 | 79.3 | 25.8 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 10 | 0.400 | 928.9 | 0.000 | 78.9 | 39.4 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 4 | 10 | 0.900 | 831.3 | 0.000 | 76.4 | 59.2 |
| tmd100keff_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 5 | 10 | 0.600 | 755.4 | 0.000 | 49.8 | 52.8 |
| tmd150keff_teall_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.940 | 248.0 | 0.000 | 5.9 | 83.0 |
| tmd150keff_teall_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.980 | 221.3 | 0.000 | 6.0 | 87.8 |
| tmd150keff_teall_exec_gasphi_cc_scale18_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.920 | 287.4 | 0.000 | 6.0 | 140.9 |
| tmd150keff_teall_exec_gasphi_cc_scale18_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.920 | 251.4 | 0.000 | 5.2 | 74.9 |
| tmd200keff_teall_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.920 | 315.4 | 0.000 | 7.5 | 104.4 |
| tmd200keff_teall_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.880 | 336.3 | 0.000 | 7.2 | 90.3 |
| tmd50k_exec_gasphi_cc_final4_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 2 | 1.000 | 670.5 | 0.000 | 6.5 | 70.5 |
| tmd50k_exec_gasphi_cc_nosticky_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 2 | 0.500 | 599.5 | 0.000 | 10.0 | 29.5 |
| tmd50k_exec_gasphi_cc_scale10_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.020 | 998.9 | 0.000 | 5.4 | 5.9 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 1 | 10 | 0.600 | 784.6 | 0.000 | 8.4 | 254.6 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 3.5 | 0.0 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 10 | 1.000 | 246.6 | 0.000 | 6.3 | 76.6 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 7.7 | 0.0 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 10 | 0.900 | 479.9 | 0.000 | 13.1 | 144.9 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 1 | 20 | 0.050 | 999.9 | 0.000 | 9.2 | 8.7 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 20 | 0.950 | 420.0 | 0.000 | 11.2 | 107.5 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 20 | 0.900 | 344.4 | 0.000 | 5.8 | 83.2 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 4 | 20 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 20 | 0.950 | 352.4 | 0.000 | 11.1 | 74.9 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 1 | 20 | 0.050 | 972.4 | 0.000 | 8.6 | 7.4 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 2 | 20 | 0.900 | 474.1 | 0.000 | 12.4 | 44.1 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 20 | 1.000 | 279.0 | 0.000 | 6.8 | 71.5 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 4 | 20 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd50k_exec_gasphi_cc_scale15_alltasks_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 5 | 20 | 0.900 | 368.8 | 0.000 | 10.3 | 116.2 |
| tmd50k_exec_gasphi_cc_scale15_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 2 | 1.000 | 220.0 | 0.000 | 6.5 | 57.5 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.920 | 336.3 | 0.000 | 5.8 | 78.3 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.940 | 301.5 | 0.000 | 6.4 | 84.5 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.920 | 410.6 | 0.000 | 7.3 | 118.6 |
| tmd50k_exec_gasphi_cc_scale15_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 1.000 | 301.8 | 0.000 | 6.5 | 92.3 |
| tmd50k_exec_gasphi_cc_scale16_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.920 | 338.5 | 0.000 | 6.4 | 117.0 |
| tmd50k_exec_gasphi_cc_scale17_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.940 | 316.9 | 0.000 | 6.2 | 153.4 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 20 | 0.900 | 288.1 | 0.000 | 5.2 | 125.7 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 20 | 0.950 | 301.6 | 0.000 | 4.5 | 127.8 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.980 | 251.6 | 0.000 | 5.5 | 127.6 |
| tmd50k_exec_gasphi_cc_scale18_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.980 | 230.8 | 0.000 | 5.3 | 104.3 |
| tmd50k_exec_gasphi_cc_scale19_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.940 | 308.4 | 0.000 | 5.3 | 124.4 |
| tmd50k_exec_gasphi_cc_scale20_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 2 | 1.000 | 235.0 | 0.000 | 5.0 | 97.5 |
| tmd50k_exec_gasphi_cc_scale20_task3_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 20 | 0.600 | 610.0 | 0.000 | 6.4 | 413.7 |
| tmd50k_exec_gasphi_cc_scale20_task3_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 20 | 0.700 | 486.2 | 0.000 | 4.0 | 254.9 |
| tmd50k_exec_gasphi_cc_scale20_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.940 | 287.3 | 0.000 | 5.1 | 120.3 |
| tmd50k_exec_gasphi_cc_scale21_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.960 | 242.8 | 0.000 | 4.5 | 145.8 |
| tmd50k_exec_gasphi_cc_scale22_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.960 | 251.2 | 0.000 | 4.4 | 167.7 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep20_gasseed42 | antmaze-medium-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 20 | 0.250 | 894.5 | 0.000 | 8.1 | 652.0 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep20_gasseed43 | antmaze-medium-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 20 | 0.400 | 812.6 | 0.000 | 8.5 | 453.9 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.820 | 446.3 | 0.000 | 5.4 | 298.3 |
| tmd50k_exec_gasphi_cc_scale25_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.780 | 456.5 | 0.000 | 4.5 | 316.0 |
| tmd50k_exec_gasphi_cc_scale30_task3_ep50_gasseed43_log | antmaze-medium-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.020 | 988.1 | 0.000 | 5.9 | 876.1 |
| tmd50k_exec_gasphi_cc_scale30_task3_ep50_log | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.000 | 1000.0 | 0.000 | 1.7 | 908.5 |
| tmd50k_exec_gasphi_cc_sticky25_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 2 | 0.500 | 735.5 | 0.000 | 7.5 | 23.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 27.8 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 10 | 0.000 | 1000.0 | 0.000 | 0.8 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 19.9 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-stitch-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 10 | 0.400 | 958.7 | 0.000 | 16.7 | 31.2 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 26.6 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 10 | 0.000 | 1000.0 | 0.000 | 0.8 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 26.1 | 0.0 |
| tmd50k_n512_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-stitch-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 5 | 10 | 0.200 | 959.7 | 0.000 | 6.1 | 9.7 |
| tmd50k_q75_t90_sticky25_tmd_graph_gas_policy_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 3 | 2 | 0.000 | 1000.0 | 0.010 | 23.0 | 0.0 |
| tmd50k_q75_t90_sticky25_tmd_graph_gas_policy_task3_ep10 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 3 | 10 | 0.000 | 1000.0 | 0.015 | 15.3 | 0.0 |
| tmd50k_q75_t90_sticky25_tmd_graph_tmd_actor_task3 | antmaze-medium-stitch-v0 | 0 |  | tmd_graph_tmd_actor | 3 | 2 | 0.000 | 1000.0 | 0.000 | 17.5 | 0.0 |
| tmd50k_q75_t90_tmd_graph_gas_policy_task3 | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 3 | 2 | 0.000 | 1000.0 | 0.019 | 343.5 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 106.9 | 134.7 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 0.0 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 10 | 0.800 | 894.8 | 0.000 | 98.1 | 68.4 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 4 | 10 | 0.400 | 946.9 | 0.000 | 94.7 | 74.6 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 10 | 0.900 | 594.5 | 0.000 | 66.2 | 107.5 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 1 | 10 | 0.100 | 993.1 | 0.000 | 123.3 | 116.9 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 5.2 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 10 | 0.900 | 815.6 | 0.000 | 100.7 | 113.7 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 4 | 10 | 0.300 | 975.3 | 0.000 | 80.3 | 28.4 |
| tmd50k_q98_exec_gasphi_cc_scale15_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 5 | 10 | 0.800 | 626.2 | 0.000 | 65.1 | 79.5 |
| tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 50 | 0.640 | 894.6 | 0.000 | 93.3 | 63.5 |
| tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 50 | 0.820 | 643.4 | 0.000 | 74.0 | 101.8 |
| tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 50 | 0.620 | 880.6 | 0.000 | 98.0 | 83.5 |
| tmd50k_q98_exec_gasphi_cc_scale15_tasks3_5_ep50_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 5 | 50 | 0.620 | 743.8 | 0.000 | 71.9 | 102.1 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 89.7 | 48.4 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 108.6 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 3 | 10 | 0.000 | 1000.0 | 0.000 | 75.3 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 44.3 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed42 | antmaze-giant-navigate-v0 | 0 | 42 | tmd_exec_graph_gas_policy | 5 | 10 | 0.000 | 1000.0 | 0.000 | 86.8 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 1 | 10 | 0.000 | 1000.0 | 0.000 | 113.4 | 134.8 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 2 | 10 | 0.000 | 1000.0 | 0.000 | 114.8 | 17.6 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 3 | 10 | 0.000 | 1000.0 | 0.000 | 78.1 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 4 | 10 | 0.000 | 1000.0 | 0.000 | 45.7 | 0.0 |
| tmd50k_q98_exec_gasphi_cc_scale30_alltasks_ep10_gasseed43 | antmaze-giant-navigate-v0 | 0 | 43 | tmd_exec_graph_gas_policy | 5 | 10 | 0.000 | 1000.0 | 0.000 | 68.9 | 0.0 |
| tmd50k_q98_graph_gas_policy_alltasks_ep2_gasseed42_gpu | antmaze-giant-navigate-v0 | 0 | 42 | tmd_graph_gas_policy | 1 | 2 | 0.000 | 1000.0 | 1.000 | 0.0 | 0.0 |
| tmd50k_q98_graph_gas_policy_alltasks_ep2_gasseed42_gpu | antmaze-giant-navigate-v0 | 0 | 42 | tmd_graph_gas_policy | 2 | 2 | 0.000 | 1000.0 | 0.214 | 175.5 | 45.5 |
| tmd50k_q98_graph_gas_policy_alltasks_ep2_gasseed42_gpu | antmaze-giant-navigate-v0 | 0 | 42 | tmd_graph_gas_policy | 3 | 2 | 0.000 | 1000.0 | 0.000 | 261.0 | 103.0 |
| tmd50k_q98_graph_gas_policy_alltasks_ep2_gasseed42_gpu | antmaze-giant-navigate-v0 | 0 | 42 | tmd_graph_gas_policy | 4 | 2 | 0.000 | 1000.0 | 1.000 | 0.0 | 0.0 |
| tmd50k_q98_graph_gas_policy_alltasks_ep2_gasseed42_gpu | antmaze-giant-navigate-v0 | 0 | 42 | tmd_graph_gas_policy | 5 | 2 | 0.000 | 1000.0 | 1.000 | 0.0 | 0.0 |
| tmd50k_q98_graph_tmd_actor_alltasks_ep2_gpu | antmaze-giant-navigate-v0 | 0 |  | tmd_graph_tmd_actor | 1 | 2 | 0.000 | 1000.0 | 1.000 | 0.0 | 0.0 |
| tmd50k_q98_graph_tmd_actor_alltasks_ep2_gpu | antmaze-giant-navigate-v0 | 0 |  | tmd_graph_tmd_actor | 2 | 2 | 0.000 | 1000.0 | 0.615 | 130.0 | 0.0 |
| tmd50k_q98_graph_tmd_actor_alltasks_ep2_gpu | antmaze-giant-navigate-v0 | 0 |  | tmd_graph_tmd_actor | 3 | 2 | 0.000 | 1000.0 | 0.821 | 57.0 | 0.0 |
| tmd50k_q98_graph_tmd_actor_alltasks_ep2_gpu | antmaze-giant-navigate-v0 | 0 |  | tmd_graph_tmd_actor | 4 | 2 | 0.000 | 1000.0 | 1.000 | 0.0 | 0.0 |
| tmd50k_q98_graph_tmd_actor_alltasks_ep2_gpu | antmaze-giant-navigate-v0 | 0 |  | tmd_graph_tmd_actor | 5 | 2 | 0.000 | 1000.0 | 1.000 | 0.0 | 0.0 |
| tmd_graph_gas_policy | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 1 | 2 | 1.000 | 651.5 | 0.000 | 65.0 | 651.5 |
| tmd_graph_gas_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 1 | 2 | 1.000 | 419.5 | 0.000 | 45.0 | 419.5 |
| tmd_graph_gas_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 2 | 2 | 1.000 | 316.5 | 0.000 | 78.0 | 316.5 |
| tmd_graph_gas_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 3 | 2 | 0.000 | 1000.0 | 0.000 | 104.0 | 1000.0 |
| tmd_graph_gas_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 4 | 2 | 0.000 | 1000.0 | 0.000 | 291.0 | 1000.0 |
| tmd_graph_gas_policy_alltasks | antmaze-medium-stitch-v0 | 0 | 42 | tmd_graph_gas_policy | 5 | 2 | 1.000 | 440.5 | 0.000 | 61.5 | 440.5 |
| tmd_graph_tmd_actor | antmaze-medium-stitch-v0 | 0 |  | tmd_graph_tmd_actor | 1 | 2 | 0.000 | 1000.0 | 0.000 | 2.0 | 0.0 |

## Hybrid rescue paired comparison

| env | seed | gas_seed | comparison | rows | gas_success | hybrid_success | delta | solved | regressed | per_task_delta |
|---|---|---|---|---|---|---|---|---|---|---|
| antmaze-giant-stitch-v0 | 0 | 42 | alltasks_ep10_gasseed42::gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | 50 | 39.0 | 38.0 | -1.0 | 1 | 2 | task1:+0, task2:-1, task3:+0, task4:-1, task5:+1 |
| antmaze-giant-stitch-v0 | 0 | 42 | alltasks_ep10_gasseed42::gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | 50 | 39.0 | 7.0 | -32.0 | 2 | 34 | task1:-8, task2:-8, task3:-7, task4:-10, task5:+1 |
| antmaze-giant-stitch-v0 | 0 | 42 | alltasks_ep10_gasseed42::gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | 50 | 39.0 | 39.0 | +0.0 | 0 | 0 | task1:+0, task2:+0, task3:+0, task4:+0, task5:+0 |
| antmaze-giant-stitch-v0 | 0 | 42 | alltasks_ep10_gasseed42::gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | 50 | 39.0 | 35.0 | -4.0 | 0 | 4 | task1:+0, task2:-1, task3:-1, task4:-1, task5:-1 |
| antmaze-giant-stitch-v0 | 0 | 42 | alltasks_ep10_gasseed42::gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | 50 | 39.0 | 37.0 | -2.0 | 1 | 3 | task1:+0, task2:-1, task3:-1, task4:-1, task5:+1 |
| antmaze-giant-stitch-v0 | 0 | 43 | alltasks_ep10_gasseed43::gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | 50 | 39.0 | 40.0 | +1.0 | 3 | 2 | task1:+0, task2:+1, task3:+0, task4:+0, task5:+0 |
| antmaze-giant-stitch-v0 | 0 | 43 | alltasks_ep10_gasseed43::gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | 50 | 39.0 | 11.0 | -28.0 | 0 | 28 | task1:-4, task2:-8, task3:-7, task4:-9, task5:+0 |
| antmaze-giant-stitch-v0 | 0 | 43 | alltasks_ep10_gasseed43::gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | 50 | 39.0 | 38.0 | -1.0 | 2 | 3 | task1:+0, task2:+0, task3:+0, task4:-1, task5:+0 |
| antmaze-giant-stitch-v0 | 0 | 43 | alltasks_ep10_gasseed43::gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | 50 | 39.0 | 40.0 | +1.0 | 2 | 1 | task1:+1, task2:+1, task3:+0, task4:-1, task5:+0 |
| antmaze-giant-stitch-v0 | 0 | 43 | alltasks_ep10_gasseed43::gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | 50 | 39.0 | 41.0 | +2.0 | 2 | 0 | task1:+1, task2:+1, task3:+0, task4:+0, task5:+0 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep100_gasseed42::gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed42 | 100 | 96.0 | 97.0 | +1.0 | 2 | 1 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep100_gasseed42::gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42 | 100 | 96.0 | 97.0 | +1.0 | 2 | 1 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep100_gasseed42::gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed42 | 100 | 96.0 | 98.0 | +2.0 | 2 | 0 | task3:+2 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 48.0 | +0.0 | 1 | 1 | task3:+0 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed42 | 50 | 48.0 | 28.0 | -20.0 | 0 | 20 | task3:-20 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 48.0 | +0.0 | 0 | 0 | task3:+0 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 49.0 | +1.0 | 1 | 0 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 49.0 | +1.0 | 1 | 0 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 49.0 | +1.0 | 1 | 0 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 47.0 | -1.0 | 0 | 1 | task3:-1 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 48.0 | +0.0 | 0 | 0 | task3:+0 |
| antmaze-medium-navigate-v0 | 0 | 42 | task3_ep50_gasseed42::gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed42 | 50 | 48.0 | 48.0 | +0.0 | 0 | 0 | task3:+0 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep100_gasseed43::gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed43 | 100 | 98.0 | 96.0 | -2.0 | 1 | 3 | task3:-2 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep100_gasseed43::gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43 | 100 | 98.0 | 99.0 | +1.0 | 1 | 0 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep100_gasseed43::gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed43 | 100 | 98.0 | 97.0 | -1.0 | 1 | 2 | task3:-1 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 47.0 | -1.0 | 2 | 3 | task3:-1 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed43 | 50 | 48.0 | 35.0 | -13.0 | 1 | 14 | task3:-13 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 48.0 | +0.0 | 1 | 1 | task3:+0 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 48.0 | +0.0 | 1 | 1 | task3:+0 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 48.0 | +0.0 | 1 | 1 | task3:+0 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 49.0 | +1.0 | 1 | 0 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 49.0 | +1.0 | 1 | 0 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 49.0 | +1.0 | 2 | 1 | task3:+1 |
| antmaze-medium-navigate-v0 | 0 | 43 | task3_ep50_gasseed43::gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed43 | 50 | 48.0 | 49.0 | +1.0 | 1 | 0 | task3:+1 |
| antmaze-medium-navigate-v0 | 1 | 42 | task3_ep100_gasseed42::gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42 | 100 | 95.0 | 95.0 | +0.0 | 0 | 0 | task3:+0 |
| antmaze-medium-navigate-v0 | 1 | 43 | task3_ep100_gasseed43::gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43 | 100 | 97.0 | 97.0 | +0.0 | 1 | 1 | task3:+0 |
| antmaze-medium-stitch-v0 | 0 | 42 | alltasks_ep10::gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep10 | 50 | 48.0 | 49.0 | +1.0 | 1 | 0 | task1:+0, task2:+0, task3:+0, task4:+0, task5:+1 |
| antmaze-medium-stitch-v0 | 0 | 42 | alltasks_ep50::gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50 | 250 | 242.0 | 244.0 | +2.0 | 3 | 1 | task1:+0, task2:+1, task3:+0, task4:+0, task5:+1 |
| antmaze-medium-stitch-v0 | 0 | 42 | alltasks_ep50::gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | 250 | 242.0 | 243.0 | +1.0 | 3 | 2 | task1:-1, task2:+1, task3:+1, task4:+0, task5:+0 |
| antmaze-medium-stitch-v0 | 0 | 43 | alltasks_ep50_gasseed43::gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_gasseed43 | 250 | 247.0 | 249.0 | +2.0 | 2 | 0 | task1:+1, task2:+0, task3:+1, task4:+0, task5:+0 |

## Rescue activation audit

| run_name | task_id | rows | success | activated | activated_success | mean_rescue_steps |
|---|---|---|---|---|---|---|
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | 1 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | 2 | 10 | 7.0 | 2 | 0.0 | 100.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | 3 | 10 | 7.0 | 3 | 0.0 | 98.3 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | 4 | 10 | 9.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed42 | 5 | 10 | 7.0 | 1 | 0.0 | 100.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | 1 | 10 | 4.0 | 2 | 0.0 | 75.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | 2 | 10 | 9.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | 3 | 10 | 8.0 | 1 | 0.0 | 100.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | 4 | 10 | 9.0 | 1 | 0.0 | 100.0 |
| gas_graph_tmd100keff_q98_rescue_s700_p20_b50_r3_alltasks_ep10_gasseed43 | 5 | 10 | 10.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_cc_task3_ep50_log | 3 | 50 | 48.0 | 11 | 10.0 | 215.5 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed42 | 3 | 50 | 48.0 | 30 | 28.0 | 169.8 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_task3_ep50_gasseed43 | 3 | 50 | 47.0 | 18 | 16.0 | 206.0 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed42 | 3 | 50 | 28.0 | 29 | 7.0 | 790.2 |
| gas_graph_tmd_exec_rescue_s100_p5_scale15_tmdactor_task3_ep50_gasseed43 | 3 | 50 | 35.0 | 18 | 3.0 | 798.8 |
| gas_graph_tmd_exec_rescue_s125_p7_scale15_cc_task3_ep50_log | 3 | 50 | 48.0 | 2 | 1.0 | 615.0 |
| gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed42 | 3 | 100 | 97.0 | 4 | 1.0 | 705.8 |
| gas_graph_tmd_exec_rescue_s150_p9_scale15_task3_ep100_gasseed43 | 3 | 100 | 96.0 | 2 | 0.0 | 850.0 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed42 | 3 | 200 | 192.0 | 7 | 1.0 | 740.1 |
| gas_graph_tmd_exec_rescue_s175_p9_scale15_task3_ep100_gasseed43 | 3 | 200 | 196.0 | 4 | 0.0 | 825.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale10_cc_sticky25_task1_ep50_log | 1 | 50 | 46.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | 1 | 50 | 45.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | 2 | 50 | 50.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | 3 | 50 | 50.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | 4 | 50 | 50.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_alltasks_ep50_rescue_log | 5 | 50 | 48.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_n512_task1_ep50_log | 1 | 50 | 47.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_nosticky_task1_ep50_log | 1 | 50 | 46.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_cc_tmdactor_task1_ep50_log | 1 | 50 | 46.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_nocc_sticky25_task1_ep50_log | 1 | 50 | 46.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed42 | 3 | 50 | 48.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale15_task3_ep50_gasseed43 | 3 | 50 | 48.0 | 1 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p15_scale20_cc_sticky25_task1_ep50_log | 1 | 50 | 46.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed42 | 3 | 50 | 49.0 | 3 | 2.0 | 354.3 |
| gas_graph_tmd_exec_rescue_s200_p6_scale15_task3_ep50_gasseed43 | 3 | 50 | 48.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p7_scale15_cc_task3_ep50_log | 3 | 50 | 49.0 | 2 | 1.0 | 709.0 |
| gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed42 | 3 | 50 | 49.0 | 2 | 1.0 | 486.5 |
| gas_graph_tmd_exec_rescue_s200_p8_scale15_task3_ep50_gasseed43 | 3 | 50 | 48.0 | 2 | 0.0 | 794.5 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed42 | 3 | 100 | 98.0 | 3 | 1.0 | 592.7 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep100_gasseed43 | 3 | 100 | 97.0 | 2 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed42 | 3 | 50 | 49.0 | 2 | 1.0 | 475.5 |
| gas_graph_tmd_exec_rescue_s200_p9_scale15_task3_ep50_gasseed43 | 3 | 50 | 49.0 | 1 | 0.0 | 800.0 |
| gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed42 | 3 | 50 | 47.0 | 2 | 0.0 | 750.0 |
| gas_graph_tmd_exec_rescue_s250_p6_scale15_task3_ep50_gasseed43 | 3 | 50 | 49.0 | 1 | 0.0 | 750.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | 1 | 10 | 0.0 | 10 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | 2 | 10 | 0.0 | 10 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | 3 | 10 | 0.0 | 10 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | 4 | 10 | 0.0 | 10 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed42 | 5 | 10 | 7.0 | 2 | 0.0 | 698.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | 1 | 10 | 0.0 | 10 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | 2 | 10 | 0.0 | 10 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | 3 | 10 | 1.0 | 10 | 1.0 | 692.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | 4 | 10 | 0.0 | 10 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p20_scale15_n512_alltasks_ep10_gasseed43 | 5 | 10 | 10.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed42 | 3 | 50 | 48.0 | 2 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s300_p5_scale15_task3_ep50_gasseed43 | 3 | 50 | 49.0 | 1 | 0.0 | 700.0 |
| gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed42 | 3 | 50 | 48.0 | 2 | 0.0 | 600.0 |
| gas_graph_tmd_exec_rescue_s400_p5_scale15_task3_ep50_gasseed43 | 3 | 50 | 49.0 | 1 | 0.0 | 600.0 |
| gas_graph_tmd_exec_rescue_s50_p7_scale15_cc_task3_ep50_log | 3 | 50 | 41.0 | 50 | 41.0 | 359.8 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | 1 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | 2 | 10 | 8.0 | 2 | 0.0 | 300.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | 3 | 10 | 7.0 | 3 | 0.0 | 300.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | 4 | 10 | 10.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed42 | 5 | 10 | 6.0 | 1 | 0.0 | 300.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | 1 | 10 | 4.0 | 2 | 0.0 | 296.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | 2 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | 3 | 10 | 8.0 | 1 | 0.0 | 300.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | 4 | 10 | 8.0 | 1 | 0.0 | 300.0 |
| gas_graph_tmd_exec_rescue_s700_p20_scale15_n512_alltasks_ep10_gasseed43 | 5 | 10 | 10.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | 1 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | 2 | 10 | 7.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | 3 | 10 | 6.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | 4 | 10 | 9.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed42 | 5 | 10 | 5.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | 1 | 10 | 5.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | 2 | 10 | 9.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | 3 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | 4 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_final_actor_after50_alltasks_ep10_gasseed43 | 5 | 10 | 10.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | 1 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | 2 | 10 | 7.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | 3 | 10 | 6.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | 4 | 10 | 9.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed42 | 5 | 10 | 7.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | 1 | 10 | 5.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | 2 | 10 | 9.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | 3 | 10 | 8.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | 4 | 10 | 9.0 | 0 | 0.0 | 0.0 |
| gas_graph_tmd_hybrid_disabled_control_alltasks_ep10_gasseed43 | 5 | 10 | 10.0 | 0 | 0.0 | 0.0 |

## Failure analysis

The comparable GAS graph baseline is `gas_graph_policy`. Pure `tmd_graph_*` rows do not exceed this baseline on success.
`gas_graph_tmd_exec_rescue_policy` is a hybrid rescue policy: it starts from the official GAS graph and switches to the TMD execution graph only when the GAS graph has not entered final-goal mode after the configured delay and the remaining GAS path is still long.
Mechanism-level evidence must come from rows where `tmd_rescue_activated=1`; aggregate success deltas without rescue activation can be explained by stochastic trajectory differences.
Continuation checks found that the original 50k checkpoint has a narrow task3 seed42 signal (`scale15`, 50/50 vs GAS 49/50), but the same setting regresses seed43 and the best cross-seed threshold scan (`scale18`) only ties GAS on task3 aggregate.
Resume training to effective 100k/150k/200k did not validate a stronger TMD backbone: 100k task3 execution collapsed, while 150k/200k automatic TE graphs shrank to very few nodes and TE-relaxed graphs still underperformed GAS.
An IQE-distance TMD run was attempted, but early throughput was about 4.8 steps/s, making a 50k checkpoint a multi-hour job; it was stopped before checkpoint save and is not counted as evidence.
Medium-navigate all-tasks/20 did not validate TMD execution: GAS seed42/43 were 209/300 and 243/300, while TMD-exec scale15 was 57/100 and 57/100; task1/task4 were the main collapses.
Medium-navigate task3/50 rejected the earlier 20-episode positive blip: GAS seed42/43 were 48/50 and 48/50, while TMD-exec scale15 was 46/50 and 47/50.
Medium-navigate effective-100k TMD did not fix task3: TMD-exec scale15 was 47/50 and 46/50 against GAS 48/50 and 48/50.
Medium-navigate rescue also failed to improve task3: s200/p15 matched 48/50 and 48/50 with activated rescues 0 and 1; s100/p5 gave 48/50 and 47/50 with activated rescues 30 and 18.
Using the TMD actor as the rescue low-level controller regressed sharply: 28/50 and 35/50.
The best late-rescue candidate, s175/p9, had a narrow seed0 gain but failed cross-reset validation: eval seed0 GAS was 96/100 and 98/100 versus rescue 97/100 and 99/100; eval seed1 GAS was 95/100 and 97/100 versus rescue 95/100 and 97/100.
Giant-stitch screening exposed useful GAS headroom but rejected the 50k TMD execution graph: GAS seed42/43 were 39/50 and 39/50, while TMD-exec n512 scale15 was 4/50 and 2/50.
Giant-stitch early long-path rescue was destructive: s300/p20 gave 7/50 and 11/50 with activated rescues 42 and 40.
Giant-stitch late rescue preserved one seed but still lacked a positive TMD mechanism: s700/p20 gave 39/50 and 38/50 against GAS 39/50 and 39/50, with activated-rescue successes 0 and 0.
Giant-stitch final-goal TMD actor rescue did not validate: the disabled hybrid control was 37/50 and 41/50, whereas final-actor-after50 was 35/50 and 40/50. The apparent seed43 gain versus GAS was present in the disabled control, so it is not attributable to TMD final rescue.
After fixing calibration to use paired H-step distances, giant-stitch effective-100k q98/t995 remained below GAS: TMD-exec was 10/50 and 23/50 against GAS 39/50 and 39/50.
The bounded effective-100k q98 late-rescue burst also lacked attributable rescue success: 38/50 and 40/50, with activated-rescue successes 0 and 0.
Giant-navigate had larger GAS headroom, but 50k TMD q98/t995 execution did not improve all-task totals: GAS seed42/43 were 27/50 and 36/50, while TMD-exec scale15 was 21/50 and 21/50.
Increasing the execution radius to scale30 was catastrophic on giant-navigate: 0/50 and 0/50.
The apparent task3/task5 ep10 signal failed ep50 confirmation: tasks3,5 GAS was 75/100 and 83/100, whereas TMD-exec was 73/100 and 62/100.
Giant-navigate effective-100k did not repair the 50k instability: TMD-exec q98/t995 was 15/50 and 5/50.
Native TMD graph-path execution was rejected by a GPU smoke test: TMD actor was 0/10 and GAS low-level on TMD graph was 0/10.
TMD-cost shaping on the GAS graph produced the first positive giant-navigate all-task signal at ep10: GAS seed42/43 were 27/50 and 36/50, while TMD-cost w0.5 was 29/50 and 41/50.
The reset-seed0 ep20 confirmation stayed positive but small, so it is promising rather than conclusive: GAS seed42/43 were 54/100 and 78/100, while TMD-cost w0.5 was 55/100 and 80/100.
The reset-seed1 ep20 check was mixed: GAS seed42/43 were 59/100 and 66/100, while TMD-cost w0.5 was 55/100 and 78/100. This suggests the cost weight is useful but currently too blunt.
Across reset seeds 0 and 1, TMD-cost w0.5 remains net positive but not uniform: GAS seed42/43 aggregate were 113/200 and 144/200, while TMD-cost w0.5 was 110/200 and 158/200.
Reducing TMD-cost weight to w0.25 fixed the w0.5 bluntness and validated a robust positive mechanism: reset-seed0 GAS seed42/43 54/100 and 78/100 became 61/100 and 86/100; reset-seed1 GAS seed42/43 59/100 and 66/100 became 63/100 and 82/100. Across both reset seeds, GAS seed42/43 were 113/200 and 144/200, while TMD-cost w0.25 was 124/200 and 168/200.
If evaluation rows are absent, inspect `construct_error.json` or `eval_error.json`.

## Decision

- GO_TMD_COST_SHAPING_GIANT_NAVIGATE_W025

## Next commands

```bash
bash scripts/tmd_test_pilot.sh ENVS=antmaze-medium-stitch-v0 SEEDS=0 EPISODES=2 QUICK=1 MODES=tmd_graph_tmd_actor
```
