# Stage3 Report

## Run Completion
- total runs: 18
- completed: 18
- failed: 0
- terminated: 0
- archived: 18

## Planner Metrics: Nonzero Pairs
_No data._

## Planner Metrics: All Pairs
_No data._

## Edge Diagnostics
_No data._

## Balanced Edge Diagnostics
_No data._

## Boundary Diagnostics
_No data._

## Edge Rollout Diagnostics
_No data._

## Graph Summary
| env | event | num_nodes | num_edges |
| --- | --- | --- | --- |
| antmaze-medium-diverse-v2 | loaded | 300.0 | 3600.0 |
| antmaze-medium-play-v2 | loaded | 300.0 | 3600.0 |

## Eval Summary
| env | variant | success_mean | success_std | return_mean | return_std | steps_mean | steps_std | replans_mean | replans_std | no_path_count_mean | no_path_count_std | last_plan_edges_mean | last_plan_edges_std | goal_distance_final_mean | goal_distance_final_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-medium-diverse-v2 | full_bars | 0.2222 | 0.4181 | 0.2222 | 0.4181 | 616.4667 | 114.5011 | 25.9778 | 16.0763 | 0.2222 | 0.4181 | 2.3 | 2.3438 | 5.0797 | 5.9474 |
| antmaze-medium-diverse-v2 | reachability | 0.1333 | 0.3418 | 0.1333 | 0.3418 | 609.8222 | 115.0658 | 30.7667 | 56.2347 | 0.3444 | 0.4778 | 2.6 | 2.6042 | 5.9769 | 7.0295 |
| antmaze-medium-diverse-v2 | shortest | 0.1222 | 0.3294 | 0.1222 | 0.3294 | 642.7667 | 100.8421 | 25.7333 | 7.7544 | 0.1778 | 0.3845 | 2.7222 | 2.1042 | 8.4863 | 8.1597 |
| antmaze-medium-play-v2 | full_bars | 0.0778 | 0.2693 | 0.0778 | 0.2693 | 637.3444 | 102.2803 | 26.8222 | 11.5909 | 0.2889 | 0.4558 | 3.0222 | 2.6731 | 8.2596 | 8.2157 |
| antmaze-medium-play-v2 | reachability | 0.1333 | 0.3418 | 0.1333 | 0.3418 | 625.8111 | 94.7092 | 31.9 | 38.1805 | 0.3556 | 0.4814 | 2.3 | 2.3293 | 5.0874 | 5.7547 |
| antmaze-medium-play-v2 | shortest | 0.0111 | 0.1054 | 0.0111 | 0.1054 | 685.2111 | 52.3822 | 29.9 | 41.5814 | 0.0889 | 0.2862 | 3.7111 | 1.8618 | 13.5424 | 8.0051 |

## Profile Summary
| env | phase | event | duration_sec |
| --- | --- | --- | --- |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_end | 0.0191 |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_end | 3.0715 |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_policy_end | 0.0054 |
| antmaze-medium-diverse-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_end | 0.0038 |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_end | 1.2985 |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_start | NaN |
| antmaze-medium-play-v2 | pipeline | embed_dataset_end | 0.0233 |
| antmaze-medium-play-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | load_dataset_end | 3.1244 |
| antmaze-medium-play-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_policy_end | 0.0081 |
| antmaze-medium-play-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_reachability_end | 0.0074 |
| antmaze-medium-play-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_tdr_end | 1.3348 |
| antmaze-medium-play-v2 | pipeline | train_tdr_start | NaN |

## Failure Modes
_No data._

## Decision
- decision: EXPAND_EVAL

## Next Actions
- python scripts/collect_csv.py --log-root runs_stage17_online_quick_loaded
- python scripts/analyze_bars_results.py --log-root runs_stage17_online_quick_loaded --stage stage3
- Increase eval episodes or expand to large tasks under the same protocol.
