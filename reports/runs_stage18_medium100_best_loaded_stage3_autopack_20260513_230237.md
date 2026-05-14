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
| env | variant | success_mean | success_std | return_mean | return_std | steps_mean | steps_std | replans_mean | replans_std | no_path_count_mean | no_path_count_std | initial_plan_failed_count_mean | initial_plan_failed_count_std | plan_failed_initial_mean | plan_failed_initial_std | fallback_used_mean | fallback_used_std | fallback_count_mean | fallback_count_std | direct_goal_attempts_mean | direct_goal_attempts_std | last_plan_edges_mean | last_plan_edges_std | first_plan_edges_mean | first_plan_edges_std | max_plan_edges_mean | max_plan_edges_std | mean_plan_edges_mean | mean_plan_edges_std | num_plan_calls_mean | num_plan_calls_std | num_subgoal_attempts_mean | num_subgoal_attempts_std | num_subgoal_reached_mean | num_subgoal_reached_std | subgoal_reach_rate_mean | subgoal_reach_rate_std | goal_distance_final_mean | goal_distance_final_std | subgoal_horizon_mean | subgoal_horizon_std | subgoal_threshold_mean | subgoal_threshold_std | success_threshold_mean | success_threshold_std | lambda_risk_mean | lambda_risk_std | lambda_boundary_mean | lambda_boundary_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-medium-diverse-v2 | full_bars | 0.4133 | 0.4933 | 0.4133 | 0.4933 | 846.57 | 212.6502 | 75.5633 | 149.282 | 0.0 | 0.0 | 5.6067 | 12.813 | 0.47 | 0.4999 | 0.47 | 0.4999 | 16.82 | 38.4389 | 5.6067 | 12.813 | 2.1267 | 2.6873 | 8.4567 | 0.499 | 8.49 | 0.5204 | 3.8748 | 2.0659 | 75.5633 | 149.282 | 75.5633 | 149.282 | 63.0633 | 151.2366 | 0.6258 | 0.2764 | 4.4686 | 6.5442 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-diverse-v2 | reachability | 0.35 | 0.4778 | 0.35 | 0.4778 | 863.6433 | 209.211 | 80.99 | 167.3835 | 0.0 | 0.0 | 5.7267 | 13.2205 | 0.4233 | 0.4949 | 0.4233 | 0.4949 | 11.4533 | 26.441 | 5.7267 | 13.2205 | 2.2933 | 2.6006 | 7.9333 | 0.6908 | 7.9767 | 0.6863 | 3.7667 | 1.9798 | 80.99 | 167.3835 | 80.99 | 167.3835 | 68.3433 | 169.8106 | 0.609 | 0.2864 | 5.4019 | 6.7727 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-diverse-v2 | shortest | 0.2933 | 0.4561 | 0.2933 | 0.4561 | 906.58 | 168.8805 | 46.3567 | 73.7675 | 0.0 | 0.0 | 5.4067 | 18.2534 | 0.2933 | 0.4561 | 0.2933 | 0.4561 | 5.4067 | 18.2534 | 5.4067 | 18.2534 | 2.63 | 2.3175 | 7.2933 | 0.6749 | 7.41 | 0.8352 | 3.8636 | 1.7676 | 46.3567 | 73.7675 | 46.3567 | 73.7675 | 31.2033 | 75.6753 | 0.4635 | 0.3179 | 8.3509 | 8.7364 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-play-v2 | full_bars | 0.49 | 0.5007 | 0.49 | 0.5007 | 825.2067 | 213.5717 | 69.51 | 86.3811 | 0.0 | 0.0 | 10.5133 | 19.0174 | 0.6167 | 0.487 | 0.6167 | 0.487 | 31.54 | 57.0522 | 10.5133 | 19.0174 | 1.8567 | 2.3639 | 8.3333 | 1.6966 | 8.48 | 1.6529 | 3.4957 | 1.9941 | 69.51 | 86.3811 | 69.51 | 86.3811 | 57.6133 | 87.5067 | 0.6851 | 0.2506 | 3.557 | 5.6456 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-play-v2 | reachability | 0.3067 | 0.4619 | 0.3067 | 0.4619 | 893.9067 | 187.509 | 61.21 | 82.0408 | 0.0 | 0.0 | 6.86 | 18.122 | 0.5967 | 0.4914 | 0.5967 | 0.4914 | 13.72 | 36.2441 | 6.86 | 18.122 | 2.05 | 2.3279 | 8.3267 | 1.692 | 8.4333 | 1.6517 | 3.6815 | 1.8828 | 61.21 | 82.0408 | 61.21 | 82.0408 | 48.14 | 83.0496 | 0.6262 | 0.2423 | 4.056 | 5.4221 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-play-v2 | shortest | 0.21 | 0.408 | 0.21 | 0.408 | 941.98 | 137.2686 | 40.9833 | 65.914 | 0.0 | 0.0 | 5.7633 | 17.8014 | 0.2867 | 0.453 | 0.2867 | 0.453 | 5.7633 | 17.8014 | 5.7633 | 17.8014 | 2.8267 | 2.1051 | 7.5833 | 1.1865 | 7.5867 | 1.1834 | 4.2569 | 1.5702 | 40.9833 | 65.914 | 40.9833 | 65.914 | 24.2933 | 67.6135 | 0.382 | 0.2953 | 9.9012 | 8.402 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |

## Profile Summary
| env | phase | event | duration_sec |
| --- | --- | --- | --- |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_end | 0.0176 |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_end | 3.196 |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_policy_end | 0.0086 |
| antmaze-medium-diverse-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_end | 0.009 |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_end | 1.4111 |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_start | NaN |
| antmaze-medium-play-v2 | pipeline | embed_dataset_end | 0.0159 |
| antmaze-medium-play-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | load_dataset_end | 3.2662 |
| antmaze-medium-play-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_policy_end | 0.0054 |
| antmaze-medium-play-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_reachability_end | 0.0049 |
| antmaze-medium-play-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_tdr_end | 1.4875 |
| antmaze-medium-play-v2 | pipeline | train_tdr_start | NaN |

## Failure Modes
_No data._

## Decision
- decision: EXPAND_EVAL

## Next Actions
- python scripts/collect_csv.py --log-root runs_stage18_medium100_best_loaded
- python scripts/analyze_bars_results.py --log-root runs_stage18_medium100_best_loaded --stage stage3
- Increase eval episodes or expand to large tasks under the same protocol.
