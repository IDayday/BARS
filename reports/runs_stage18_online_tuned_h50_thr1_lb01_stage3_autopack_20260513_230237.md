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
| antmaze-medium-diverse-v2 | full_bars | 0.44 | 0.498 | 0.44 | 0.498 | 829.1933 | 222.0041 | 68.86 | 140.0127 | 0.0 | 0.0 | 5.9067 | 12.2081 | 0.5267 | 0.501 | 0.5267 | 0.501 | 17.72 | 36.6244 | 5.9067 | 12.2081 | 1.98 | 2.6835 | 8.4467 | 0.4988 | 8.4733 | 0.501 | 3.841 | 2.0498 | 68.86 | 140.0127 | 68.86 | 140.0127 | 56.6533 | 141.7301 | 0.6295 | 0.2715 | 4.2294 | 6.4605 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-diverse-v2 | reachability | 0.38 | 0.487 | 0.38 | 0.487 | 841.52 | 222.7259 | 81.1467 | 173.7681 | 0.0 | 0.0 | 5.5133 | 12.1178 | 0.4533 | 0.4995 | 0.4533 | 0.4995 | 11.0267 | 24.2357 | 5.5133 | 12.1178 | 2.2867 | 2.6275 | 7.92 | 0.6905 | 7.96 | 0.694 | 3.8476 | 1.9438 | 81.1467 | 173.7681 | 81.1467 | 173.7681 | 69.0 | 176.1521 | 0.6218 | 0.2827 | 5.4362 | 6.7714 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-diverse-v2 | shortest | 0.2867 | 0.4537 | 0.2867 | 0.4537 | 911.6667 | 163.8912 | 45.6933 | 68.7272 | 0.0 | 0.0 | 7.5267 | 23.9926 | 0.3 | 0.4598 | 0.3 | 0.4598 | 7.5267 | 23.9926 | 7.5267 | 23.9926 | 2.82 | 2.4221 | 7.3067 | 0.665 | 7.4067 | 0.8118 | 3.9022 | 1.8913 | 45.6933 | 68.7272 | 45.6933 | 68.7272 | 30.3933 | 70.5701 | 0.4513 | 0.3254 | 8.9581 | 9.0977 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-play-v2 | full_bars | 0.5133 | 0.5015 | 0.5133 | 0.5015 | 806.1333 | 219.4798 | 68.0867 | 89.5451 | 0.0 | 0.0 | 9.7333 | 18.504 | 0.6267 | 0.4853 | 0.6267 | 0.4853 | 29.2 | 55.512 | 9.7333 | 18.504 | 1.7933 | 2.4585 | 8.3267 | 1.7008 | 8.48 | 1.6898 | 3.5253 | 2.0661 | 68.0867 | 89.5451 | 68.0867 | 89.5451 | 56.58 | 90.4378 | 0.6843 | 0.2435 | 3.442 | 5.2954 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-play-v2 | reachability | 0.3133 | 0.4654 | 0.3133 | 0.4654 | 891.0533 | 191.1941 | 56.8533 | 69.6321 | 0.0 | 0.0 | 6.6867 | 18.272 | 0.6 | 0.4915 | 0.6 | 0.4915 | 13.3733 | 36.5441 | 6.6867 | 18.272 | 2.1 | 2.3962 | 8.32 | 1.6961 | 8.4733 | 1.6413 | 3.7081 | 1.8925 | 56.8533 | 69.6321 | 56.8533 | 69.6321 | 43.6867 | 70.5161 | 0.6097 | 0.2457 | 4.1386 | 5.1147 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-medium-play-v2 | shortest | 0.2 | 0.4013 | 0.2 | 0.4013 | 943.52 | 132.1732 | 42.1733 | 68.5121 | 0.0 | 0.0 | 6.1067 | 18.5878 | 0.28 | 0.4505 | 0.28 | 0.4505 | 6.1067 | 18.5878 | 6.1067 | 18.5878 | 2.8467 | 2.0846 | 7.5667 | 1.1839 | 7.5733 | 1.1778 | 4.2268 | 1.6027 | 42.1733 | 68.5121 | 42.1733 | 68.5121 | 25.4 | 70.397 | 0.376 | 0.3045 | 10.0401 | 8.4709 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |

## Profile Summary
| env | phase | event | duration_sec |
| --- | --- | --- | --- |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_end | 0.0151 |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_end | 3.2425 |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_policy_end | 0.0089 |
| antmaze-medium-diverse-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_end | 0.0059 |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_end | 1.408 |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_start | NaN |
| antmaze-medium-play-v2 | pipeline | embed_dataset_end | 0.0167 |
| antmaze-medium-play-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | load_dataset_end | 3.1322 |
| antmaze-medium-play-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_policy_end | 0.0049 |
| antmaze-medium-play-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_reachability_end | 0.0053 |
| antmaze-medium-play-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_tdr_end | 1.3283 |
| antmaze-medium-play-v2 | pipeline | train_tdr_start | NaN |

## Failure Modes
_No data._

## Decision
- decision: EXPAND_EVAL

## Next Actions
- python scripts/collect_csv.py --log-root runs_stage18_online_tuned_h50_thr1_lb01
- python scripts/analyze_bars_results.py --log-root runs_stage18_online_tuned_h50_thr1_lb01 --stage stage3
- Increase eval episodes or expand to large tasks under the same protocol.
