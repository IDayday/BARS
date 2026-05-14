# Stage3 Report

## Run Completion
- total runs: 6
- completed: 6
- failed: 0
- terminated: 0
- archived: 6

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
| antmaze-medium-diverse-v2 | full_bars | 0.48 | 0.5013 | 0.48 | 0.5013 | 818.66 | 220.5248 | 50.1933 | 71.4775 | 0.0 | 0.0 | 6.3667 | 13.0309 | 0.5267 | 0.501 | 0.5267 | 0.501 | 19.1 | 39.0927 | 6.3667 | 13.0309 | 1.56 | 2.1936 | 8.54 | 0.5001 | 8.5733 | 0.5226 | 3.5665 | 1.8366 | 50.1933 | 71.4775 | 50.1933 | 71.4775 | 38.1333 | 72.4619 | 0.6416 | 0.2562 | 2.9521 | 5.034 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.3 | 0.0 |
| antmaze-medium-play-v2 | full_bars | 0.3133 | 0.4654 | 0.3133 | 0.4654 | 886.5467 | 193.1512 | 54.9267 | 57.8161 | 0.0 | 0.0 | 15.5 | 40.9263 | 0.48 | 0.5013 | 0.48 | 0.5013 | 46.5 | 122.7789 | 15.5 | 40.9263 | 2.68 | 2.8622 | 8.3267 | 1.7008 | 8.52 | 1.6042 | 4.1055 | 2.2021 | 54.9267 | 57.8161 | 54.9267 | 57.8161 | 40.64 | 60.0803 | 0.5385 | 0.3379 | 7.3729 | 8.2757 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.3 | 0.0 |

## Profile Summary
| env | phase | event | duration_sec |
| --- | --- | --- | --- |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_end | 0.0157 |
| antmaze-medium-diverse-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_end | 3.1692 |
| antmaze-medium-diverse-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_policy_end | 0.0049 |
| antmaze-medium-diverse-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_end | 0.0042 |
| antmaze-medium-diverse-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_end | 1.1916 |
| antmaze-medium-diverse-v2 | pipeline | train_tdr_start | NaN |
| antmaze-medium-play-v2 | pipeline | embed_dataset_end | 0.0176 |
| antmaze-medium-play-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | load_dataset_end | 3.3325 |
| antmaze-medium-play-v2 | pipeline | load_dataset_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_policy_end | 0.0043 |
| antmaze-medium-play-v2 | pipeline | train_policy_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_reachability_end | 0.0034 |
| antmaze-medium-play-v2 | pipeline | train_reachability_start | NaN |
| antmaze-medium-play-v2 | pipeline | train_tdr_end | 1.2263 |
| antmaze-medium-play-v2 | pipeline | train_tdr_start | NaN |

## Failure Modes
_No data._

## Decision
- decision: EXPAND_EVAL

## Next Actions
- python scripts/collect_csv.py --log-root runs_stage18_online_tuned_h50_thr1_lb03_fullbars
- python scripts/analyze_bars_results.py --log-root runs_stage18_online_tuned_h50_thr1_lb03_fullbars --stage stage3
- Increase eval episodes or expand to large tasks under the same protocol.
