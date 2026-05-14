# Stage3 Report

## Run Completion
- total runs: 12
- completed: 12
- failed: 0
- terminated: 0
- archived: 12

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
| antmaze-large-diverse-v2 | loaded | 300.0 | 3600.0 |
| antmaze-large-play-v2 | loaded | 300.0 | 3600.0 |

## Eval Summary
| env | variant | success_mean | success_std | return_mean | return_std | steps_mean | steps_std | replans_mean | replans_std | no_path_count_mean | no_path_count_std | initial_plan_failed_count_mean | initial_plan_failed_count_std | plan_failed_initial_mean | plan_failed_initial_std | fallback_used_mean | fallback_used_std | fallback_count_mean | fallback_count_std | direct_goal_attempts_mean | direct_goal_attempts_std | last_plan_edges_mean | last_plan_edges_std | first_plan_edges_mean | first_plan_edges_std | max_plan_edges_mean | max_plan_edges_std | mean_plan_edges_mean | mean_plan_edges_std | num_plan_calls_mean | num_plan_calls_std | num_subgoal_attempts_mean | num_subgoal_attempts_std | num_subgoal_reached_mean | num_subgoal_reached_std | subgoal_reach_rate_mean | subgoal_reach_rate_std | goal_distance_final_mean | goal_distance_final_std | subgoal_horizon_mean | subgoal_horizon_std | subgoal_threshold_mean | subgoal_threshold_std | success_threshold_mean | success_threshold_std | lambda_risk_mean | lambda_risk_std | lambda_boundary_mean | lambda_boundary_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-large-diverse-v2 | reachability | 0.0 | 0.0 | 0.0 | 0.0 | 1000.0 | 0.0 | 72.8 | 149.5684 | 0.0 | 0.0 | 0.0444 | 0.2558 | 0.0333 | 0.1805 | 0.0333 | 0.1805 | 0.0889 | 0.5115 | 0.0444 | 0.2558 | 6.0222 | 3.3853 | 11.6667 | 0.474 | 11.7444 | 0.5913 | 8.293 | 1.8256 | 72.8 | 149.5684 | 72.8 | 149.5684 | 56.0444 | 152.5957 | 0.5016 | 0.2594 | 17.4644 | 10.1407 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-large-diverse-v2 | shortest | 0.0 | 0.0 | 0.0 | 0.0 | 1000.0 | 0.0 | 26.7778 | 38.2792 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.5889 | 1.8475 | 8.3333 | 0.474 | 8.3556 | 0.4814 | 6.6767 | 0.9476 | 26.7778 | 38.2792 | 26.7778 | 38.2792 | 7.2222 | 39.1206 | 0.1162 | 0.1764 | 25.5605 | 9.6493 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-large-play-v2 | reachability | 0.0111 | 0.1054 | 0.0111 | 0.1054 | 997.9889 | 19.0791 | 56.1889 | 110.4756 | 0.0 | 0.0 | 0.2 | 1.3171 | 0.0444 | 0.2072 | 0.0444 | 0.2072 | 0.4 | 2.6343 | 0.2 | 1.3171 | 7.3111 | 3.0192 | 10.0333 | 1.3193 | 10.4333 | 1.6425 | 8.4116 | 2.0446 | 56.1889 | 110.4756 | 56.1889 | 110.4756 | 38.3778 | 112.7823 | 0.3981 | 0.3139 | 23.2672 | 9.5957 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| antmaze-large-play-v2 | shortest | 0.0 | 0.0 | 0.0 | 0.0 | 1000.0 | 0.0 | 26.9556 | 11.53 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.4222 | 1.349 | 7.5444 | 0.6388 | 7.8111 | 0.7173 | 6.2513 | 0.7396 | 26.9556 | 11.53 | 26.9556 | 11.53 | 7.5667 | 11.9757 | 0.1872 | 0.2369 | 26.2703 | 10.3676 | 50.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.0 | 1.0 | 0.0 | 0.1 | 0.0 |

## Profile Summary
| env | phase | event | duration_sec |
| --- | --- | --- | --- |
| antmaze-large-diverse-v2 | pipeline | embed_dataset_end | 0.0168 |
| antmaze-large-diverse-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-large-diverse-v2 | pipeline | load_dataset_end | 3.3629 |
| antmaze-large-diverse-v2 | pipeline | load_dataset_start | NaN |
| antmaze-large-diverse-v2 | pipeline | train_policy_end | 0.0063 |
| antmaze-large-diverse-v2 | pipeline | train_policy_start | NaN |
| antmaze-large-diverse-v2 | pipeline | train_reachability_end | 0.0094 |
| antmaze-large-diverse-v2 | pipeline | train_reachability_start | NaN |
| antmaze-large-diverse-v2 | pipeline | train_tdr_end | 1.6496 |
| antmaze-large-diverse-v2 | pipeline | train_tdr_start | NaN |
| antmaze-large-play-v2 | pipeline | embed_dataset_end | 0.0146 |
| antmaze-large-play-v2 | pipeline | embed_dataset_start | NaN |
| antmaze-large-play-v2 | pipeline | load_dataset_end | 3.2517 |
| antmaze-large-play-v2 | pipeline | load_dataset_start | NaN |
| antmaze-large-play-v2 | pipeline | train_policy_end | 0.0048 |
| antmaze-large-play-v2 | pipeline | train_policy_start | NaN |
| antmaze-large-play-v2 | pipeline | train_reachability_end | 0.0041 |
| antmaze-large-play-v2 | pipeline | train_reachability_start | NaN |
| antmaze-large-play-v2 | pipeline | train_tdr_end | 1.3946 |
| antmaze-large-play-v2 | pipeline | train_tdr_start | NaN |

## Failure Modes
_No data._

## Decision
- decision: EXPAND_EVAL

## Next Actions
- python scripts/collect_csv.py --log-root runs_stage18_large_quick_bars_lite
- python scripts/analyze_bars_results.py --log-root runs_stage18_large_quick_bars_lite --stage stage3
- Increase eval episodes or expand to large tasks under the same protocol.
