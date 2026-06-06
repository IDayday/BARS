# Stage33 Contract-Rank Deployment Analysis

本表汇总 `cage_contract_rank` 部署后的 success、coverage、候选来源、拒绝计数和局部安全循环代理指标。缺失字段记为 NA。

## Grouped

| contract_best_non_gas_score_mean | contract_candidate_count_mean | contract_candidate_coverage_mean | contract_gas_score_mean | contract_selected_score_mean | env_name | extreme_negative_reject_count_mean | fallback_to_gas_step_count_mean | final_goal_on_rate_mean | global_replan_request_count_mean | local_safe_loop_count | mean_segment_progress_mean | num_jobs | rejected_count_mean | segment_target_reach_rate_mean | source_cage_rate_mean | source_committed_rate_mean | source_gas_rate_mean | stall_count_mean | success_rate_mean | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | 0.0000 | 9.0800 | 0.4800 | 0.0000 | 0 | 0.1735 | 1 | 0.0000 | 0.1975 | NA | NA | NA | 19.0400 | 0.4000 | cage_contract_commit |
| 0.5741 | 2.6324 | 1.0000 | 0.5419 | 0.6437 | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.4800 | 0.0000 | 0 | 0.1379 | 1 | 0.0000 | 0.3910 | 0.0411 | 0.4801 | 0.4021 | 19.6000 | 0.4400 | cage_contract_rank |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.8000 | 5.5600 | 0 | 0.2430 | 1 | 0.0000 | 0.0128 | NA | NA | NA | 10.8000 | 0.7200 | cage_safe_full |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.7200 | 0.0000 | 0 | 0.2210 | 1 | 0.0000 | 0.0535 | NA | NA | NA | 0.0000 | 0.6000 | cage_trace_only |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | NA | NA | NA | NA | 0 | NA | 1 | NA | NA | NA | NA | NA | NA | 0.6000 | gas |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | 0.0000 | 14.8400 | 0.0000 | 0.0000 | 1 | 0.0564 | 1 | 0.0000 | 0.8483 | NA | NA | NA | 51.4400 | 0.0000 | cage_contract_commit |
| -0.1144 | 2.9387 | 0.9965 | -0.2215 | -0.0522 | antmaze-giant-stitch-v0 | 6.4800 | 0.0000 | 0.6800 | 0.0000 | 0 | 0.1741 | 1 | 0.0036 | 0.3094 | 0.0602 | 0.6601 | 0.2504 | 22.5600 | 0.6400 | cage_contract_rank |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | 0.0000 | 0.0000 | 0.8000 | 5.6800 | 0 | 0.2710 | 1 | 0.0000 | 0.0086 | NA | NA | NA | 8.6400 | 0.8000 | cage_safe_full |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | 0.0000 | 0.0000 | 0.8400 | 0.0000 | 0 | 0.2945 | 1 | 0.0000 | 0.0439 | NA | NA | NA | 0.0000 | 0.8000 | cage_trace_only |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | NA | NA | NA | NA | 0 | NA | 1 | NA | NA | NA | NA | NA | NA | 0.8000 | gas |

## Job Rows

| contract_best_non_gas_score | contract_candidate_count | contract_candidate_coverage | contract_gas_score | contract_selected_score | env_name | extreme_negative_reject_count | fallback_to_gas_step_count | final_goal_on_rate | global_replan_request_count | job_id | local_safe_loop | mean_segment_progress | rejected_count | seed | segment_target_reach_rate | selected_source_distribution | source_cage_rate | source_committed_rate | source_gas_rate | stall_count | status | success_rate | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | NA | NA | NA | NA | antmaze-giant-navigate-v0__seed42__gas | False | NA | NA | 42 | NA | {} | NA | NA | NA | NA | succeeded | 0.6000 | gas |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.7200 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_trace_only | False | 0.2210 | 0.0000 | 42 | 0.0535 | {} | NA | NA | NA | 0.0000 | succeeded | 0.6000 | cage_trace_only |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.8000 | 5.5600 | antmaze-giant-navigate-v0__seed42__cage_safe_full | False | 0.2430 | 0.0000 | 42 | 0.0128 | {} | NA | NA | NA | 10.8000 | succeeded | 0.7200 | cage_safe_full |
| NA | NA | NA | NA | NA | antmaze-giant-navigate-v0 | 0.0000 | 9.0800 | 0.4800 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_contract_commit | False | 0.1735 | 0.0000 | 42 | 0.1975 | {} | NA | NA | NA | 19.0400 | succeeded | 0.4000 | cage_contract_commit |
| 0.5741 | 2.6324 | 1.0000 | 0.5419 | 0.6437 | antmaze-giant-navigate-v0 | 0.0000 | 0.0000 | 0.4800 | 0.0000 | antmaze-giant-navigate-v0__seed42__cage_contract_rank | False | 0.1379 | 0.0000 | 42 | 0.3910 | {"cage": 0.04110228865016347, "committed": 0.4801494628678188, "gas": 0.4021485287248949, "path_later": 0.07659971975712283} | 0.0411 | 0.4801 | 0.4021 | 19.6000 | succeeded | 0.4400 | cage_contract_rank |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | NA | NA | NA | NA | antmaze-giant-stitch-v0__seed42__gas | False | NA | NA | 42 | NA | {} | NA | NA | NA | NA | succeeded | 0.8000 | gas |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | 0.0000 | 0.0000 | 0.8400 | 0.0000 | antmaze-giant-stitch-v0__seed42__cage_trace_only | False | 0.2945 | 0.0000 | 42 | 0.0439 | {} | NA | NA | NA | 0.0000 | succeeded | 0.8000 | cage_trace_only |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | 0.0000 | 0.0000 | 0.8000 | 5.6800 | antmaze-giant-stitch-v0__seed42__cage_safe_full | False | 0.2710 | 0.0000 | 42 | 0.0086 | {} | NA | NA | NA | 8.6400 | succeeded | 0.8000 | cage_safe_full |
| NA | NA | NA | NA | NA | antmaze-giant-stitch-v0 | 0.0000 | 14.8400 | 0.0000 | 0.0000 | antmaze-giant-stitch-v0__seed42__cage_contract_commit | True | 0.0564 | 0.0000 | 42 | 0.8483 | {} | NA | NA | NA | 51.4400 | succeeded | 0.0000 | cage_contract_commit |
| -0.1144 | 2.9387 | 0.9965 | -0.2215 | -0.0522 | antmaze-giant-stitch-v0 | 6.4800 | 0.0000 | 0.6800 | 0.0000 | antmaze-giant-stitch-v0__seed42__cage_contract_rank | False | 0.1741 | 0.0036 | 42 | 0.3094 | {"cage": 0.06020245071923282, "committed": 0.6600958977091103, "gas": 0.2503995737879595, "path_later": 0.02930207778369739} | 0.0602 | 0.6601 | 0.2504 | 22.5600 | succeeded | 0.6400 | cage_contract_rank |
