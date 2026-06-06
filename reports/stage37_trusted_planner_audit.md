# Stage37 Trusted ECG Planner Audit

| variant | planner | found | no_path | length | multihop | min_contract | success_lcb | negative_risk | diff | improve_contract | reduce_risk | knn_usage | observed_usage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observed_only | bottleneck_robust_path | 1.0000 | 0.0000 | 18.8333 | 1.0000 | 0.3436 | 0.1062 | 10.4037 | 0.1111 | 0.0556 | 0.0000 | 0.0000 | 1.0000 |
| observed_only | max_contract_path | 1.0000 | 0.0000 | 17.4444 | 1.0000 | 0.3436 | 0.1062 | 10.0563 | 0.0556 | 0.0556 | 0.0556 | 0.0000 | 1.0000 |
| observed_only | progress_contract_path | 1.0000 | 0.0000 | 17.4444 | 1.0000 | 0.3436 | 0.1062 | 10.0563 | 0.0556 | 0.0556 | 0.0556 | 0.0000 | 1.0000 |
| observed_only | risk_constrained_path | 0.2778 | 0.7222 | 5.2000 | 1.0000 | 0.6705 | 0.3824 | 2.5871 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| observed_only | shortest_by_dphi | 1.0000 | 0.0000 | 17.5556 | 1.0000 | 0.3286 | 0.1062 | 10.0848 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| observed_plus_final | bottleneck_robust_path | 1.0000 | 0.0000 | 14.9474 | 1.0000 | 0.1125 | 0.0161 | 8.1523 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| observed_plus_final | max_contract_path | 1.0000 | 0.0000 | 14.9474 | 1.0000 | 0.1125 | 0.0161 | 8.1523 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| observed_plus_final | progress_contract_path | 1.0000 | 0.0000 | 15.6316 | 1.0000 | 0.1125 | 0.0161 | 8.3839 | 0.0526 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| observed_plus_final | risk_constrained_path | 0.0526 | 0.9474 | 7.0000 | 1.0000 | 0.6901 | 0.3050 | 3.7677 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| observed_plus_final | shortest_by_dphi | 1.0000 | 0.0000 | 14.9474 | 1.0000 | 0.1125 | 0.0161 | 8.1523 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| trusted_conservative | bottleneck_robust_path | 0.8293 | 0.1707 | 32.9118 | 1.0000 | 0.6420 | 0.0858 | 12.4054 | 0.8603 | 0.6765 | 0.0735 | 0.6074 | 0.3926 |
| trusted_conservative | max_contract_path | 1.0000 | 0.0000 | 15.3780 | 1.0000 | 0.6117 | 0.1493 | 6.1574 | 0.8415 | 0.2927 | 0.7073 | 0.4925 | 0.5075 |
| trusted_conservative | progress_contract_path | 0.9878 | 0.0122 | 22.5123 | 1.0000 | 0.6593 | 0.1421 | 8.6773 | 0.4259 | 0.3642 | 0.0247 | 0.5703 | 0.4297 |
| trusted_conservative | risk_constrained_path | 0.3963 | 0.6037 | 9.0308 | 1.0000 | 0.7225 | 0.3513 | 3.1411 | 0.3846 | 0.1385 | 0.2923 | 0.6694 | 0.3306 |
| trusted_conservative | shortest_by_dphi | 1.0000 | 0.0000 | 17.3902 | 1.0000 | 0.5071 | 0.1384 | 7.0412 | 0.0000 | 0.0000 | 0.0000 | 0.5748 | 0.4252 |
| full | bottleneck_robust_path | 0.8177 | 0.1823 | 35.1506 | 1.0000 | 0.6498 | 0.0706 | 14.5002 | 0.8976 | 0.7590 | 0.0602 | 0.6660 | 0.3325 |
| full | max_contract_path | 1.0000 | 0.0000 | 15.6650 | 1.0000 | 0.5978 | 0.1254 | 6.7112 | 0.8768 | 0.2562 | 0.7586 | 0.5427 | 0.4553 |
| full | progress_contract_path | 0.9951 | 0.0049 | 23.6238 | 1.0000 | 0.6591 | 0.1177 | 9.9060 | 0.4554 | 0.3713 | 0.0149 | 0.6277 | 0.3719 |
| full | risk_constrained_path | 0.3350 | 0.6650 | 9.0000 | 1.0000 | 0.7208 | 0.3486 | 3.1821 | 0.3676 | 0.1324 | 0.2794 | 0.6844 | 0.3146 |
| full | shortest_by_dphi | 1.0000 | 0.0000 | 18.4581 | 1.0000 | 0.5305 | 0.1164 | 8.0608 | 0.0000 | 0.0000 | 0.0000 | 0.6210 | 0.3776 |

## Gate

- full_signal: `True`
- trusted_signal: `True`
- trusted_planner_status: `TRUSTED_OFFLINE_SIGNAL`

若只有 full graph 有 signal，则不能进入 online。只有 trusted_conservative 或 observed_plus_final 也有 planner signal，才可考虑下一轮 limited AntMaze smoke。
