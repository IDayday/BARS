# Stage35 Contract Planner Offline Audit

Graph: `results/cage_ecg/contract_graph/contract_graph.json`

| planner | found | length | min_contract | success_lcb | negative_risk | bottlenecks | diff_from_shortest |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottleneck_robust_path | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 128 | 0.0000 |
| max_contract_path | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 128 | 0.0000 |
| progress_contract_path | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 128 | 0.0000 |
| risk_constrained_path | 0.4531 | 1 | 0.7520 | 0.7209 | 0.2644 | 58 | 0.0000 |
| shortest_by_dphi | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 128 | 0.0000 |

Planner difference gate: INCONCLUSIVE

若图中多数 pair 只有 direct edge，则 planner 可能都退化为同一路径；这不是在线结果，只是合同图连通性和风险约束的离线审计。
