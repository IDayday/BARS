# Stage36 Transition Contract Planner Offline Audit

Graph: `results/cage_ecg/transition_contract_graph/contract_graph_augmented.json`

| planner | found | length | multihop | min_contract | success_lcb | negative_risk | diff_from_shortest | improve_contract | reduce_risk | no_path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bottleneck_robust_path | 0.7588 | 33.0993 | 1.0000 | 0.7398 | 0.0872 | 13.4631 | 0.9868 | 0.8609 | 0.0662 | 0.2412 |
| max_contract_path | 1.0000 | 16.0503 | 1.0000 | 0.6509 | 0.1533 | 6.6498 | 0.9347 | 0.3015 | 0.8693 | 0.0000 |
| progress_contract_path | 0.9799 | 24.1641 | 1.0000 | 0.7108 | 0.1409 | 9.9336 | 0.4410 | 0.3333 | 0.0205 | 0.0201 |
| risk_constrained_path | 0.3920 | 9.0513 | 1.0000 | 0.7358 | 0.3607 | 3.1974 | 0.3974 | 0.1282 | 0.3718 | 0.6080 |
| shortest_by_dphi | 1.0000 | 19.5879 | 1.0000 | 0.5822 | 0.1370 | 8.2664 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Planner difference gate: PLANNER_OFFLINE_SIGNAL
Multihop pair gate: PASS

## Bottleneck Edge Type Distribution

{
  "knn_bridge_candidate": 407,
  "original_contract": 47,
  "path_adjacency": 38,
  "recovery_candidate": 2,
  "temporal_transition": 328
}

若图中多数 pair 只有 direct edge，则 planner 可能都退化为同一路径；这不是在线结果，只是合同图连通性和风险约束的离线审计，不得写成 online SOTA。
