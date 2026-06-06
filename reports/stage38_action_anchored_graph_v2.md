# Stage38 Action-Anchored ECG Graph v2

- status: `ACTION_ANCHORED_GRAPH_READY`
- graph: `/tmp/pytest-of-root/pytest-129/test_action_anchored_graph_has0/graph/contract_graph.json`
- node_count: 9
- edge_count: 8
- action_anchored_edge_rate: 1.0
- final_goal_edge_count: 1
- final_goal_edge_rate: 0.125
- unverified_knn_main_edge_count: 0
- knn_main_edge_rate: 0.0

主执行边只来自 offline temporal/final positive action-anchored samples；未验证 KNN bridge 没有进入主 planner edge。

- edge_type_counts: `{'offline_temporal_future_positive': 7, 'final_goal_positive': 1}`