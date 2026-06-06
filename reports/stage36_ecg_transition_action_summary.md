# Stage36 ECG Transition / Action 总结

## 1. 当前 branch / commit

- branch: `codex/cage-mvp`
- base commit: `c81eca7`
- 本轮没有启动 humanoid/teleport，也没有启动大规模 online benchmark。

## 2. 新增/修改文件

- `scripts/build_cage_transition_contract_graph.py`
- `scripts/augment_final_recovery_contracts.py`
- `scripts/mine_action_supervised_contract_examples.py`
- `scripts/build_ecg_policy_alignment_dataset_v2.py`
- `scripts/evaluate_cage_contract_planner_offline.py`
- `external_src/GAS/cage/contract_graph.py`
- `docs/cage_ecg_framework_design.md`
- `reports/stage36_ecg_transition_action_audit.md`
- `reports/stage36_transition_contract_graph_build.md`
- `reports/stage36_final_recovery_contract_augmentation.md`
- `reports/stage36_transition_contract_planner_offline.md`
- `reports/stage36_action_supervised_contract_mining.md`
- `reports/stage36_ecg_policy_alignment_v2.md`
- `tests/test_transition_contract_graph.py`
- `tests/test_final_recovery_contract_augmentation.py`
- `tests/test_action_supervised_contract_mining.py`
- `tests/test_ecg_policy_alignment_v2.py`

## 3. 测试结果

```bash
python -m py_compile \
  scripts/build_cage_transition_contract_graph.py \
  scripts/augment_final_recovery_contracts.py \
  scripts/mine_action_supervised_contract_examples.py \
  scripts/build_ecg_policy_alignment_dataset_v2.py \
  scripts/evaluate_cage_contract_planner_offline.py
```

结果：PASS。

```bash
pytest \
  tests/test_transition_contract_graph.py \
  tests/test_final_recovery_contract_augmentation.py \
  tests/test_action_supervised_contract_mining.py \
  tests/test_ecg_policy_alignment_v2.py \
  tests/test_contract_graph.py \
  tests/test_contract_planner.py \
  -q
```

结果：`8 passed`。

## 4. Transition graph 构建结果

命令：

```bash
python scripts/build_cage_transition_contract_graph.py \
  --base_contract_graph results/cage_ecg/contract_graph/contract_graph.json \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --segment_capture_roots results/cage_clp1/segment_capture results/cage_clp1/segment_capture_candidate \
  --out_dir results/cage_ecg/transition_contract_graph \
  --max_transition_edges 50000 \
  --max_knn_edges_per_node 8 \
  --min_contract_lcb -1.0 \
  --clear
```

| metric | Stage35 | Stage36 |
|---|---:|---:|
| node_count | 470 | 7479 |
| edge_count | 1542 | 51542 |
| boundary_contract_count | 1414 | 28848 |
| avg_out_degree | about 3.28 | 6.1806 |
| weak_component_count | NA | 5 |
| largest_weak_component | NA | 5703 |
| strong_component_count | NA | 2522 |
| largest_strong_component | NA | 4725 |
| path_pair_reachability_proxy | NA | 0.0037 |
| transition_edge_rate | NA | 0.9113 |

Edge type 分布：

```json
{
  "final_goal_candidate": 2995,
  "knn_bridge_candidate": 39514,
  "original_contract": 1542,
  "path_adjacency": 340,
  "recovery_candidate": 34,
  "temporal_transition": 7117
}
```

说明：transition graph 成功解决 Stage35 mostly-direct-edge 的结构短板，但 KNN bridge candidate 占比很高，这些是候选连通边，不是真实观测 transition。

## 5. Final / recovery augmentation 结果

命令：

```bash
python scripts/augment_final_recovery_contracts.py \
  --contract_graph_path results/cage_ecg/transition_contract_graph/contract_graph.json \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --oracle_summary results/cage_clp1/oracle_candidate/contract_oracle_summary.json \
  --out_graph_path results/cage_ecg/transition_contract_graph/contract_graph_augmented.json \
  --out_report reports/stage36_final_recovery_contract_augmentation.md
```

| metric | value |
|---|---:|
| final_goal_edge_count | 2995 |
| final_goal_edge_rate | 0.0581 |
| final_goal_positive_rate | 0.0648 |
| final_goal_negative_rate | 0.7860 |
| recovery_edge_count | 34 |
| recovery_edge_rate | 0.0007 |
| recovery_positive_rate | 0.0294 |
| recovery_negative_rate | 0.0882 |

- final_goal_status: `ok`
- recovery_status: `RECOVERY_CONTRACT_UNDERPOWERED`

说明：final coverage 已从 Stage35 的 0 提升；recovery 仍极少，不足以支持强结论。

## 6. Planner re-audit 结果

命令：

```bash
python scripts/evaluate_cage_contract_planner_offline.py \
  --contract_graph_path results/cage_ecg/transition_contract_graph/contract_graph_augmented.json \
  --out_dir results/cage_ecg/transition_contract_planner \
  --num_pairs 256 \
  --seed 0 \
  --require_multihop_pairs
```

输出：`995` 行 planner audit。

| planner | found | length | multihop | min_contract | success_lcb | negative_risk | diff_from_shortest | improve_contract | reduce_risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shortest_by_dphi | 1.0000 | 19.5879 | 1.0000 | 0.5822 | 0.1370 | 8.2664 | 0.0000 | 0.0000 | 0.0000 |
| max_contract_path | 1.0000 | 16.0503 | 1.0000 | 0.6509 | 0.1533 | 6.6498 | 0.9347 | 0.3015 | 0.8693 |
| risk_constrained_path | 0.3920 | 9.0513 | 1.0000 | 0.7358 | 0.3607 | 3.1974 | 0.3974 | 0.1282 | 0.3718 |
| bottleneck_robust_path | 0.7588 | 33.0993 | 1.0000 | 0.7398 | 0.0872 | 13.4631 | 0.9868 | 0.8609 | 0.0662 |
| progress_contract_path | 0.9799 | 24.1641 | 1.0000 | 0.7108 | 0.1409 | 9.9336 | 0.4410 | 0.3333 | 0.0205 |

Planner difference gate: `PLANNER_OFFLINE_SIGNAL`。这是离线图信号，不是 online success 或 SOTA 证据。

## 7. Action-supervised mining 结果

命令：

```bash
python scripts/mine_action_supervised_contract_examples.py \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --segment_capture_roots results/cage_clp1/segment_capture results/cage_clp1/segment_capture_candidate \
  --out_jsonl results/cage_ecg/policy_alignment/action_supervised_contract_examples.jsonl \
  --out_report reports/stage36_action_supervised_contract_mining.md
```

| metric | value |
|---|---:|
| total_candidates | 2247 |
| action_available_count | 0 |
| action_supervision_rate | 0.0000 |
| positive_with_action_count | 0 |
| final_goal_with_action_count | 0 |
| recovery_with_action_count | 0 |

缺失原因：`segment_capture_has_no_action_fields = 2247`。

结论：action supervision gate 失败，不能进入 BC policy alignment training。

## 8. Policy alignment v2 数据集结果

命令：

```bash
python scripts/build_ecg_policy_alignment_dataset_v2.py \
  --contract_graph_path results/cage_ecg/transition_contract_graph/contract_graph_augmented.json \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --action_supervised_path results/cage_ecg/policy_alignment/action_supervised_contract_examples.jsonl \
  --out_jsonl results/cage_ecg/policy_alignment/ecg_policy_alignment_v2.jsonl \
  --out_report reports/stage36_ecg_policy_alignment_v2.md
```

| metric | value |
|---|---:|
| total_examples | 80390 |
| positive_contract_rate | 0.2836 |
| negative_contract_rate | 0.2256 |
| action_supervision_rate | 0.0000 |
| bc_trainable_count | 0 |
| final_phase_rate | 0.0373 |
| recovery_rate | 0.0006 |

该数据集可用于 ranking / contrastive / conservative filtering；不能用于 BC。

## 9. Stage36 Gate 表

| Gate | 状态 | 依据 |
|---|---|---|
| Transition Graph Gate | PASS | edge_count 1542 -> 51542，avg out-degree 提升，multihop planner audit 有样本。 |
| Final/Recovery Coverage Gate | PARTIAL | final_goal_edge_count > 0；recovery_edge_rate 仅 0.0007，标记 UNDERPOWERED。 |
| Planner Difference Gate | PLANNER_OFFLINE_SIGNAL | 多个 contract planner 产生不同于 shortest 的路径，并改善 min_contract 或 negative_risk。 |
| Action Supervision Gate | FAIL | action_supervision_rate = 0，positive_with_action_count = 0。 |
| Online Readiness Gate | FAIL | 本轮不进入 online benchmark；action supervision 和 recovery coverage 仍阻塞。 |

## 10. 是否建议进入 policy training

不建议进入 BC policy training。可以进入 contract ranking / contrastive / conservative filtering 的离线模型设计；若要做 policy finetuning，必须先恢复 action-supervised hard-positive examples。

## 11. 是否建议进入 online benchmark

不建议。虽然 planner re-audit 有离线信号，但 action supervision gate 失败，recovery coverage underpowered，且没有 online AntMaze smoke 证据。humanoid/teleport 和大规模 SOTA benchmark 继续禁止。

## 12. 下一条最推荐命令

下一步应先采集或恢复带 action 的 segment contract trace：

```bash
python scripts/run_contract_capture_smoke.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_ecg/action_segment_capture_antmaze \
  --envs antmaze-giant-navigate-v0 \
  --seeds 42 \
  --variants gas cage_trace_only cage_safe_full \
  --episodes_per_goal 2 \
  --goals_per_env 2 \
  --store_contract_state_refs \
  --contract_capture_action_stats
```

需要先确认该 capture 路径会写出 action 或 action_sequence 字段；否则继续只会得到 ranking 数据，不能做 BC。
