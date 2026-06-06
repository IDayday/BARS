# Stage35 CAGE-ECG 总结

## 1. 当前 branch / commit

- branch: `codex/cage-mvp`
- base commit: `5dffb01`
- 本轮未启动 humanoid/teleport 或大规模 SOTA benchmark。

## 2. 代码改动摘要

本轮从 Stage34 的失败中转向 CAGE-ECG：执行合同图框架。新增内容包括：

- 执行合同图数据结构：funnel node、contract edge、boundary contract。
- 离线合同图构建脚本。
- 合同路径规划器：shortest、max-contract、risk-constrained、bottleneck-robust、progress-contract。
- 机制规律分析脚本。
- 图诱导低层策略对齐数据集构建脚本。
- CAGE-ECG 设计文档和 Stage35 审计/构建/规划/数据集报告。

## 3. 机制规律分析结论

`reports/stage35_cage_ecg_mechanism_laws.md` 总结了 Stage32-34 的失败演化：

- `mean_segment_progress` 与 success 正相关，观测相关系数为 `0.9341`。
- `final_goal_on_rate` 与 success 正相关，观测相关系数为 `0.9804`。
- `stall_count` 与 success 负相关，观测相关系数为 `-0.8286`。
- `segment_target_reach_rate` 与 success 在当前 compact rows 上呈负相关，说明局部 segment reach 不足以解释任务成功。
- `intervention_rate` 与 success 的关系弱，Stage34 表明只做执行时干预不足以修复 GAS。
- local-safe-loop proxy 检出 `3` 个样本。

结论：当前失败不是 no-path、hard gate reject、committed target 过用或 replan storm，而是图路径、闭环执行合同和任务推进目标之间的不一致。需要 ECG 的 contract graph、boundary compatibility 和 policy alignment。

## 4. Contract graph 构建结果

命令：

```bash
python scripts/build_cage_contract_graph.py \
  --contract_model_path results/cage_v02_contract/models/contract_model.json \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --out_dir results/cage_ecg/contract_graph \
  --clear
```

结果：

| metric | value |
|---|---:|
| node_count | 470 |
| edge_count | 1542 |
| boundary_contract_count | 1414 |
| low_contract_edge_rate | 0.5603 |
| high_negative_edge_rate | 0.2821 |
| uncertain_edge_rate | 0.0720 |
| final_goal_edge_rate | 0.0000 |
| recovery_edge_rate | 0.0039 |

说明：合同图成功构建，能识别 low-contract 和 high-risk edges。当前输入合同数据缺少 final-goal edge，recovery edge 很少；这限制了 final/recovery 机制分析。

## 5. Contract planner 离线审计结果

命令：

```bash
python scripts/evaluate_cage_contract_planner_offline.py \
  --contract_graph_path results/cage_ecg/contract_graph/contract_graph.json \
  --out_dir results/cage_ecg/contract_planner \
  --num_pairs 128 \
  --seed 0
```

输出 `640` 行 planner audit。主要结果：

| planner | found | length | min_contract | success_lcb | negative_risk | diff_from_shortest |
|---|---:|---:|---:|---:|---:|---:|
| shortest_by_dphi | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 0.0000 |
| max_contract_path | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 0.0000 |
| risk_constrained_path | 0.4531 | 1 | 0.7520 | 0.7209 | 0.2644 | 0.0000 |
| bottleneck_robust_path | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 0.0000 |
| progress_contract_path | 1.0000 | 1 | 0.3441 | 0.3281 | 0.3557 | 0.0000 |

Planner difference gate: `INCONCLUSIVE`。当前合同图多数 pair 只有 direct edge，导致不同 planner 退化为同一路径；这说明还需要更密集的 transition/boundary graph，而不是启动 online benchmark。

## 6. Policy alignment dataset 审计结果

命令：

```bash
python scripts/build_graph_contract_policy_dataset.py \
  --contract_graph_path results/cage_ecg/contract_graph/contract_graph.json \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --out_jsonl results/cage_ecg/policy_alignment/graph_contract_policy_dataset.jsonl \
  --out_report reports/stage35_graph_contract_policy_dataset.md
```

结果：

| metric | value |
|---|---:|
| total_examples | 2956 |
| positive_contract_rate | 0.2710 |
| negative_contract_rate | 0.4259 |
| action_supervision_rate | 0.0000 |
| final_phase_rate | 0.0000 |
| recovery_rate | 0.0041 |

数据集可以用于 contract ranking、contrastive objective、conservative filtering 和 curriculum weighting。由于 action supervision rate 为 `0.0000`，不能声称已经可以直接做 low-level BC finetuning。

## 7. Stage35 Gate 表

| Gate | 状态 | 依据 |
|---|---|---|
| Mechanism Law Gate | INCONCLUSIVE / 有机制证据 | 可解释 Stage32-34 失败演化，但样本小，不声明显著性。 |
| Contract Graph Gate | PASS | graph 构建成功，edge 有 LCB、negative risk、uncertainty，并能识别 low-contract/high-risk edges。 |
| Contract Planner Gate | INCONCLUSIVE | planner 可运行并报告 bottleneck/success LCB，但路径多数为 direct edge，尚未产生不同于 shortest 的路径。 |
| Policy Dataset Gate | PARTIAL | 能生成正/负合同样本，但 action supervision 为 0，不能直接 BC training。 |
| Online Readiness Gate | FAIL | 本轮不进入 online benchmark；需先补 transition graph、final/recovery 数据和 action supervision。 |

## 8. 验证结果

```bash
python -m py_compile \
  external_src/GAS/cage/contract_graph.py \
  external_src/GAS/cage/contract_planner.py \
  scripts/analyze_cage_ecg_mechanism.py \
  scripts/build_cage_contract_graph.py \
  scripts/evaluate_cage_contract_planner_offline.py \
  scripts/build_graph_contract_policy_dataset.py
```

结果：PASS。

```bash
pytest \
  tests/test_contract_graph.py \
  tests/test_contract_planner.py \
  tests/test_graph_contract_policy_dataset.py \
  tests/test_cage_ecg_mechanism.py \
  -q
```

结果：`5 passed in 0.23s`。

## 9. 是否建议进入 policy training

不建议直接进入 low-level BC policy training。当前 dataset 可用于 contract model / ranking / conservative filtering；若要做 policy alignment，需要先采集或恢复 action-supervised hard-positive examples。

## 10. 是否建议进入 online benchmark

不建议。AntMaze online success safety 在 Stage34 仍失败，Stage35 离线 planner gate 还没有证明 contract planner 能产生更优路径。humanoid/teleport 和大规模 SOTA benchmark 继续阻塞。

## 11. 下一条推荐命令

下一步应先补合同图连通性和 action-supervised hard positives，而不是调在线阈值：

```bash
python scripts/build_cage_contract_graph.py \
  --contract_model_path results/cage_v02_contract/models/contract_model.json \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --out_dir results/cage_ecg/contract_graph_all_edges \
  --min_contract_lcb -1.0 \
  --clear
```

该命令会保留 negative-LCB edges，用于检查 planner 是否需要低合同桥接边、以及 boundary contract 是否是当前图不连通的真正瓶颈。
