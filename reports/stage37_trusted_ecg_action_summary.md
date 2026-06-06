# Stage37 Trusted ECG / Action 总结

## 1. 当前 branch / commit

- branch: `codex/cage-mvp`
- base commit: `6a5af6b`
- 本轮没有启动 humanoid/teleport，没有启动 online benchmark，也没有调 CAGE online 干预阈值。

## 2. 新增/修改文件

- `scripts/audit_ecg_edge_provenance.py`
- `scripts/build_trusted_ecg_contract_graphs.py`
- `scripts/evaluate_trusted_ecg_planners.py`
- `scripts/recover_ogbench_action_supervision.py`
- `scripts/build_ecg_policy_alignment_dataset_v3.py`
- `tests/test_ecg_edge_provenance_audit.py`
- `tests/test_trusted_ecg_contract_graphs.py`
- `tests/test_recover_ogbench_action_supervision.py`
- `tests/test_ecg_policy_alignment_v3.py`
- `docs/cage_ecg_framework_design.md`
- `reports/stage37_trusted_ecg_action_audit.md`
- `reports/stage37_edge_provenance_audit.md`
- `reports/stage37_trusted_graph_build.md`
- `reports/stage37_trusted_planner_audit.md`
- `reports/stage37_ogbench_action_supervision_recovery.md`
- `reports/stage37_ecg_policy_alignment_v3.md`

## 3. 测试结果

```bash
python -m py_compile \
  scripts/audit_ecg_edge_provenance.py \
  scripts/build_trusted_ecg_contract_graphs.py \
  scripts/evaluate_trusted_ecg_planners.py \
  scripts/recover_ogbench_action_supervision.py \
  scripts/build_ecg_policy_alignment_dataset_v3.py
```

结果：PASS。

```bash
pytest \
  tests/test_ecg_edge_provenance_audit.py \
  tests/test_trusted_ecg_contract_graphs.py \
  tests/test_recover_ogbench_action_supervision.py \
  tests/test_ecg_policy_alignment_v3.py \
  tests/test_contract_graph.py \
  tests/test_contract_planner.py \
  -q
```

结果：`8 passed`。

## 4. Edge provenance audit 结论

命令：

```bash
python scripts/audit_ecg_edge_provenance.py \
  --contract_graph_path results/cage_ecg/transition_contract_graph/contract_graph_augmented.json \
  --planner_audit_path results/cage_ecg/transition_contract_planner/offline_plan_audit.csv \
  --out_csv results/cage_ecg/trusted_graph/edge_provenance_audit.csv \
  --out_md reports/stage37_edge_provenance_audit.md
```

结果：

| metric | value |
|---|---:|
| edge_count | 51542 |
| knn_edge_rate | 0.7666 |
| observed_edge_rate | 0.1746 |
| improved_planner_rows | 452 |
| improved_paths_with_knn_rate | 1.0000 |
| improved_paths_observed_only_count | 0 |

- provenance_status: `KNN_DEPENDENT_PLANNER_SIGNAL`
- trusted_signal_status: `TRUSTED_SIGNAL_NOT_OBSERVED_IN_FULL_AUDIT`

解释：Stage36 planner signal 主要依赖 KNN bridge candidate；KNN 不是观测 transition，不能作为已验证可执行边。

## 5. Trusted graph variant 统计

命令：

```bash
python scripts/build_trusted_ecg_contract_graphs.py \
  --contract_graph_path results/cage_ecg/transition_contract_graph/contract_graph_augmented.json \
  --out_dir results/cage_ecg/trusted_graph \
  --min_contract_lcb -1.0 \
  --max_knn_negative_risk 0.50 \
  --max_knn_uncertainty 0.50 \
  --clear
```

| variant | nodes | edges | avg_out | final_rate | recovery_rate | knn_rate | high_negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| observed_only | 7467 | 8999 | 1.0479 | 0.0000 | 0.0000 | 0.0000 | 0.4641 |
| observed_plus_final | 7475 | 11994 | 1.4475 | 0.2497 | 0.0000 | 0.0000 | 0.4481 |
| trusted_conservative | 7479 | 44635 | 5.3633 | 0.0671 | 0.0000 | 0.7313 | 0.2592 |
| full | 7479 | 51542 | 6.1806 | 0.0581 | 0.0007 | 0.7666 | 0.3581 |

trusted_conservative 降低了 high-negative KNN，但仍高度依赖 KNN bridge；observed_plus_final 不含 KNN，但连通性更弱。

## 6. Trusted planner audit 结论

命令：

```bash
python scripts/evaluate_trusted_ecg_planners.py \
  --graph_roots \
    results/cage_ecg/trusted_graph/observed_only \
    results/cage_ecg/trusted_graph/observed_plus_final \
    results/cage_ecg/trusted_graph/trusted_conservative \
    results/cage_ecg/trusted_graph/full \
  --out_dir results/cage_ecg/trusted_graph/planner_audit \
  --num_pairs 256 \
  --seed 0 \
  --require_multihop_pairs
```

关键结果：

- observed_only 有少量 planner difference，但 risk_constrained no_path_rate 高。
- observed_plus_final 有少量 progress_contract difference，但没有改善 contract/risk。
- trusted_conservative 有明确离线 signal：
  - max_contract_path diff_from_shortest `0.8415`
  - max_contract_path reduce_risk_rate `0.7073`
  - risk_constrained_path improve_contract_rate `0.1385`
- full graph 也有 signal，但 KNN usage 更高。

trusted_planner_status: `TRUSTED_OFFLINE_SIGNAL`。

注意：这是离线 planner signal，不是 online success 证据。

## 7. OGBench action supervision recovery 结果

命令：

```bash
python scripts/recover_ogbench_action_supervision.py \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --contract_graph_path results/cage_ecg/transition_contract_graph/contract_graph_augmented.json \
  --envs antmaze-giant-navigate-v0 antmaze-giant-stitch-v0 humanoidmaze-large-navigate-v0 \
  --max_examples 50000 \
  --phi_match_threshold 1e-4 \
  --loose_phi_match_threshold 1e-2 \
  --out_jsonl results/cage_ecg/policy_alignment/ogbench_action_supervised_contract_examples.jsonl \
  --out_report reports/stage37_ogbench_action_supervision_recovery.md
```

结果：

| metric | value |
|---|---:|
| status | BLOCKED_NO_ACTION_MATCH |
| total_examples | 7785 |
| exact_action_count | 0 |
| loose_action_count | 0 |
| action_supervision_rate | 0.0000 |
| positive_with_action_count | 0 |
| final_goal_with_action_count | 0 |
| recovery_with_action_count | 0 |

Per-env：

- antmaze-giant-navigate-v0: dataset_files `3`，examples `6324`，action_available `0`
- antmaze-giant-stitch-v0: dataset_files `3`，examples `0`
- humanoidmaze-large-navigate-v0: dataset_files `0`，examples `1461`

缺失原因：

- `no_phi_match_in_dataset`: 6324
- `no_dataset_npz_with_tdr_emb_actions`: 1461

结论：离线 BC 仍被 action supervision blocker 阻塞。不能从 phi-only 样本虚构 action。

## 8. ECG policy alignment v3 结果

命令：

```bash
python scripts/build_ecg_policy_alignment_dataset_v3.py \
  --contract_graph_path results/cage_ecg/trusted_graph/trusted_conservative/contract_graph.json \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --action_supervised_path results/cage_ecg/policy_alignment/ogbench_action_supervised_contract_examples.jsonl \
  --out_jsonl results/cage_ecg/policy_alignment/ecg_policy_alignment_v3.jsonl \
  --out_report reports/stage37_ecg_policy_alignment_v3.md
```

结果：

| metric | value |
|---|---:|
| total_examples | 44635 |
| action_supervision_rate | 0.0000 |
| positive_with_action_count | 0 |
| final_goal_with_action_count | 0 |
| recovery_with_action_count | 0 |
| bc_trainable_count | 0 |
| bc_split_generated | False |

Training stage counts：

```json
{
  "contract_ranking_contrastive": 5042,
  "final_goal_contract_ranking": 2995,
  "negative_contract_conservative_filtering": 3449,
  "positive_contract_ranking": 508,
  "trusted_bridge_conservative_filtering": 32641
}
```

## 9. Stage37 Gate 表

| Gate | 状态 | 依据 |
|---|---|---|
| Edge Provenance Gate | KNN_DEPENDENT_SIGNAL | Stage36 improved paths with KNN rate = 1.0。 |
| Trusted Planner Gate | PARTIAL / TRUSTED_OFFLINE_SIGNAL | trusted_conservative 有 signal，但仍大量使用 KNN；observed_plus_final signal 很弱。 |
| Offline Action Supervision Gate | FAIL | action_supervision_rate = 0，positive_with_action_count = 0。 |
| Final/Recovery Gate | FAIL / UNDERPOWERED | final with action = 0，recovery with action = 0；recovery coverage 仍弱。 |
| Online Readiness Gate | FAIL | action supervision gate 未过，仍禁止 online benchmark。 |

## 10. 是否建议进入 policy training

不建议进入 BC policy training。可以继续做 ranking / contrastive / conservative filtering 的离线模型，但不能做 action-supervised BC。

## 11. 是否建议进入 limited online AntMaze smoke

暂不建议。trusted planner 有离线信号，但 action supervision gate 失败；如果下一轮只验证 planner，不涉及 policy training，可考虑先做更严格的 observed_plus_final / trusted_conservative AntMaze offline-to-online smoke 设计，但本轮不执行。

## 12. 是否建议进入 humanoid/teleport 或 SOTA benchmark

不建议，继续禁止。

## 13. 下一条最推荐命令

下一步应恢复同源 TDR embedding 与 action 的映射，而不是进入 online：

```bash
python scripts/recover_ogbench_action_supervision.py \
  --contract_dataset_path results/cage_clp1/datasets/closed_loop_contracts.jsonl \
  --contract_graph_path results/cage_ecg/transition_contract_graph/contract_graph_augmented.json \
  --envs antmaze-giant-navigate-v0 \
  --dataset_cache_root artifacts/gas_ogbench_offline_full_20260522_165138 \
  --max_examples 50000 \
  --phi_match_threshold 1e-3 \
  --loose_phi_match_threshold 5e-2 \
  --out_jsonl results/cage_ecg/policy_alignment/ogbench_action_supervised_contract_examples_loose.jsonl \
  --out_report reports/stage37_ogbench_action_supervision_recovery_loose.md
```

如果仍为 0，需要回到 CLP1 segment capture，显式写出 `action_sequence` 或保存生成 `phi_s` 的 dataset row id。
