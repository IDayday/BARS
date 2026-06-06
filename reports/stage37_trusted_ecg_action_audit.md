# Stage37 Trusted ECG / Action 审计

## 审计输入

- `reports/stage36_ecg_transition_action_summary.md`
- `reports/stage36_transition_contract_graph_build.md`
- `reports/stage36_final_recovery_contract_augmentation.md`
- `reports/stage36_transition_contract_planner_offline.md`
- `reports/stage36_action_supervised_contract_mining.md`
- `reports/stage36_ecg_policy_alignment_v2.md`
- `reports/stage35_cage_ecg_summary.md`
- `docs/cage_ecg_framework_design.md`
- `external_src/GAS/cage/contract_graph.py`
- `external_src/GAS/cage/contract_planner.py`
- `scripts/build_cage_transition_contract_graph.py`
- `scripts/augment_final_recovery_contracts.py`
- `scripts/mine_action_supervised_contract_examples.py`
- `scripts/build_ecg_policy_alignment_dataset_v2.py`
- `results/cage_ecg/transition_contract_graph/contract_graph_augmented.json`
- `results/cage_ecg/transition_contract_planner/offline_plan_audit.csv`
- `results/cage_ecg/policy_alignment/ecg_policy_alignment_v2.jsonl`

## Stage36 结论

Stage36 Transition Graph Gate 已通过：edge_count 从 `1542` 增至 `51542`，avg out-degree 提升，planner audit 出现 offline signal。

但 Stage36 仍不能进入 online benchmark：

1. KNN bridge candidate 占比过高，属于候选连通边，不是观测 transition。
2. recovery contract 仍 underpowered，recovery_edge_rate 只有 `0.0007`。
3. action_supervision_rate 仍为 `0.0000`。
4. policy alignment 只能用于 ranking / contrastive / conservative filtering，不能做 BC。

## Stage37 目标

Stage37 只做离线可信性和动作监督恢复，不做 CAGE online 阈值调参，不跑 humanoid/teleport，不跑大规模 online benchmark：

- trusted edge provenance audit；
- trusted contract graph variants；
- trusted planner re-audit；
- offline OGBench / cached dataset action supervision recovery；
- trainability gate for ECG policy alignment。

本轮禁止 online benchmark 和 SOTA claim。
