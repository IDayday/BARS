# Stage36 ECG Transition / Action 审计

## 审计输入

- `reports/stage35_cage_ecg_summary.md`
- `reports/stage35_cage_ecg_mechanism_laws.md`
- `reports/stage35_contract_graph_build.md`
- `reports/stage35_contract_planner_offline.md`
- `reports/stage35_graph_contract_policy_dataset.md`
- `docs/cage_ecg_framework_design.md`
- `external_src/GAS/cage/contract_graph.py`
- `external_src/GAS/cage/contract_planner.py`
- `scripts/build_cage_contract_graph.py`
- `scripts/evaluate_cage_contract_planner_offline.py`
- `scripts/build_graph_contract_policy_dataset.py`
- `results/cage_ecg/contract_graph/contract_graph.json`
- `results/cage_ecg/contract_planner/offline_plan_audit.csv`
- `results/cage_ecg/policy_alignment/graph_contract_policy_dataset.jsonl`

## Stage35 结论

Stage35 已经完成 CAGE-ECG 的离线框架雏形：

1. `ContractGraph` / `ContractPlanner` 基础数据结构和离线 planner audit 已实现。
2. Contract Graph Gate 已通过：图成功构建，edge 包含 `contract_lcb`、negative risk、uncertainty。
3. Contract Planner Gate 仍是 `INCONCLUSIVE`，因为多数 sampled pair 只有 direct edge，planner 没有产生不同于 shortest 的路径。
4. Policy Dataset Gate 是 `PARTIAL`，因为 `action_supervision_rate = 0`，不能直接做 BC policy training。
5. `final_goal_edge_rate = 0`，`recovery_edge_rate = 0.0039`，final/recovery 合同覆盖不足。

## Stage36 目标

本轮不跑 online benchmark，不启动 humanoid/teleport 或大规模 SOTA benchmark。Stage36 只补离线 ECG 缺口：

- transition-augmented contract graph；
- final/recovery contract augmentation；
- action-supervised hard-positive mining；
- offline planner re-audit；
- policy alignment dataset v2 re-audit。

## 预期 Gate

| Gate | Stage35 状态 | Stage36 目标 |
|---|---|---|
| Transition Graph | NA | edge_count / avg out-degree 增加，multihop pair rate > 0 |
| Final/Recovery Coverage | FAIL / PARTIAL | final edge > 0；recovery 若仍少则标记 UNDERPOWERED |
| Planner Difference | INCONCLUSIVE | 若 path differs 或 risk/contract 改善，标记 offline signal |
| Action Supervision | FAIL | 若 action rate 仍为 0，明确禁止 BC |
| Online Readiness | FAIL | 本轮默认 FAIL |
