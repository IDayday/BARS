# Stage35 CAGE-ECG 审计

## 审计输入

- `reports/stage34_contract_intervention_summary.md`
- `reports/stage34_contract_intervention_deployment.md`
- `reports/stage34_contract_intervene_minipilot_antmaze.md`
- `reports/stage34_contract_shadow_rank_antmaze.md`
- `docs/cage_v04_contract_intervention_design.md`
- `reports/stage33_contract_rank_summary.md`
- `reports/stage33_contract_rank_deployment.md`
- `reports/stage32_contract_commit_summary.md`
- `docs/cage_gp0_alignment_report.md`
- `docs/cage_clp1_final_report.md`
- `external_src/GAS/cage/contract_intervention.py`
- `external_src/GAS/cage/contract_ranker.py`
- `external_src/GAS/cage/state_machine.py`
- `external_src/GAS/cage/config.py`
- `results/cage_v04_contract_intervene/deployment_analysis.csv`
- `results/cage_v04_contract_intervene/minipilot_antmaze/tables/minipilot_summary.csv`

## Stage32-34 失败演化

Stage32 `cage_contract_commit` 证明 hard gate 会过度拒绝候选目标。AntMaze stitch 的 gate reject rate 接近 1，replan 降为 0 但 success 回退到 0。

Stage33 `cage_contract_rank` 修复了 hard gate 过度拒绝，coverage 接近 1，reject rate 降为 0，但 ranker 作为默认目标选择器过度选择 committed target。AntMaze navigate/stitch 都比 GAS 低约 16pp。

Stage34 `cage_contract_intervene` 完成 v0.4 contract-intervention，工程测试通过。它比 v0.3 更安全：

- committed source rate 显著下降；
- global replan 仍为 0；
- success 高于 `cage_contract_rank`。

但 v0.4 仍未通过 success safety：

| env | GAS | cage_contract_intervene | delta |
|---|---:|---:|---:|
| antmaze-giant-navigate-v0 | 0.640 | 0.560 | -0.080 |
| antmaze-giant-stitch-v0 | 0.840 | 0.720 | -0.120 |

当前失败已经不是 hard gate reject、committed target 过用或 replan storm。更深层问题是：执行时干预不足以解决图路径、闭环执行合同和任务推进之间的不一致。

## GP0 / CLP1 机制背景

GP0 表明 coarse phi-space q_train support 不能解释主要失败。q_G 与 q_train 在粗 TDR 距离尺度上重叠，但 closed-loop execution 仍会失败。

CLP1 表明：

- Humanoid 的 closed-loop policy contractibility 明显低于 AntMaze；
- farther path target 和 final-goal target 经常 low-contract；
- same-data contract model 有信号，但不是 held-out SOTA 证据；
- policy alignment hard-goal dataset 缺少 supervised hard-positive examples，不能直接做 naive BC。

## Stage35 方向

下一步必须推进 CAGE-ECG，而不是继续只调干预阈值：

1. 执行 funnel node：表示策略能稳定进入/离开的闭环区域。
2. 执行合同 edge：表示从一个 funnel 到另一个 funnel 的可执行闭环合同。
3. 边界兼容 contract：表示连续 edge 拼接处是否会破坏低层策略。
4. 风险约束合同路径规划：规划不再只看 TDR/graph distance，还看 bottleneck、negative risk、uncertainty。
5. 图诱导低层策略对齐：把 high-contract / low-contract / final / recovery / boundary 目标转成训练或排序数据。

## Gate

| Gate | 状态 |
|---|---|
| Stage34 engineering | PASS |
| Stage34 success safety | FAIL |
| Mechanism law readiness | NEEDS OFFLINE ANALYSIS |
| Contract graph readiness | NEEDS BUILD |
| Contract planner readiness | NEEDS OFFLINE AUDIT |
| Policy alignment readiness | NEEDS DATASET AUDIT |
| Online benchmark readiness | FAIL，本轮禁止进入 online benchmark |
