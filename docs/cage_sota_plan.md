# CAGE SOTA 研究计划

## 基本原则

CAGE 不能只和 GAS 比，也不能用小样本 same-data 结果声称 SOTA。所有 SOTA 声明必须对齐：

- 相同环境版本
- 相同 seeds
- 相同 episode 数
- 相同 goal protocol
- 相同 evaluation horizon
- checkpoint 和训练预算清晰记录
- 均值、标准差和置信区间

## Baseline 范围

计划中至少纳入：

- OGBench reference baselines
- GCBC
- HIQL
- HILP
- GAS
- LAVL
- 其他当前公开强基线，如果 checkpoint 和协议可复现

## 两层 SOTA 路线

第一层：GAS backbone 上证明 execution-layer improvement。

- 冻结 TDR、keygraph、low-level policy。
- 比较 `gas`, `cage_trace_only`, `cage_fixed_commit`, `cage_safe_full`, `cage_contract_commit`。
- 目标不是立刻提升所有成功率，而是先证明 replan churn、target switching、contract-negative target selection 下降，并且标准任务不明显退化。

第二层：接入更强 backbone。

- CAGE 应作为可插拔执行层，不绑定单一 GAS。
- 如果在 GAS backbone 上证明 execution-layer 改进，再尝试接入更强 value/backbone。
- 验证 CAGE 是否能叠加提升，而不是替代 backbone 本身贡献。

## 不允许的声明

- 不能用 1x1 smoke 或 5x5 minipilot 说 SOTA。
- 不能把 fallback 到 GAS 带来的成功率当成 planner 改进。
- 不能把 same-data 合同模型评估当作泛化证据。
- 不能忽略失败 job 或只报告有利 variant。

## 当前阶段结论

CAGE-v0.2 当前只进入算法安全和诊断阶段。由于 AntMaze minipilot 显示 `cage_contract_commit` 成功率明显回退，当前不支持进入大规模 SOTA benchmark。下一步应修复合同模型泛化和 gate 过保守问题，再重新跑 staged minipilot。
