# Stage34 CAGE-v0.4 Contract-Intervention 审计

## 审计输入

- `reports/stage33_contract_rank_summary.md`
- `reports/stage33_contract_rank_minipilot_antmaze.md`
- `reports/stage33_contract_rank_deployment.md`
- `reports/stage33_contract_rank_audit.md`
- `docs/cage_v03_contract_rank_design.md`
- `external_src/GAS/cage/contract_ranker.py`
- `external_src/GAS/cage/state_machine.py`
- `external_src/GAS/cage/config.py`
- `results/cage_v03_contract_rank/minipilot_antmaze/deployment_analysis.csv`
- `results/cage_v03_contract_rank/minipilot_antmaze/tables/minipilot_summary.md`

## Stage33 结论

Stage33 已修复 Stage32 hard gate 过度拒绝问题。`cage_contract_rank` 使用候选排序和 coverage floor 后，AntMaze minipilot 中 coverage 接近 1，reject rate 降为 0。

但 success safety 仍失败：

| env | GAS | cage_contract_rank | delta |
|---|---:|---:|---:|
| antmaze-giant-navigate-v0 | 0.600 | 0.440 | -0.160 |
| antmaze-giant-stitch-v0 | 0.800 | 0.640 | -0.160 |

当前主要问题不是 replan storm，也不是 reject 过多。Stage33 deployment 显示 `global_replan_request_count=0`，`contract_rejected_count≈0`，但 `committed` source 占比过高：

- nav committed source rate：0.480
- stitch committed source rate：0.660

这说明 `cage_contract_rank` 仍把 CAGE 当默认排序器使用，经常覆盖 GAS 原始目标，并把执行锁定在局部安全但不一定推动最终成功的 committed target 上。`final_goal_on_rate` 也低于 safe_full/trace_only 的稳定水平，说明最终目标推进不足。

## 当前判断

1. Stage33 已修复 Stage32 hard gate 过度拒绝问题。
2. `cage_contract_rank` coverage 接近 1，reject rate 降为 0。
3. success safety 仍失败：AntMaze nav 和 stitch 均低于 GAS 约 16pp。
4. 当前主要问题不是 replan storm，也不是 reject 过多，而是 committed target 使用过多、GAS 被不必要覆盖、final-goal 推进不足。
5. 下一步不能进入 humanoid/teleport，也不能进入大规模 SOTA benchmark。
6. 必须进入 CAGE-v0.4：GAS 锚定的最小必要干预。

## Stage34 Gate

| Gate | 标准 |
|---|---|
| trace-only parity | trace_only 与 GAS 成功率不得系统性退化 |
| shadow safety | shadow_override_on_success_rate < 0.20，shadow_final_phase_override_rate < 0.10 |
| success safety | `cage_contract_intervene` 在 AntMaze 中不得低于 GAS 超过 5pp |
| intervention discipline | intervention_rate 建议 < 0.30，GAS 正常推进时不得频繁替换 |
| committed control | committed source rate 必须低于 Stage33 contract_rank，stale/lockout 可见 |
| final-goal preservation | final_goal_on_rate 不得明显低于 trace_only/safe_full |
| churn safety | global replans 不得显著高于 safe_full，不得重现 replan storm |
| forward progress | 不允许 high segment reach but low success 的局部安全循环 |
| failure-dense readiness | AntMaze success safety 通过前，不进入 humanoid/teleport |

## BLOCKED 状态

当前未发现代码审计层面的 BLOCKED。后续实验仍可能因 checkpoint、依赖、GPU 或磁盘配额失败；若失败，需要在 Stage34 结果报告中明确记录原因，不得虚构结果。
