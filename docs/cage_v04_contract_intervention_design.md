# CAGE-v0.4: GAS 锚定的合同干预算法

## 为什么 v0.3 仍失败

Stage33 的 `cage_contract_rank` 修复了 Stage32 hard gate 过度拒绝，AntMaze 中 candidate coverage 接近 1，reject rate 降为 0。但它仍低于 GAS 约 16pp。部署分析显示失败不来自 no-path、replan storm 或 reject 过多，而来自 CAGE ranker 作为默认排序器过度接管 GAS：

- committed target 使用过多；
- GAS 原始目标被不必要覆盖；
- final-goal 推进不足；
- 部分 episode 出现局部安全但最终不成功的执行模式。

## 为什么 CAGE 应从默认排序器变成 GAS 锚定干预器

GAS 原始目标是官方 pipeline 的默认执行接口。只要 GAS 目标没有明显合同风险，并且 rollout 没有真实停滞，CAGE 不应替换它。CAGE-v0.4 的主假设是：合同模型当前还不足以作为全局目标选择器，但可以作为最小必要干预器，识别少数高风险 GAS 决策并替换为更安全、更前进的候选目标。

## 为什么 shadow rank 必须先跑

`cage_contract_shadow_rank` 构造与 `cage_contract_rank` 相同的候选集和分数，但实际仍执行 GAS 原始目标。它用于测量 ranker 是否会频繁覆盖成功 GAS 决策：

- 如果 successful episodes 中 shadow override 过高，ranker 不可信；
- 如果 final phase shadow override 过高，存在最终目标接口风险；
- shadow 不改变策略动作，因此可作为集成安全审计。

## 为什么 committed target 需要进展看门狗

Stage33 显示 committed source rate 很高。承诺执行只有在目标距离和最终目标距离仍有进展时才合理。v0.4 为 committed target 增加 watchdog：

- committed target distance 不下降则标记 stale；
- final-goal distance 不改善则标记 stale；
- stall 时进入 committed lockout；
- lockout 期间 committed target 不能进入候选集。

这避免 ranker 因惯性长期选择局部安全但不前进的旧目标。

## 为什么 final-goal phase 必须保护

最终目标阶段是 GAS 成功率的重要接口。v0.4 默认保留 GAS/final-goal 目标，只有在 extreme negative-progress risk 下才允许 override，并记录 `final_phase_preserved_count`、`final_phase_override_count` 和原因。

## 干预算法伪代码

```text
input: current state s, final goal g, GAS target g_gas, path P

if shadow_rank:
    candidates = {g_gas, cage_selected, path_later, committed_if_progressing, final_if_phase}
    scores = contract_rank(candidates)
    log what CAGE would select
    return g_gas

if contract_intervene:
    candidates = {g_gas, cage_selected, path_later, committed_if_progressing, final_if_phase}
    scores = contract_rank(candidates)
    gas_risk = predicted_negative_progress(g_gas)
    best_alt = best non-GAS candidate without extreme negative risk and without path regression
    intervention_gain =
        alt_score - gas_score
        + final_goal_progress_gain
        + path_index_gain
        - intervention_cost
        - switch_cost

    if final_phase and preserve_final_phase and gas_risk is not extreme:
        return g_gas

    if (gas_risk high or gas progress stalled)
       and intervention_gain > intervention_margin
       and best_alt does not regress path_position:
        return best_alt

    return g_gas
```

## 与其他变体的区别

| variant | 行为 |
|---|---|
| GAS | 官方目标，完全无 CAGE |
| cage_safe_full | 启发式 CAGE full + churn guard |
| cage_contract_commit | hard gate，过度拒绝候选目标 |
| cage_contract_rank | 默认排序器，修复拒绝但仍过度覆盖 GAS |
| cage_contract_shadow_rank | 只记录 ranker 会怎么选，实际执行 GAS |
| cage_contract_intervene | GAS 锚定，只在高风险且高收益时替换 |

## 进入 failure-dense 和 SOTA benchmark 的条件

必须先满足：

1. trace-only parity 通过；
2. shadow override on success 不过高；
3. AntMaze success safety 通过或至少不出现系统性退化；
4. committed source rate 低于 Stage33 contract_rank；
5. final-goal phase 没有被频繁覆盖；
6. global replan 不重现 storm；
7. 不出现 high segment reach but low success 的局部安全循环。

在这些 gate 未通过前，不进入 humanoid/teleport，更不能启动大规模 SOTA benchmark。
