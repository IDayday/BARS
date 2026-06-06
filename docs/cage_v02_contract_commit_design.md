# CAGE-v0.2 合同校准承诺执行设计

## 算法定义

CAGE-v0.2 的正式变体是 `cage_contract_commit`。它不是 `cage_full` 改名，而是一个保守的合同校准执行层：

1. GAS 仍提供原始图路径和原始子目标。
2. CAGE 只在 `--use_cage --cage_contract_commit` 显式开启时介入。
3. 任何候选子目标切换都必须通过闭环执行合同 gate。
4. gate 不确定或高风险时，优先保持当前 committed target；无可保持目标时 fallback 到 GAS 原始子目标。
5. churn guard 默认开启，aggressive recovery 默认关闭。

## 为什么 cage_full 不是主算法

Pilot-0/Repair-0 显示 `cage_full` 会把 drift、recovery 和 replanning 组合成 replan churn，特别是在 humanoid 上出现零 segment reach、零 recovery success 和大量 global replan request。这不是闭环执行合同，而是启发式补丁叠加。

## 为什么 commitment 是正信号

`cage_fixed_commit` 在 humanoid pilot 中曾出现正信号，说明一部分失败来自频繁 target switching 和低层 policy 无法完成未被打断的闭环收缩。v0.2 保留最小 commitment，并把切换作为需要合同验收的动作。

## 为什么 recovery 必须 contract-gated

Recovery target 并不天然可执行。没有合同验收时，recovery 可能把 policy 推向低 contract target，然后触发反复 stall/replan。v0.2 默认关闭 aggressive recovery；只有 `recovery_candidate` 的合同 LCB 通过 `cage_contract_recovery_threshold` 才允许启用。

## 为什么 final-goal phase 必须单独建模

最终目标通常不是普通 path waypoint。它可能存在语义接口、goal observation 噪声和 final-support mismatch。v0.2 对 `final_goal` 使用单独阈值 `cage_contract_final_goal_threshold`，不允许因为“离终点近”就无条件切换到 final goal。

## 合同模型输入输出

输入特征：

- `phi_s`, `phi_g`, `phi_delta`, `d_phi`
- `target_mode`
- `path_position`
- `final_phase`, `recovery_candidate`
- `recent_stall_count`, `recent_drift_count`
- `commitment_steps`
- `previous_target_distance`, `current_target_distance`
- `q_train_support`
- `env_name` bucket

输出：

- `predicted_hit`
- `predicted_contract_positive`
- `predicted_negative_progress`
- `uncertainty`
- `lower_confidence_bound`

如果模型不可用，`ContractScorer` 使用保守距离 fallback，并在 trace 中写 `contract_model_loaded=0`。

## 伪代码

```text
gas_target = GAS.select_target(phi_s, graph_path, final_goal)
candidate = CAGE.select_candidate(phi_s, graph_path, final_goal)

if in_min_commit and current_target_not_reached:
    return current_target

if final_goal_phase:
    if contract(final_goal).pass:
        return final_goal
    return keep_current_or_gas()

if drift_or_stall:
    if recovery_enabled and contract(recovery_candidate).pass:
        return recovery_candidate
    return keep_current_or_gas()

if contract(candidate).pass:
    return candidate

return keep_current_or_gas()
```

## 与 GAS 的区别

GAS 只按图路径和表示空间距离选子目标。CAGE-v0.2 不改图、不改 policy、不改 TDR，只在图目标进入 low-level policy 前做闭环合同验收和保守承诺。

## 与 reachability-only planner 的区别

reachability-only planner 通常在图搜索阶段过滤边。CAGE-v0.2 当前不改图搜索，只约束执行接口，并记录 contract gate、fallback 和实际 rollout 指标。未来可把同一合同模型前移到 path scoring。

## 与 simple fallback 的区别

simple fallback 只是出错后回到 GAS。CAGE-v0.2 明确记录为什么拒绝 target、合同分数、uncertainty、negative-progress 风险和 fallback 使用率；fallback 带来的成功率不会被计为 planner 改进。
