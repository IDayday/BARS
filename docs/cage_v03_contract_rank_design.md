# CAGE-v0.3 Contract-Rank 设计

## 为什么 v0.2 失败

`cage_contract_commit` 把闭环合同作为 hard gate：低 LCB 或高 negative-progress risk 的候选目标直接被拒绝。Stage32 显示该策略能降低 replan churn，但在 AntMaze 上成功率明显回退，尤其 antmaze-stitch 的 gate reject rate 达到 0.998。失败不是 GAS path 缺失，也不是 trace-only 破坏，而是过度拒绝导致长期保持、fallback 和 stall。

## 为什么 hard gate 会导致安全局部循环

hard gate 会把“可能可执行但模型不够确信”的目标也拒掉。低层策略随后反复执行旧目标或 GAS fallback，局部 segment reach 可能上升，但没有持续向最终目标推进。表现为低 replan、低 switch、高 stall、final-goal 推进不足，成功率下降。

## v0.3 算法定义

`cage_contract_rank` 使用合同排序而不是硬拒绝。每一步构造候选集：

1. GAS 原始目标。
2. CAGE selector 目标。
3. 当前 committed target，前提是仍有进展。
4. nearest later path node。
5. final goal，仅在 final phase 中加入。
6. recovery candidate 默认不加入，除非未来显式开启并通过合同。

对每个候选计算合同预测：`predicted_hit`, `predicted_contract_positive`, `predicted_negative_progress`, `uncertainty`, `contract_lcb`。

排序分数：

```text
score =
  contract_weight * contract_lcb
  + progress_weight * path_progress_score
  - negative_weight * predicted_negative_progress
  - uncertainty_weight * uncertainty
  - switch_penalty * target_changed
```

只有当 `predicted_negative_progress > contract_rank_extreme_negative_threshold` 时才 hard reject。coverage 不得低于 `contract_rank_min_candidate_coverage`。如果 best non-GAS score 没有超过 GAS score 至少 `contract_rank_prefer_gas_margin`，选择 GAS 原始目标。

## 伪代码

```text
gas_target = original GAS planner target
cage_target = distance/adaptive selector target
candidates = [gas_target, cage_target, nearest_later_path]
if current_target_has_progress:
    candidates.append(current_committed_target)
if final_goal_phase:
    candidates.append(final_goal)

for c in candidates:
    pred = contract_scorer(phi_s, phi_c, context)
    c.rejected = pred.negative_progress > extreme_negative_threshold
    c.score = lcb + progress_bonus - negative_penalty - uncertainty_penalty - switch_penalty

restore coverage floor if too many candidates were rejected
if best_non_gas_score < gas_score + prefer_gas_margin:
    selected = gas_target
elif committed_score is close to best_score:
    selected = committed_target
else:
    selected = best_candidate
execute selected with the unchanged low-level GAS policy interface
```

## 与 GAS 的区别

GAS 直接执行图规划给出的当前目标。`cage_contract_rank` 保留 GAS 目标作为候选基准，但允许合同分数明显更好的 CAGE/path 目标替换它。

## 与 cage_safe_full 的区别

`cage_safe_full` 主要防 replan churn。`cage_contract_rank` 进一步估计候选子目标的闭环合同，并约束替换 GAS 目标的纪律。

## 与 cage_contract_commit 的区别

`cage_contract_commit` 是 hard gate；`cage_contract_rank` 是 soft ranking + coverage control。v0.3 只对 extreme negative risk 做硬拒绝，避免 Stage32 的 0.998 reject rate。

## 与 reachability-only planner 的区别

reachability-only 通常只判断边是否可达。CAGE-v0.3 的合同包含 closed-loop progress、negative-progress risk、uncertainty、commitment context 和 target mode，直接服务执行接口，而不是重建图或替换 backbone。

## 进入 SOTA benchmark 的条件

必须先满足：

1. trace-only parity 不破坏 GAS。
2. AntMaze success safety 不明显低于 GAS。
3. rank coverage 不低于 0.30，reject rate 不接近 Stage32 的 0.998。
4. replan 不显著高于 safe_full。
5. 不出现高 segment reach 但 success 为 0 的安全局部循环。
6. 合同分数与实际 progress/hit 有正相关，且 held-out 样本量足够。
