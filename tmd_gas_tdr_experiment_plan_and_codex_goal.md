# TMD/TDR 增强 GAS 的后续实验计划与 Codex Goal 指令

生成日期：2026-05-27
基于结果包：`tmd_test_reports_stage25_protocol_oracle_drift`
当前结果规模：`eval_all.csv` 共 9102 行；`graph_all.csv` 共 26 行。

---

## 1. 当前证据摘要

### 1.1 当前最稳妥的结论

1. **目前不支持用 TMD hard graph 或 TMD actor 直接替换 GAS。** TMD exec graph 在部分单任务/单 scale 下可用，但 all-task 和 giant 环境不稳；TMD actor rescue 在 medium-navigate task3 上明显劣化。
2. **目前最值得推进的是 GAS graph + TMD soft cost blend。** 在 `antmaze-giant-navigate-v0` all tasks 上，`w=0.25` 对同一组 env_seed/gas_seed 的总体成功率和 task-wise 成功率都有提升。
3. **TMD rescue 目前更像 failure detector，而不是真正的 rescue policy。** 被触发 episode 的成功率很低；在 giant-stitch 的 b50/r3 rescue 中，activated subset 全失败，aggregate success 与 hybrid-disabled control 持平。
4. **更普适的低层 condition 是关键工程主线。** 当前失败模式说明高层 target/TMD 几何/TDR 几何没有被低层 actor 稳定消费，因此下一阶段应优先实现 `[TDR direction, TDR distance scale, task-factor mask, masked task-factor residual]`。

### 1.2 关键实验结果表

| variant                        | scope                    |   episodes |   success |   success_rate |   mean_steps | env_seeds   | gas_seeds   |
|:-------------------------------|:-------------------------|-----------:|----------:|---------------:|-------------:|:------------|:------------|
| GAS graph baseline             | giant-navigate all tasks |        400 |       257 |          0.642 |        803.4 | 0/1         | 42/43       |
| GAS graph + TMD cost w=0.25    | giant-navigate all tasks |        400 |       292 |          0.73  |        779.4 | 0/1         | 42/43       |
| GAS graph + TMD cost w=0.50    | giant-navigate all tasks |        400 |       268 |          0.67  |        802.8 | 0/1         | 42/43       |
| GAS graph baseline             | medium-stitch all tasks  |        500 |       489 |          0.978 |        246.5 | 0           | 42/43       |
| GAS graph + TMD rescue         | medium-stitch all tasks  |        500 |       493 |          0.986 |        242.6 | 0           | 42/43       |
| GAS graph baseline             | medium-navigate task3    |        400 |       386 |          0.965 |        186.4 | 0/1         | 42/43       |
| GAS graph + TMD rescue s175/p9 | medium-navigate task3    |        400 |       388 |          0.97  |        183.1 | 0/1         | 42/43       |
| GAS graph baseline             | giant-stitch all tasks   |        100 |        78 |          0.78  |        750.4 | 0           | 42/43       |
| GAS graph + TMD rescue b50/r3  | giant-stitch all tasks   |        100 |        78 |          0.78  |        744.5 | 0           | 42/43       |
| Hybrid-disabled control        | giant-stitch all tasks   |        100 |        78 |          0.78  |        749   | 0           | 42/43       |
| TMD exec graph                 | medium-stitch all tasks  |         50 |        25 |          0.5   |        702.2 | 0           | 42          |
| TMD exec graph q98 scale15     | giant-navigate all tasks |        100 |        42 |          0.42  |        884.6 | 0           | 42/43       |
| TMD actor rescue               | medium-navigate task3    |        100 |        63 |          0.63  |        495.2 | 0           | 42/43       |
| GAS actor rescue same trigger  | medium-navigate task3    |        100 |        95 |          0.95  |        217.8 | 0           | 42/43       |

### 1.3 主要 delta 与不确定性

下面的 95% CI 是简单两比例近似，仅用于快速判断方向；正式报告必须使用 seed/task 分层 bootstrap 或 paired analysis。

| comparison                           |   baseline_n |   baseline_success |   variant_n |   variant_success |   delta_success | approx_95ci_delta   |   delta_mean_steps |
|:-------------------------------------|-------------:|-------------------:|------------:|------------------:|----------------:|:--------------------|-------------------:|
| giant-navigate: cost w=0.25 vs GAS   |          400 |              0.642 |         400 |             0.73  |           0.088 | [+0.023, +0.152]    |              -24   |
| giant-navigate: cost w=0.50 vs GAS   |          400 |              0.642 |         400 |             0.67  |           0.028 | [-0.038, +0.093]    |               -0.6 |
| medium-stitch: rescue vs GAS         |          500 |              0.978 |         500 |             0.986 |           0.008 | [-0.008, +0.024]    |               -3.9 |
| medium-navigate task3: rescue vs GAS |          400 |              0.965 |         400 |             0.97  |           0.005 | [-0.020, +0.030]    |               -3.3 |
| giant-stitch: rescue b50/r3 vs GAS   |          100 |              0.78  |         100 |             0.78  |           0     | [-0.115, +0.115]    |               -5.8 |

解释：

- `giant-navigate: cost w=0.25 vs GAS` 是当前唯一明显达到“可作为主线继续推进”的信号：成功率从 `0.642` 到 `0.730`，delta 约 `+8.8pp`，mean steps 降低约 `24.1`。
- `w=0.50` 成功率只从 `0.642` 到 `0.670`，mean steps 基本不变，说明 TMD cost 有信息但不能主导 route selection。
- medium 环境上的 rescue 改善很小，且 baseline 已接近饱和，不能作为强证据。
- giant-stitch rescue 与 hybrid-disabled control 持平，不能证明 TMD action 本身有效。

### 1.4 giant-navigate 上 `w=0.25` 的 task-wise 效果

|   task_id |   episodes_per_variant |   GAS_success |   GAS_steps |   w025_success |   w025_steps |   delta_success |   delta_steps |
|----------:|-----------------------:|--------------:|------------:|---------------:|-------------:|----------------:|--------------:|
|         1 |                     80 |         0.138 |       987.1 |          0.225 |        979.3 |           0.087 |          -7.8 |
|         2 |                     80 |         0.825 |       758.4 |          0.887 |        746.4 |           0.062 |         -12   |
|         3 |                     80 |         0.725 |       845.3 |          0.875 |        814.1 |           0.15  |         -31.2 |
|         4 |                     80 |         0.812 |       777.5 |          0.912 |        719.6 |           0.1   |         -57.9 |
|         5 |                     80 |         0.713 |       648.7 |          0.75  |        637.6 |           0.037 |         -11.1 |

该表是最重要的正向证据：`w=0.25` 不是只靠单个 task 偶然变好，而是在 5 个 task 上均有 success 提升。但 task1 仍然很难，说明后续应重点分析 hard-task route choice 和 final-goal handoff。

### 1.5 Rescue conditional summary

| run                                         |   episodes |   activated_n | activated_success   | non_activated_success   |   mean_first_trigger_step |
|:--------------------------------------------|-----------:|--------------:|:--------------------|:------------------------|--------------------------:|
| medium-stitch s200/p15 all tasks rescue_log |        250 |             2 | 0/2                 | 243/248                 |                     201   |
| medium-navigate s175/p9 task3 gas_seed42    |        200 |             7 | 1/7                 | 191/193                 |                     176   |
| medium-navigate s175/p9 task3 gas_seed43    |        200 |             4 | 0/4                 | 196/196                 |                     176   |
| giant-stitch 100keff q98 b50/r3 gas_seed42  |         50 |             6 | 0/6                 | 38/44                   |                     850   |
| giant-stitch 100keff q98 b50/r3 gas_seed43  |         50 |             4 | 0/4                 | 40/46                   |                     814.8 |

结论：当前 rescue 不能只看 aggregate success。尤其 giant-stitch 的 b50/r3 rescue，activated subset 是 `0/6` 和 `0/4`，所以它没有证明“能救回失败 episode”。后续必须做 trigger-state replay。

### 1.6 Graph diagnostics 摘要

| env                        |   graph_rows | nodes   | mean_out_degree   | largest_scc_ratio   | edge_threshold_range   | directed_asymmetry_mean   |
|:---------------------------|-------------:|:--------|:------------------|:--------------------|:-----------------------|:--------------------------|
| antmaze-giant-navigate-v0  |            3 | 297-415 | 9.20-30.11        | 0.734-0.841         | 2.00381-3.03884        | 0.288991-0.884137         |
| antmaze-giant-stitch-v0    |            5 | 309-328 | 1.68-36.67        | 0.037-1.000         | 0.959612-28.9109       | 0.133725-1.41026          |
| antmaze-medium-navigate-v0 |            2 | 256-256 | 31.39-33.91       | 0.953-0.969         | 0.0011223-0.00118375   | 8.42314e-06-2.63996e-05   |
| antmaze-medium-stitch-v0   |           16 | 7-512   | 0.27-36.66        | 0.020-0.979         | 0.00110725-0.712409    | 8.34933e-06-0.0359718     |

解释：

- medium-stitch / medium-navigate 的 TMD edge threshold 大量处在 `1e-3` 量级，directed asymmetry 也非常小，提示 distance scale 可能塌缩或 calibration 不可直接用于 hard graph。
- giant-stitch 的 graph health 对 threshold 极敏感：有的配置 largest SCC 很小，有的配置过密。TMD hard graph 暂时只能做诊断线。

---

## 2. 核心研究问题与假设

### RQ1：TMD 是否应该作为 GAS 的 soft cost，而不是 hard replacement？

假设 H1：TMD distance/cost 对 route selection 有弱但有用的排序信息；最佳权重在 `0.20–0.30` 附近。过高权重会放大 calibration error。

### RQ2：低层 actor 能否通过通用 condition 更稳定地执行高层 subgoal？

假设 H2：使用 `[TDR direction, TDR distance scale, task-factor mask, masked task-factor residual]` 能提高 subgoal executability，特别是在 giant / hard-task / final-goal handoff 场景。

### RQ3：TMD rescue 是真正 rescue，还是只是 failure detector？

假设 H3：当前 rescue trigger 偏晚，TMD actor 与 GAS low-level action distribution 不匹配；真正有效的 rescue 更可能是“TMD-selected subgoal + GAS actor”或“earlier replan”，而不是直接切到 TMD actor。

### RQ4：TMD hard graph 失败来自 graph topology、distance calibration，还是 low-level execution？

假设 H4：主要原因是 calibration + low-level compatibility 双重问题。TMD hard graph 只有在 graph health 与 low-level executability 同时通过后才值得推广。

---

## 3. 实验可靠性原则

### 3.1 对照原则

每个主实验必须尽量匹配：

```text
env
数据集/checkpoint
任务集合 task_id
环境 seed env_seed
GAS seed gas_seed
episode budget
planner/replan schedule
fallback 设置
```

### 3.2 预注册结论标准

一个 variant 可被称为“可靠提升”，至少满足：

```text
success delta 的 95% CI lower bound > 0
且 task-wise 没有关键 task 明显退化
且 mean steps 没有显著恶化
且 failure analysis 没有发现系统性副作用
```

建议阈值：

```text
主线推广：aggregate success +3pp 以上，且 CI lower bound > 0
强推广：aggregate success +5pp 以上，且至少 70% task 非负提升
只作为诊断：aggregate 改善 <3pp 或 CI 跨 0
降级/停止：success 下降 >3pp，或 steps 增加 >5%，或 hard task 崩溃
```

### 3.3 统计分析

正式报告至少包含：

1. aggregate success / mean steps；
2. per-task success / steps；
3. per-seed 与 per-gas_seed delta；
4. success delta 的 Wilson/two-proportion CI；
5. stratified bootstrap CI，分层单位为 `(env_seed, gas_seed, task_id)`；
6. failed episode taxonomy；
7. rescue activated subset vs non-activated subset；
8. graph health 与 execution failure 的关联。

---

## 4. 后续实验阶段

### Phase A：复现与分析框架固定

目的：确保所有后续结论来自同一个 evaluation/analyze pipeline。

交付物：

```text
reports/stage26_tmd_tdr_manifest.md
reports/stage26_tmd_tdr_protocol.md
reports/stage26_tmd_tdr_eval_all.csv
reports/stage26_tmd_tdr_graph_all.csv
reports/stage26_tmd_tdr_summary.md
reports/stage26_tmd_tdr_failure_analysis.md
reports/stage26_tmd_tdr_decisions.md
```

必须记录：branch、commit、dirty state、Python/JAX/Torch/CUDA 版本、dataset path、checkpoint path、所有 command line、run_name、seed、task_id、episode budget。

---

### Phase B：确认 TMD soft cost blend

主线矩阵：

| env | variants | seeds | episodes |
|---|---|---:|---:|
| `antmaze-giant-navigate-v0` | GAS, `w ∈ {0.10,0.20,0.25,0.30,0.40,0.50}` | env_seed `(0, 1, 2)` × gas_seed `(42, 43)` | pilot `20/task`; confirm `50/task` |
| `antmaze-giant-stitch-v0` | 同上 | 同上 | 同上 |
| `antmaze-medium-navigate-v0` | GAS, `w ∈ {0.10,0.25}` | env_seed `(0, 1)` × gas_seed `(42, 43)` | `20/task` |
| `antmaze-medium-stitch-v0` | GAS, `w ∈ {0.10,0.25}` | env_seed `(0, 1)` × gas_seed `(42, 43)` | `20/task` |

重点分析：

- `w=0.25` 是否在更多 env_seed 上仍稳定；
- `w=0.20/0.30` 是否比 `0.25` 更稳；
- `w=0.50` 是否在 hard task 上放大 calibration error；
- 提升来自 no-path 减少、path 更短、subgoal switching 更少，还是 final-goal handoff 更好。

结论 gate：

```text
如果 w≈0.20–0.30 在 giant-navigate 和至少一个额外 hard env 上通过 CI 和 task-wise gate，
则 TMD cost blend 晋升为主线增强方法。
否则只保留为 giant-navigate-specific exploratory trick。
```

---

### Phase C：实现 universal low-level condition

目标 condition：

```text
cond(s, g_local, g_task)
=
[
  TDR direction(s, g_local),
  TDR distance scale(s, g_local),
  task-factor mask(task),
  task-factor mask(task) ⊙ task-factor residual(s, g_task)
]
```

关键设计：

| component | target | normalization |
|---|---|---|
| `TDR direction` | local subgoal | TDR latent diff unit-normalized |
| `TDR distance scale` | local subgoal | `log1p(d_raw / q90)` + clip |
| `task-factor mask` | final task semantics | binary/soft explicit mask |
| `masked residual` | final task goal | factor std normalization + clip |

注意：

- `g_local` 是 high-level planner 当前 subgoal/keynode；
- `g_task` 是 episode final task goal；
- 如果没有 final task goal，fallback 为 `g_task = g_local`，但必须在日志里标注；
- mask 必须显式输入 actor，否则 residual 为 0 无法区分“已满足”和“不相关”。

建议文件结构：

```text
bars/conditioning/
  __init__.py
  low_level_condition.py
  task_factors.py
  stats.py
scripts/
  stage26_fit_lowcond_stats.py
  stage26_train_lowcond_actor.py
  stage26_eval_policy.py
  stage26_analyze.py
configs/stage26_tdr_factor/
  *.yaml
```

最小 API：

```python
cond = builder.encode(
    obs=obs,
    local_target_obs=subgoal_obs,
    task_goal=final_task_goal,
    task_id=task_id,
    goal_info=info,
)
```

shape：

```text
cond_dim = D_z + 1 + D_f + D_f
```

如果跨 env 共享，使用 `D_f_max` padding：

```text
mask[:D_f] = real_mask
mask[D_f:] = 0
residual[:D_f] = real_residual
residual[D_f:] = 0
```

---

### Phase D：低层 condition ablation

先固定 GAS graph planner，只改变 low-level actor condition。

Ablation：

| name | condition |
|---|---|
| A | current/raw GAS low-level condition |
| B | `TDR direction + TDR distance scale` only |
| C | `task-factor mask + masked task-factor residual` only |
| D | full condition |
| E | full condition but residual uses local subgoal, not final task goal |
| F | full condition but no explicit mask |
| G | full condition but raw TDR distance, no log/q90 normalization |

初始环境：

```text
antmaze-medium-stitch-v0
antmaze-medium-navigate-v0
antmaze-giant-navigate-v0
antmaze-giant-stitch-v0
```

推广环境：

```text
pointmaze-medium/large/giant navigate/stitch
humanoidmaze-medium/large navigate/stitch
D4RL antmaze variants
kitchen / manipulation variants, if task-factor adapter is valid
```

判定逻辑：

- 如果 D > B，说明 task-factor residual 有贡献；
- 如果 B > A，说明 TDR geometry 被 actor 成功消费；
- 如果 F 崩，说明 mask 对 multi-task/task-factor ambiguity 很重要；
- 如果 G 崩，说明当前 TDR/TMD distance scale 需要 log/q90 normalization；
- 如果 D 仍不能超过 A，优先排查 subgoal obs representation、TDR encoder stats、factor adapter index、训练 horizon sampling。

---

### Phase E：组合最佳 cost blend 与最佳低层 condition

组合矩阵：

```text
GAS graph baseline
GAS graph + best TMD cost blend
GAS graph + best TDR-factor low-level condition
GAS graph + best TMD cost blend + best TDR-factor low-level condition
```

目标是判断：

1. cost blend 与 lowcond 是否互补；
2. cost blend 是否只改善 route，lowcond 是否改善 execution；
3. 如果组合变差，是否因为 high-level target distribution 与 low-level training distribution mismatch。

---

### Phase F：Rescue 诊断，而不是继续盲调

每个 rescue 实验必须包含四个 matched controls：

```text
GAS baseline
trigger/replan control without TMD action
TMD-selected subgoal + GAS actor
TMD actor rescue
```

必须保存 trigger-state replay 数据：

```text
env, task_id, env_seed, gas_seed, episode
trigger_step
obs
current_subgoal
final_goal
planned_path
TMD/TDR distance to subgoal
TMD/TDR distance to final goal
spatial distance to subgoal/final goal
remaining_horizon
outcome
```

Replay policies：

```text
GAS continue
GAS replan
TMD-selected subgoal + GAS actor
TMD actor rescue
oracle/local target if available
```

结论 gate：

```text
如果 activated-subset success 不能显著高于 trigger/replan control，
则 rescue 不作为主线；最多保留为 failure detector / replan trigger。
```

---

### Phase G：TMD hard graph replacement 保持诊断线

不要再大规模跑 all-task hard graph replacement，除非 graph health 与 low-level executability 同时通过。

小矩阵：

```text
env/task:
  medium-stitch task3
  giant-navigate task3/task5
  giant-stitch hard tasks

threshold:
  graph q95/q98/q99
  target t99/t995/t999

execution scale:
  12/15/18/21

low-level:
  GAS actor only
```

必须输出：

```text
SCC ratio
out-degree distribution
edge threshold
target threshold
directed asymmetry
no-path rate
path length
subgoal-reach rate
final-goal handoff failure
```

结论 gate：

```text
TMD hard graph 只有在 no-path 低、SCC 合理、subgoal reach 高、final-goal handoff 稳定时才进入主线。
```

---

## 5. Failure analysis taxonomy

所有失败 episode 至少归入一个类别：

| failure type | diagnosis signal | likely fix |
|---|---|---|
| `no_path_or_disconnected_graph` | no_path_rate 高、SCC 小 | graph threshold / keynode coverage |
| `overpermissive_graph_shortcut` | graph 很连通但 subgoal 不可达 | soft cost instead of hard edge；edge calibration |
| `subgoal_unreachable_by_low_actor` | path exists，但 local target 长时间无法接近 | low-level condition / target distribution |
| `final_goal_handoff_failure` | subgoal 到了，但 final goal mode steps 高 | final task residual / final target adapter |
| `rescue_trigger_too_late` | first_trigger_step 接近 horizon 末尾 | earlier trigger / failure predictor |
| `rescue_actor_mismatch` | TMD actor rescue 比 GAS actor control 差 | 不直接切 actor；改成 TMD-selected target |
| `tdr_distance_scale_collapse` | thresholds 极小或跨 env 不可比 | log/q90 calibration；pair calibration |
| `task_factor_mask_or_residual_error` | 特定 task/systematic failure | adapter index / mask mapping |
| `dataset_coverage_gap` | hard task 路径区域缺数据 | keynode coverage / dataset augmentation |

最终报告必须为主要 failure type 给出 trace 示例和经验规律，而不只是 aggregate metric。

---

## 6. 推荐最终 decision table 模板

| branch | promote? | evidence | caveat | next action |
|---|---|---|---|---|
| TMD cost blend | TBD | success delta, CI, task-wise | weight sensitivity | keep/broaden/tune |
| TDR-factor lowcond | TBD | execution/subgoal reach | training cost | integrate or revise |
| TMD rescue | likely diagnostic only | activated subset | trigger timing | replay diagnosis |
| TMD hard graph | diagnostic only now | graph health | calibration/execution | keep compact tests |
| TMD actor direct | deprioritize | actor mismatch | maybe checkpoint-specific | stop mainline |

---

## 7. Codex Goal Mode 指令

下面这段可以直接粘贴给 Codex 的 goal mode：

```text
You are Codex working inside the BARS repository. Goal: complete a rigorous Stage-26 experimental study of TMD/TDR as a replacement or augmentation for GAS, following `reports/stage26_tmd_tdr_experiment_plan.md` if present. If that file is absent, create it from this prompt and proceed.

Primary scientific objective
1. Determine whether TMD should augment GAS through soft cost blending rather than replace GAS graph topology.
2. Implement and evaluate a universal low-level condition for OGBench/D4RL-style tasks:
   `[TDR direction, TDR distance scale, task-factor mask, masked task-factor residual]`.
3. Diagnose why TMD rescue, TMD graph execution, or TMD actor execution succeeds or fails.
4. Produce conclusions that are statistically defensible, seed-matched, task-wise, and useful for future research.

Hard reliability rules
- Do not cherry-pick. Every run launched must be represented in a manifest, CSV, and summary report.
- Always compare variants against GAS baselines with the same env, env_seed, gas_seed, task_id set, and episode budget whenever possible.
- Do not call a result “improved” unless the pre-registered acceptance rule passes: success-rate delta lower 95% CI > 0, or a clearly labeled exploratory result with effect size and uncertainty.
- Pilot/smoke results may guide debugging but must not be presented as final evidence.
- If a planned matrix is incomplete because of resources or failures, label it incomplete, explain exactly what ran, and report only supported conclusions.
- Preserve existing artifacts. Do not delete or overwrite previous stage25/stage25-protocol-oracle-drift outputs.
- Keep `fallback=none` unless a fallback is explicitly part of a named control condition.
- When a variant fails, diagnose mechanism rather than just reporting the metric.

Repository setup
1. Start with `git status`, record branch, commit, dirty state, Python/CUDA/JAX/Torch versions, dataset paths, and checkpoint paths.
2. Create a new working branch named `stage26-tdr-factor-tmdcost` unless already on a suitable branch.
3. Inspect existing files under `bars/tmd_test`, `configs/tmd_test`, `scripts/tmd_test_*`, `reports/tmd_test_*`, and current GAS evaluation code before changing anything.
4. Create a reproducibility manifest at `reports/stage26_tmd_tdr_manifest.md`.

Implementation tasks
A. Add the low-level condition implementation:
- `bars/conditioning/low_level_condition.py`
- `bars/conditioning/task_factors.py`
- `bars/conditioning/stats.py`
Required API:
- `LowLevelConditionBuilder.encode(obs, local_target_obs, task_goal=None, task_id=None, goal_info=None) -> cond`
- output order: `[u_tdr, d_tdr_norm, task_mask, task_mask * residual]`
- `u_tdr`: unit-normalized direction in normalized TDR latent space from current obs to local subgoal.
- `d_tdr_norm`: log/quantile-normalized TDR distance scale, default `log1p(d_raw / q90)` clipped.
- `task_mask`: explicit binary/soft mask over task factors.
- `masked residual`: normalized goal-factor residual to the final task goal, not the local subgoal by default.
- Implement `MazeXYFactorAdapter`, `HumanoidMazeXYFactorAdapter`, and a configurable object-factor adapter for Kitchen/manipulation tasks.
- Support padding to a fixed `factor_dim_max` for cross-env training.
- TDR encoder must be frozen by default during actor training.

B. Add stats and training utilities:
- `scripts/stage26_fit_lowcond_stats.py`: fit z mean/std, TDR distance q90, factor mean/std from same-trajectory future pairs. Use log-uniform horizons `{1,2,4,8,16,32,64,128}` where valid.
- `scripts/stage26_train_lowcond_actor.py`: train or finetune low-level actor with BC/MSE or existing actor objective using the new condition.
- `scripts/stage26_eval_policy.py`: evaluate GAS graph planner using old condition, new condition, TMD cost blend, and combinations.
- `scripts/stage26_analyze.py`: aggregate CSVs, compute task-wise deltas, seed-wise deltas, Wilson/two-proportion CIs, stratified bootstrap CIs, rescue conditional summaries, graph-health summaries, and failure classifications.
- `scripts/stage26_run_matrix.sh` or a Python orchestrator: resumable, logs all commands, writes manifest rows before and after each run.

C. Add tests/smokes:
- Unit tests for condition shape, normalization, zero-distance behavior, mask behavior, final-goal-vs-local-subgoal residual behavior, and deterministic output for fixed inputs.
- Smoke evaluation on `antmaze-medium-stitch-v0` and `antmaze-medium-navigate-v0` before full sweeps.
- Fail fast if NaN/Inf appears in conditions, actions, distances, or graph weights.

Experiment matrix
Phase 1: confirm TMD soft-cost signal.
- Envs: `antmaze-giant-navigate-v0`, `antmaze-giant-stitch-v0`, plus medium sanity `antmaze-medium-navigate-v0`, `antmaze-medium-stitch-v0`.
- Variants: GAS graph baseline; GAS graph + TMD cost blend with `w ∈ {0.10,0.20,0.25,0.30,0.40,0.50}`.
- Seeds: env_seed `{0,1,2}`, gas_seed `{42,43}`.
- Episodes: pilot `20/task`; confirmation `50/task` for variants that pass pilot.
- Required tables: aggregate, task-wise, seed-wise paired deltas, step deltas, no_path_rate, final_goal_mode_steps, subgoal_switch_count.

Phase 2: evaluate universal low-level condition under fixed GAS graph.
- Keep high-level GAS graph and subgoal schedule fixed. Only change actor condition.
- Ablations:
  A. current/raw GAS low-level condition baseline.
  B. `TDR direction + TDR distance scale` only.
  C. `task-factor mask + masked task-factor residual` only.
  D. full condition.
  E. full condition but residual uses local subgoal instead of final task goal.
  F. full condition without explicit mask.
  G. full condition with raw TDR distance, no log/q90 normalization.
- Envs: start with `antmaze-medium-stitch-v0`, `antmaze-medium-navigate-v0`; then `antmaze-giant-navigate-v0`, `antmaze-giant-stitch-v0`; then broaden to available OGBench/D4RL PointMaze/HumanoidMaze/Kitchen where adapters are valid.
- Promote only if full condition improves or matches success while improving steps/executability, and no important task regresses catastrophically.

Phase 3: combine best low-level condition with TMD cost blend.
- Variants: GAS graph + best lowcond; GAS graph + best cost blend; GAS graph + best lowcond + best cost blend.
- Use the same seeds/tasks as Phase 1.
- Determine whether gains are additive, redundant, or conflicting.

Phase 4: rescue diagnostics, not blind tuning.
- For each rescue run include four matched controls: GAS baseline; trigger/replan control without TMD action; TMD-selected subgoal with GAS actor; TMD actor rescue.
- Save trigger-state replay data: env, task_id, env_seed, gas_seed, episode, trigger_step, obs, subgoal, final goal, planned path, TMD distance to subgoal/final goal, spatial distance to subgoal/final goal, remaining horizon, and outcome.
- Replay trigger states with: GAS continue, GAS replan, TMD-selected subgoal + GAS actor, TMD actor rescue, and oracle/local target if available.
- Report activated-subset success separately from non-activated success. Do not infer rescue works from aggregate success if activated episodes fail.

Phase 5: TMD graph replacement stays diagnostic.
- Do not promote TMD hard graph unless graph health and executability pass.
- Run only compact diagnostics: medium-stitch task3, giant-navigate task3/5, giant-stitch hard tasks.
- Sweep graph threshold `q95/q98/q99`, target threshold `t99/t995/t999`, execution scale `12/15/18/21`, low-level GAS actor only.
- Report graph health: SCC ratio, out-degree distribution, edge threshold, target threshold, directed asymmetry, path length, no-path rate, subgoal-reach rate, final-goal handoff failure.

Failure analysis taxonomy
Classify every failed episode into at least one bucket:
- no_path_or_disconnected_graph
- overpermissive_graph_shortcut
- subgoal_unreachable_by_low_actor
- final_goal_handoff_failure
- rescue_trigger_too_late
- rescue_actor_mismatch
- tdr_distance_scale_collapse
- task_factor_mask_or_residual_error
- dataset_coverage_gap
- environment_or_checkpoint_issue
Include examples/traces for each frequent failure type.

Required final outputs
- `reports/stage26_tmd_tdr_manifest.md`
- `reports/stage26_tmd_tdr_protocol.md`
- `reports/stage26_tmd_tdr_eval_all.csv`
- `reports/stage26_tmd_tdr_graph_all.csv`
- `reports/stage26_tmd_tdr_rescue_replay.csv` if rescue diagnostics run
- `reports/stage26_tmd_tdr_summary.md`
- `reports/stage26_tmd_tdr_failure_analysis.md`
- `reports/stage26_tmd_tdr_decisions.md`

Final report requirements
1. State what is supported, what is not supported, and what remains uncertain.
2. Separate primary confirmed results from exploratory/smoke results.
3. Include exact run counts, seeds, task IDs, episodes, success rates, mean steps, confidence intervals, and task-wise deltas.
4. Explain success and failure mechanisms using diagnostics, not just aggregate metrics.
5. End with concrete next-step rules, e.g. when to use TMD cost blend, when to avoid TMD hard graph, when rescue is worth revisiting, and how the low-level condition should be improved next.

Proceed end-to-end: implement, test, run the planned smoke and full matrices as resources allow, analyze, and produce the reports. If anything cannot be completed, record the blocker precisely and produce the best supported partial conclusion without overstating it.
```
