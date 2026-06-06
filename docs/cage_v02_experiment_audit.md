# CAGE-v0.2 实验审计报告

日期：2026-06-06

## 1. 审计对象

- 分支：`codex/cage-mvp`
- 最新提交：`17bc9c72f8a0ad2d943dbb7ee22bf142548c8ab6`
- 提交标题：`Add CAGE v0.2 contract commit pipeline`
- 远端状态：`origin/codex/cage-mvp` 已包含该提交
- 审计报告来源：
  - `reports/stage32_contract_commit_summary.md`
  - `reports/stage32_contract_commit_smoke.md`
  - `reports/stage32_contract_commit_minipilot.md`
  - `results/cage_v02_contract/models/eval_metrics.json`
  - `results/cage_v02_contract_commit/smoke/compact_summary.md`
  - `results/cage_v02_contract_commit/minipilot/compact_summary.md`

## 2. 功能完成度审计

| 任务 | 状态 | 证据 |
|---|---|---|
| 合同模型接口 | DONE | `external_src/GAS/cage/contract_model.py` 中 `ContractScorer` 输出 `predicted_hit`, `predicted_contract_positive`, `predicted_negative_progress`, `uncertainty`, `lower_confidence_bound` |
| 合同 gate | DONE | `external_src/GAS/cage/closed_loop_contracts.py` 中 `evaluate_contract_gate` |
| `cage_contract_commit` 变体 | DONE | `external_src/GAS/evaluate_gas.py`, `scripts/build_cage_eval_command.py`, `external_src/GAS/cage/state_machine.py` |
| 子目标切换合同门控 | DONE | `CAGEController._contract_gate` 在 normal/final/recovery 分支调用 |
| 承诺执行 | DONE | CAGE 已保留 min commitment；v0.2 使用 `cage_contract_min_commit_steps` |
| 阶段化合同 | DONE | final-goal 使用 `cage_contract_final_goal_threshold`；recovery 使用 `cage_contract_recovery_threshold` |
| held-out split | DONE | `scripts/build_contract_dataset_splits.py` 输出 train/val/test 和 split summary |
| 合同模型训练/评估 | DONE | `train_cage_contract.py`, `evaluate_cage_contract.py` 支持 split、AUROC、Brier、calibration、selective curve |
| step-level trace | DONE | CAGE debug trace 包含合同分数、拒绝原因、fallback 原因、StateRef 摘要 |
| smoke/minipilot 报告 | DONE | `reports/stage32_contract_commit_*.md` |
| SOTA 计划 | DONE | `docs/cage_sota_plan.md` |

## 3. 合同数据和模型审计

数据来源为 CLP1 closed-loop contract dataset：

- split 输入：`results/cage_clp1/datasets/closed_loop_contracts.jsonl`
- split 输出：`results/cage_v02_contract/splits/`
- split status：`ok`
- 总记录：9216

模型输出：

- 模型：`results/cage_v02_contract/models/contract_model.json`
- 指标：`results/cage_v02_contract/models/eval_metrics.json`
- 报告：`results/cage_v02_contract/models/eval_report.md`

测试集有效特征样本数为 192。关键指标：

| 指标 | 模型 | d_phi baseline |
|---|---:|---:|
| hit AUROC | 1.000 | 0.997 |
| contract-positive AUROC | 0.954 | 0.873 |
| negative-progress AUROC | 0.719 | 0.339 |
| hit Brier | 0.020 | NA |
| contract-positive Brier | 0.079 | NA |
| negative-progress Brier | 0.222 | NA |

解释：

- negative-progress 预测确实优于 d_phi-only baseline。
- 但 test 有效样本只有 192，且数据来自当前 CLP1 小规模 probe；不能作为 SOTA 级泛化证据。
- 当前模型可用于 smoke/minipilot 的诊断 gate，不应作为最终方法证据。

## 4. Smoke 审计

范围：

- env：`antmaze-giant-navigate-v0`
- seed：42
- budget：`episodes_per_goal=1`, `goals_per_env=1`
- variants：`gas`, `cage_trace_only`, `cage_fixed_commit`, `cage_safe_full`, `cage_contract_commit`

结果：

| variant | status | success | replans | segment reach | contract loaded | gate reject rate |
|---|---|---:|---:|---:|---:|---:|
| gas | succeeded | 0.000 | NA | NA | NA | NA |
| cage_trace_only | succeeded | 0.000 | 0.000 | 0.078 | 0.000 | 0.000 |
| cage_fixed_commit | succeeded | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cage_safe_full | succeeded | 0.000 | 5.000 | 0.011 | 0.000 | 0.000 |
| cage_contract_commit | succeeded | 0.000 | 0.000 | 0.457 | 1.000 | 0.897 |

Smoke 结论：

- 所有 job 返回 0。
- GAS 命令未包含 `--use_cage`。
- `cage_contract_commit` 命令包含 `--use_cage --cage_contract_commit`。
- `cage_contract_commit` 消除了 smoke 中的 CAGE-induced replans，但该样本成功率全为 0，不能判断性能提升。

## 5. Minipilot 审计

范围：

- env/seed：
  - `antmaze-giant-navigate-v0:42`
  - `antmaze-giant-stitch-v0:42`
  - `humanoidmaze-large-navigate-v0:44`
- budget：`episodes_per_goal=5`, `goals_per_env=5`
- variants：`gas`, `cage_trace_only`, `cage_fixed_commit`, `cage_safe_full`, `cage_contract_commit`

AntMaze 结果：

| env | variant | status | success | switches | stall | replans | fallback steps | segment reach | gate reject rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | gas | succeeded | 0.640 | NA | NA | NA | NA | NA | NA |
| antmaze-giant-navigate-v0 | cage_trace_only | succeeded | 0.640 | 113.160 | 0.000 | 0.000 | 0.000 | 0.053 | 0.000 |
| antmaze-giant-navigate-v0 | cage_safe_full | succeeded | 0.680 | 80.120 | 10.800 | 5.560 | 0.000 | 0.012 | 0.000 |
| antmaze-giant-navigate-v0 | cage_contract_commit | succeeded | 0.360 | 39.760 | 19.040 | 0.000 | 9.200 | 0.197 | 0.517 |
| antmaze-giant-stitch-v0 | gas | succeeded | 0.800 | NA | NA | NA | NA | NA | NA |
| antmaze-giant-stitch-v0 | cage_trace_only | succeeded | 0.800 | 135.880 | 0.000 | 0.000 | 0.000 | 0.044 | 0.000 |
| antmaze-giant-stitch-v0 | cage_safe_full | succeeded | 0.800 | 72.200 | 8.720 | 5.720 | 0.000 | 0.009 | 0.000 |
| antmaze-giant-stitch-v0 | cage_contract_commit | succeeded | 0.000 | 12.760 | 51.440 | 0.000 | 14.840 | 0.848 | 0.998 |

Humanoid 状态：

- `humanoidmaze-large-navigate-v0` GAS job failed。
- 后续 CAGE variants 未运行。
- 失败原因为 exact StateRef/debug contract trace 写入触发 `OSError: [Errno 122] Disk quota exceeded`。
- 该失败属于 trace 基础设施/磁盘配额问题，不是算法性能结论。

## 6. Gate 审计

| Gate | 状态 | 证据 |
|---|---|---|
| trace-only parity | PASS on AntMaze, BLOCKED on Humanoid | AntMaze 两个 env 中 `cage_trace_only` 成功率与 GAS 完全一致；humanoid 未完成 |
| churn reduction | PASS on AntMaze | `cage_contract_commit` global replan 为 0，低于 `cage_safe_full` 的约 5.6 |
| success safety | FAIL | antmaze-nav 从 0.64 降到 0.36；antmaze-stitch 从 0.80 降到 0.00 |
| failure-dense improvement | BLOCKED | humanoid 因 disk quota 未完成；teleport 未运行 |
| contract validity | INCONCLUSIVE | held-out negative-progress 优于 d_phi baseline，但在线 gate 过保守导致成功率回退 |

## 7. 失败模式判断

当前 `cage_contract_commit` 的主要失败不是：

- graph no-path；
- global replan storm；
- trace-only instrumentation 破坏；
- policy action interface 被改坏。

主要失败是：

- contract gate 在线部署过保守；
- AntMaze stitch 上 gate reject rate 达到 0.998；
- 大量拒绝导致保持/回退和 stall 增多；
- fallback 使用率上升，但成功率没有提高，因此不能算 planner 改进。

这说明 CAGE-v0.2 的方向是可诊断的，但当前合同模型/阈值/target-mode 泛化不足，不能作为主结果扩展。

## 8. 是否建议进入大规模 benchmark

不建议。

原因：

1. 标准 AntMaze 上 `cage_contract_commit` 明显低于 GAS。
2. Humanoid failure-dense 环境因 trace quota 未完成。
3. 合同模型 test 样本有效数较小，且在线 gate 显示过保守。
4. 当前正向结果主要是 churn reduction，不是 success safety。

下一阶段应先做合同 gate calibration 和无 debug/no exact StateRef 重测，不应启动 8-env SOTA benchmark。

## 9. 推荐下一步

先复查不带 exact StateRef 大 trace 的 AntMaze 5x5 结果，确认 success 回退是否来自在线 gate 逻辑本身，而不是 debug/trace 开销：

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_v02_contract_commit/recheck_antmaze_no_debug \
  --envs antmaze-giant-navigate-v0 antmaze-giant-stitch-v0 \
  --seeds 42 \
  --variants gas cage_trace_only cage_safe_full cage_contract_commit \
  --episodes_per_goal 5 \
  --goals_per_env 5 \
  --manifest_path results/cage_v02_contract_commit/recheck_antmaze_no_debug/manifests/recheck_manifest.jsonl \
  --strict_paths \
  --cage_contract_model_path /mnt/project/BARS/results/cage_v02_contract/models/contract_model.json

/root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path results/cage_v02_contract_commit/recheck_antmaze_no_debug/manifests/recheck_manifest.jsonl \
  --max_jobs 8
```

如果 recheck 仍然回退，应暂停 benchmark 扩展，进入 CAGE-v0.3：合同校准与阈值预注册。若 recheck 恢复安全性，再重跑 humanoid，并禁用大型 exact StateRef trace。

## 10. PR/commit 清单

当前已推送提交：

- `17bc9c7 Add CAGE v0.2 contract commit pipeline`

该提交包含：

- CAGE-v0.2 算法代码；
- 合同模型接口；
- held-out split / train / evaluate 脚本；
- step-level contract trace 字段；
- smoke/minipilot compact summaries；
- stage32 中文报告；
- contract gate / split / contract commit 单测。

本审计报告为追加文档，不改变算法行为。
