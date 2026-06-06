# Stage34 CAGE-v0.4 Contract-Intervention 总结

## 本轮目标

实现并验证 CAGE-v0.4：GAS 锚定的合同干预算法。目标不是扩大 benchmark，而是在 Stage33 `cage_contract_rank` 低于 GAS 后，验证“只在必要时替换 GAS 目标”的保守干预是否能提升 safety。

## 代码改动摘要

- 新增 `cage_contract_shadow_rank`：构造合同候选并记录 ranker 会如何接管，但实际执行 GAS 原始目标。
- 新增 `cage_contract_intervene`：默认执行 GAS，只有 GAS 合同高风险或停滞、且替代目标 intervention gain 过阈值时才替换。
- 新增 `external_src/GAS/cage/contract_intervention.py`：独立实现干预收益和决策。
- 为 CAGEController 增加 committed target progress watchdog、final-goal phase preservation、shadow/intervention episode diagnostics。
- 扩展 evaluator flags、manifest/command builder、aggregator 和 deployment analyzer。
- 新增 v0.4 单元测试。

## 验证命令

```bash
python -m py_compile external_src/GAS/evaluate_gas.py external_src/GAS/O_utils/evaluation.py external_src/GAS/cage/*.py scripts/analyze_contract_intervention_deployment.py scripts/build_cage_eval_command.py scripts/cage_experiment_manifest.py scripts/aggregate_cage_experiments.py
pytest tests/test_cage_contract_ranker.py tests/test_cage_contract_rank_variant.py tests/test_cage_contract_shadow_rank.py tests/test_cage_contract_intervention.py tests/test_committed_progress_watchdog.py tests/test_final_phase_preservation.py -q
pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py -q
pytest tests/test_cage_state_machine.py tests/test_cage_subgoal_selector.py tests/test_cage_trace_only.py tests/test_cage_churn_guard.py -q
```

结果：

- py_compile：返回 0。
- v0.4 tests：11 passed。
- manifest/aggregation tests：4 passed。
- existing CAGE tests：17 passed。

## Shadow Rank 结果

修正后的 shadow parity 输出：

- `results/cage_v04_contract_intervene/shadow_antmaze_rank_parity/`
- `reports/stage34_contract_shadow_rank_antmaze.md`

| env | GAS | trace_only | shadow_rank | shadow override on success | shadow final override |
|---|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | 0.680 | 0.640 | 0.600 | 0.100 | 0.004 |
| antmaze-giant-stitch-v0 | 0.800 | 0.840 | 0.800 | 0.198 | 0.058 |

Shadow gate 结论：PASS，但 stitch 接近 0.20 阈值，说明 ranker 在成功轨迹上仍偏积极。

## Smoke 结果

输出：

- `results/cage_v04_contract_intervene/smoke/`
- `reports/stage34_contract_intervene_smoke.md`

5 个 job 全部返回 0。1×1 smoke 成功率全为 0，不做性能结论，仅确认新 flags、trace 和 evaluator 接口可运行。

## AntMaze Minipilot 结果

输出：

- `results/cage_v04_contract_intervene/minipilot_antmaze/`
- `reports/stage34_contract_intervene_minipilot_antmaze.md`

| env | GAS | trace_only | safe_full | contract_rank | contract_intervene |
|---|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | 0.640 | 0.680 | 0.680 | 0.440 | 0.560 |
| antmaze-giant-stitch-v0 | 0.840 | 0.800 | 0.800 | 0.640 | 0.720 |

`cage_contract_intervene` 比 `contract_rank` 更安全：

- nav：0.560 vs 0.440
- stitch：0.720 vs 0.640
- committed source rate 从 Stage33 的 nav 0.480 / stitch 0.660 降到 nav 0 / stitch 0.010。
- global replan request count 为 0，未重现 replan storm。

但 success safety 仍失败：

- nav 比 GAS 低 8pp。
- stitch 比 GAS 低 12pp。

## Gate 状态

| Gate | 状态 | 依据 |
|---|---|---|
| trace-only parity | PASS | nav +4pp，stitch -4pp，无系统性退化 |
| shadow safety | PASS / BORDERLINE | stitch shadow_override_on_success_rate=0.198，接近 0.20 |
| success safety | FAIL | contract_intervene 低于 GAS 8pp/12pp |
| intervention discipline | PARTIAL / FAIL | nav intervention_rate=0.013，stitch=0.322，高于建议 0.30 |
| committed control | PASS | committed source rate 显著低于 Stage33 contract_rank |
| final-goal preservation | PARTIAL | final_goal_on_rate 低于 trace_only/safe_full，需继续诊断 |
| churn safety | PASS | global replans 为 0 |
| forward progress | PASS / INCONCLUSIVE | 未检测到 local_safe_loop，但 success 仍不足 |
| failure-dense readiness | FAIL | AntMaze success safety 未过 |

## 解释

CAGE-v0.4 修复了 Stage33 的主要执行形态问题：不再大量选择 committed target，也没有重规划风暴。它比 `cage_contract_rank` 更接近 GAS，但仍没有通过 AntMaze success safety。当前问题已经从“硬拒绝”或“committed 过用”转为更细粒度的干预质量问题：在 stitch 中 intervention rate 仍偏高，final-goal 推进也没有稳定追平 trace_only/safe_full。

因此，v0.4 不是可进入 humanoid/teleport 或大规模 SOTA benchmark 的版本。

## 下一步

建议下一阶段不要扩大 benchmark，而是做 CAGE-v0.5 的干预校准：

1. 分析 intervention step-level traces，区分成功/失败 episode 中的干预位置、target mode、final phase 和 path progress。
2. 预注册更保守的 intervention variant，例如提高 `contract_intervention_margin`、降低允许干预的 final phase 范围、对 stitch 的非 GAS 替换加更强 path-index/final-progress 约束。
3. 先只重跑 AntMaze minipilot，不进入 humanoid/teleport。

推荐下一条命令：

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/analyze_contract_intervention_deployment.py \
  --input_roots results/cage_v04_contract_intervene/shadow_antmaze_rank_parity results/cage_v04_contract_intervene/minipilot_antmaze \
  --out_csv results/cage_v04_contract_intervene/deployment_analysis.csv \
  --out_md reports/stage34_contract_intervention_deployment.md
```
