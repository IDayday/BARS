# Stage33 CAGE-v0.3 Contract-Rank 总结

## 本轮目标

实现并验证 `cage_contract_rank`：校准覆盖的前进型合同执行算法。目标是修复 Stage32 `cage_contract_commit` 的 hard gate 过度拒绝问题，在不改 GAS backbone、不改低层 policy、不启动大规模 benchmark 的前提下，验证合同排序是否能降低拒绝率并保持执行安全。

## 代码改动摘要

- 新增 `cage_contract_rank` variant。
- 新增 `external_src/GAS/cage/contract_ranker.py`，实现候选合同排序、coverage floor、GAS margin discipline 和 extreme-negative-only hard reject。
- 扩展 CAGE config、evaluator flags、manifest/command builder/aggregator。
- 扩展 `CAGEController`：保留 v0.2 hard gate，不改变 `cage_contract_commit`；新增 v0.3 rank 分支和 episode-level ranking diagnostics。
- 增加 light debug trace：`cage_debug_light`, `cage_disable_exact_state_ref_trace`, `cage_max_debug_steps_per_episode`, `cage_trace_phi_vectors`。
- 修复 `run_cage_manifest.py`：使用调用脚本的 Python 解释器，并新增 `--parallel_jobs`/resume skip succeeded。
- 新增 Stage33 单元测试和部署分析脚本。

## 验证命令

```bash
python -m py_compile external_src/GAS/evaluate_gas.py external_src/GAS/O_utils/evaluation.py external_src/GAS/cage/*.py scripts/analyze_contract_rank_deployment.py scripts/run_cage_manifest.py scripts/build_cage_eval_command.py scripts/cage_experiment_manifest.py scripts/aggregate_cage_experiments.py scripts/run_contract_capture_smoke.py
pytest tests/test_cage_contract_gate.py tests/test_cage_contract_commit.py tests/test_cage_contract_ranker.py tests/test_contract_rank_coverage.py tests/test_debug_light_trace.py -q
pytest tests/test_cage_experiment_manifest.py tests/test_cage_aggregation.py tests/test_cage_state_machine.py tests/test_cage_subgoal_selector.py tests/test_cage_trace_only.py tests/test_cage_churn_guard.py -q
```

结果：

- py_compile：返回 0。
- Stage33 tests：11 passed。
- 既有 CAGE tests：21 passed。

## Recheck 结果

no-debug recheck 证明 v0.2 回退不是 debug/StateRef 开销导致，而是 hard gate 逻辑本身：

| env | gas | trace_only | safe_full | contract_commit | commit reject |
|---|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | 0.600 | 0.640 | 0.680 | 0.400 | 0.519 |
| antmaze-giant-stitch-v0 | 0.800 | 0.800 | 0.800 | 0.000 | 0.998 |

## Smoke 结果

见：`reports/stage33_contract_rank_smoke.md`

- 5 个 job 全部返回 0。
- `cage_contract_rank` coverage 1.000，replans 0，gate reject 0。
- smoke 成功率全为 0，不做性能结论。

## AntMaze Minipilot 结果

见：`reports/stage33_contract_rank_minipilot_antmaze.md`

| env | gas | trace_only | safe_full | contract_commit | contract_rank |
|---|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | 0.600 | 0.600 | 0.720 | 0.400 | 0.440 |
| antmaze-giant-stitch-v0 | 0.800 | 0.800 | 0.800 | 0.000 | 0.640 |

`cage_contract_rank` 修复了 v0.2 的过度拒绝：

- nav coverage：1.000，reject：0.000
- stitch coverage：0.992，reject：0.000

但 success safety 仍失败：

- nav 比 GAS 低 16pp。
- stitch 比 GAS 低 16pp。

## Stage33 Gate

| Gate | 状态 |
|---|---|
| trace-only parity | PASS |
| success safety | FAIL |
| rank coverage | PASS |
| churn safety | PASS |
| forward progress | PARTIAL |
| GAS replacement discipline | PASS by implementation, NEEDS MORE DATA |
| failure-dense readiness | FAIL |

## 并行和 GPU 利用率

- 已启动 `/mnt/project/gpu_stress.py` 维持 GPU utilization，PID：`logs/gpu_stress_stage33_contract_rank.pid`。
- 当前两张 A800 utilization 约 100%。
- `run_cage_manifest.py` 已支持 `--parallel_jobs`，本轮剩余 4 个 stitch CAGE jobs 使用 `--parallel_jobs 4` 并行补跑。

## 是否进入更大规模 benchmark

不建议。`cage_contract_rank` 是对 v0.2 的明确修复，但 AntMaze success safety 未过。下一步不应进入 humanoid/teleport 或 8-env benchmark，应继续做 contract rank 的选择偏差诊断。

## 下一步命令

推荐先做部署诊断，不扩大 benchmark：

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/analyze_contract_rank_deployment.py \
  --input_root results/cage_v03_contract_rank/minipilot_antmaze \
  --out_csv results/cage_v03_contract_rank/minipilot_antmaze/deployment_analysis.csv \
  --out_md reports/stage33_contract_rank_deployment.md
```
