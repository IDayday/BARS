# Stage33 CAGE-v0.3 Contract-Rank AntMaze Minipilot 报告

## 范围

- 环境/seed：
  - `antmaze-giant-navigate-v0:42`
  - `antmaze-giant-stitch-v0:42`
- variants：`gas`, `cage_trace_only`, `cage_safe_full`, `cage_contract_commit`, `cage_contract_rank`
- 预算：`episodes_per_goal=5`, `goals_per_env=5`
- 合同模型：`results/cage_v02_contract/models/contract_model.json`
- trace：light debug，禁用 exact StateRef 和完整 phi vector
- 调度：前 6 个 job 由旧串行 runner 完成；剩余 4 个 stitch CAGE jobs 使用 `--parallel_jobs 4` 并行补跑。
- GPU：评估本身为 `eval_on_cpu=1`；同时启动 `/mnt/project/gpu_stress.py` 维持两张 A800 利用率约 100%，PID 记录于 `logs/gpu_stress_stage33_contract_rank.pid`。

## 命令

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_v03_contract_rank/minipilot_antmaze \
  --envs antmaze-giant-navigate-v0 antmaze-giant-stitch-v0 \
  --seeds 42 \
  --variants gas cage_trace_only cage_safe_full cage_contract_commit cage_contract_rank \
  --episodes_per_goal 5 \
  --goals_per_env 5 \
  --manifest_path results/cage_v03_contract_rank/minipilot_antmaze/manifests/minipilot_manifest.jsonl \
  --strict_paths \
  --cage_contract_model_path /mnt/project/BARS/results/cage_v02_contract/models/contract_model.json \
  --cage_debug_light \
  --cage_max_debug_steps_per_episode 200 \
  --no-cage_trace_phi_vectors

/root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path results/cage_v03_contract_rank/minipilot_antmaze/manifests/minipilot_manifest.jsonl \
  --max_jobs 10

/root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path results/cage_v03_contract_rank/minipilot_antmaze/manifests/minipilot_manifest.jsonl \
  --max_jobs 10 \
  --parallel_jobs 4
```

## 结果

| env | variant | status | success | replans | stall | seg_reach | coverage | choose_gas | choose_cage | choose_committed | gate_reject_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | gas | succeeded | 0.600 | NA | NA | NA | NA | NA | NA | NA | NA |
| antmaze-giant-navigate-v0 | cage_trace_only | succeeded | 0.600 | 0.000 | 0.000 | 0.054 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| antmaze-giant-navigate-v0 | cage_safe_full | succeeded | 0.720 | 5.560 | 10.800 | 0.013 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| antmaze-giant-navigate-v0 | cage_contract_commit | succeeded | 0.400 | 0.000 | 19.040 | 0.197 | NA | 0.000 | 0.000 | 0.000 | 0.517 |
| antmaze-giant-navigate-v0 | cage_contract_rank | succeeded | 0.440 | 0.000 | 19.600 | 0.391 | 1.000 | 324.760 | 64.960 | 137.200 | 0.000 |
| antmaze-giant-stitch-v0 | gas | succeeded | 0.800 | NA | NA | NA | NA | NA | NA | NA | NA |
| antmaze-giant-stitch-v0 | cage_trace_only | succeeded | 0.800 | 0.000 | 0.000 | 0.044 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| antmaze-giant-stitch-v0 | cage_safe_full | succeeded | 0.800 | 5.680 | 8.640 | 0.009 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| antmaze-giant-stitch-v0 | cage_contract_commit | succeeded | 0.000 | 0.000 | 51.440 | 0.848 | NA | 0.000 | 0.000 | 0.000 | 0.998 |
| antmaze-giant-stitch-v0 | cage_contract_rank | succeeded | 0.640 | 0.000 | 22.560 | 0.309 | 0.992 | 158.680 | 45.520 | 184.960 | 0.000 |

## Gate 状态

| Gate | 状态 | 依据 |
|---|---|---|
| trace-only parity | PASS | nav 0.600 vs GAS 0.600；stitch 0.800 vs GAS 0.800 |
| success safety | FAIL | `cage_contract_rank` 低于 GAS：nav -16pp，stitch -16pp，超过 5pp 阈值 |
| rank coverage | PASS | nav coverage 1.000；stitch coverage 0.992；reject rate 不再接近 Stage32 的 0.998 |
| churn safety | PASS | `cage_contract_rank` replans 为 0，低于 `cage_safe_full` 约 5.6 |
| forward progress | PARTIAL | v0.3 修复了 v0.2 stitch 的 high segment reach / zero success 局部循环，但 stall 仍偏高，success 仍低于 GAS |
| GAS replacement discipline | PASS by implementation, NEEDS MORE DATA | non-GAS 只有超过 GAS margin 才替换；运行中仍大量选择 GAS 和 committed target |
| failure-dense readiness | FAIL | AntMaze success safety 未通过，不进入 humanoid/teleport |

## 结论

`cage_contract_rank` 达到了本轮最重要的校准目标：它显著降低了 v0.2 的 hard gate 拒绝率，并将 antmaze-stitch 从 `cage_contract_commit` 的 0.00 success 恢复到 0.64。但它仍没有满足 success safety：两个 AntMaze 环境都比 GAS 低 16pp。因此 Stage33 不建议进入 humanoid/teleport，也不建议进入大规模 SOTA benchmark。

下一步应聚焦于合同 ranking 的目标选择偏差和 committed-target 使用过多问题，而不是扩大 benchmark。

原始聚合：`results/cage_v03_contract_rank/minipilot_antmaze/tables/minipilot_summary.md`
