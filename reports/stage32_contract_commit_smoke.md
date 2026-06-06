# Stage32 CAGE-v0.2 Smoke 报告

## 范围

- 环境：`antmaze-giant-navigate-v0`
- seed：42
- variants：`gas`, `cage_trace_only`, `cage_fixed_commit`, `cage_safe_full`, `cage_contract_commit`
- 预算：`episodes_per_goal=1`, `goals_per_env=1`
- Python：`/root/miniconda3/envs/gcrlo/bin/python`
- 合同模型：`results/cage_v02_contract/models/contract_model.json`

## 命令

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/run_contract_capture_smoke.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_v02_contract_commit/smoke \
  --env_seed_pairs antmaze-giant-navigate-v0:42 \
  --variants gas cage_trace_only cage_fixed_commit cage_safe_full cage_contract_commit \
  --episodes_per_goal 1 \
  --goals_per_env 1 \
  --status_path results/cage_v02_contract_commit/smoke/status.jsonl \
  --cage_debug \
  --cage_contract_model_path /mnt/project/BARS/results/cage_v02_contract/models/contract_model.json
```

## 结果

| variant | status | success | switches | stall | replans | fallback_steps | seg_reach | contract_loaded | gate_reject_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gas | succeeded | 0.000 | NA | NA | NA | NA | NA | NA | NA |
| cage_trace_only | succeeded | 0.000 | 140.000 | 0.000 | 0.000 | 0.000 | 0.078 | 0.000 | 0.000 |
| cage_fixed_commit | succeeded | 0.000 | 82.000 | 34.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cage_safe_full | succeeded | 0.000 | 91.000 | 12.000 | 5.000 | 0.000 | 0.011 | 0.000 | 0.000 |
| cage_contract_commit | succeeded | 0.000 | 34.000 | 42.000 | 0.000 | 15.000 | 0.457 | 1.000 | 0.897 |

## Smoke 结论

- 所有 smoke job 返回 0。
- GAS 命令未包含 `--use_cage`。
- `cage_contract_commit` 命令包含 `--use_cage --cage_contract_commit`。
- trace-only 在该 1 episode 样本上与 GAS 成功率一致，但样本过小。
- `cage_contract_commit` 在 smoke 中把 global replan 从 safe_full 的 5 降到 0，同时大量拒绝候选 target；这说明 churn guard 生效，但不说明成功率提升。

原始 compact summary：`results/cage_v02_contract_commit/smoke/compact_summary.md`
