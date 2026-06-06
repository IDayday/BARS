# Stage33 CAGE-v0.3 Contract-Rank Smoke 报告

## 范围

- 环境：`antmaze-giant-navigate-v0`
- seed：42
- variants：`gas`, `cage_trace_only`, `cage_safe_full`, `cage_contract_commit`, `cage_contract_rank`
- 预算：`episodes_per_goal=1`, `goals_per_env=1`
- Python：`/root/miniconda3/envs/gcrlo/bin/python`
- 合同模型：`results/cage_v02_contract/models/contract_model.json`
- trace：light debug，禁用 exact StateRef 和完整 phi vector

## 命令

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/cage_experiment_manifest.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_v03_contract_rank/smoke \
  --envs antmaze-giant-navigate-v0 \
  --seeds 42 \
  --variants gas cage_trace_only cage_safe_full cage_contract_commit cage_contract_rank \
  --episodes_per_goal 1 \
  --goals_per_env 1 \
  --manifest_path results/cage_v03_contract_rank/smoke/manifests/smoke_manifest.jsonl \
  --strict_paths \
  --cage_contract_model_path /mnt/project/BARS/results/cage_v02_contract/models/contract_model.json \
  --cage_debug_light \
  --cage_max_debug_steps_per_episode 200 \
  --no-cage_trace_phi_vectors

/root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py \
  --manifest_path results/cage_v03_contract_rank/smoke/manifests/smoke_manifest.jsonl \
  --max_jobs 5
```

## 结果

| variant | status | success | replans | stall | seg_reach | coverage | choose_gas | choose_cage | choose_committed | gate_reject_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gas | succeeded | 0.000 | NA | NA | NA | NA | NA | NA | NA | NA |
| cage_trace_only | succeeded | 0.000 | 0.000 | 0.000 | 0.078 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| cage_safe_full | succeeded | 0.000 | 5.000 | 12.000 | 0.011 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| cage_contract_commit | succeeded | 0.000 | 0.000 | 42.000 | 0.457 | NA | 0.000 | 0.000 | 0.000 | 0.897 |
| cage_contract_rank | succeeded | 0.000 | 0.000 | 27.000 | 0.472 | 1.000 | 133.000 | 66.000 | 270.000 | 0.000 |

## Smoke 结论

- 5 个 job 全部返回 0。
- GAS 命令未包含 `--use_cage`。
- `cage_contract_rank` 命令包含 `--use_cage --cage_contract_rank`。
- `cage_contract_rank` 不再复现 v0.2 hard gate 的高拒绝率：coverage 为 1.000，gate reject rate 为 0。
- smoke 成功率全为 0，不能做性能判断；但 smoke 通过了进入 AntMaze 5x5 minipilot 的工程 gate。

原始聚合：`results/cage_v03_contract_rank/smoke/tables/smoke_summary.md`
