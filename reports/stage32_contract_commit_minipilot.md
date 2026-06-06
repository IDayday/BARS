# Stage32 CAGE-v0.2 Minipilot 报告

## 范围

- 环境/seed：
  - `antmaze-giant-navigate-v0:42`
  - `antmaze-giant-stitch-v0:42`
  - `humanoidmaze-large-navigate-v0:44`
- variants：`gas`, `cage_trace_only`, `cage_fixed_commit`, `cage_safe_full`, `cage_contract_commit`
- 预算：`episodes_per_goal=5`, `goals_per_env=5`
- 合同模型：`results/cage_v02_contract/models/contract_model.json`

## 命令

```bash
/root/miniconda3/envs/gcrlo/bin/python scripts/run_contract_capture_smoke.py \
  --checkpoint_root /mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138 \
  --output_root results/cage_v02_contract_commit/minipilot \
  --env_seed_pairs antmaze-giant-navigate-v0:42 antmaze-giant-stitch-v0:42 humanoidmaze-large-navigate-v0:44 \
  --variants gas cage_trace_only cage_fixed_commit cage_safe_full cage_contract_commit \
  --episodes_per_goal 5 \
  --goals_per_env 5 \
  --status_path results/cage_v02_contract_commit/minipilot/status.jsonl \
  --cage_debug \
  --cage_contract_model_path /mnt/project/BARS/results/cage_v02_contract/models/contract_model.json
```

## 结果

| env | variant | status | success | switches | stall | drift | replans | churn | fallback_steps | seg_reach | contract_loaded | gate_reject_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| antmaze-giant-navigate-v0 | gas | succeeded | 0.640 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| antmaze-giant-navigate-v0 | cage_trace_only | succeeded | 0.640 | 113.160 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.053 | 0.000 | 0.000 |
| antmaze-giant-navigate-v0 | cage_fixed_commit | succeeded | 0.000 | 68.240 | 37.800 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| antmaze-giant-navigate-v0 | cage_safe_full | succeeded | 0.680 | 80.120 | 10.800 | 0.000 | 5.560 | 0.000 | 0.000 | 0.012 | 0.000 | 0.000 |
| antmaze-giant-navigate-v0 | cage_contract_commit | succeeded | 0.360 | 39.760 | 19.040 | 0.000 | 0.000 | 0.000 | 9.200 | 0.197 | 1.000 | 0.517 |
| antmaze-giant-stitch-v0 | gas | succeeded | 0.800 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| antmaze-giant-stitch-v0 | cage_trace_only | succeeded | 0.800 | 135.880 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.044 | 0.000 | 0.000 |
| antmaze-giant-stitch-v0 | cage_fixed_commit | succeeded | 0.120 | 63.120 | 30.880 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 |
| antmaze-giant-stitch-v0 | cage_safe_full | succeeded | 0.800 | 72.200 | 8.720 | 0.000 | 5.720 | 0.000 | 0.000 | 0.009 | 0.000 | 0.000 |
| antmaze-giant-stitch-v0 | cage_contract_commit | succeeded | 0.000 | 12.760 | 51.440 | 0.000 | 0.000 | 0.000 | 14.840 | 0.848 | 1.000 | 0.998 |
| humanoidmaze-large-navigate-v0 | gas | failed | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| humanoidmaze-large-navigate-v0 | cage_trace_only | missing | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| humanoidmaze-large-navigate-v0 | cage_fixed_commit | missing | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| humanoidmaze-large-navigate-v0 | cage_safe_full | missing | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| humanoidmaze-large-navigate-v0 | cage_contract_commit | missing | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## BLOCKED

Humanoid GAS job 在写 exact StateRef contract trace 时触发：

```text
OSError: [Errno 122] Disk quota exceeded
```

随后脚本在创建下一个 humanoid job 目录时再次触发 disk quota，导致 status 文件未能正常写出。已删除本轮 minipilot 根目录下的大型 `*_segments.jsonl` raw trace 和 CAGE debug raw trace，并保留 compact summary、`eval.csv`、stdout/stderr。

## 结论

- Trace-only parity：AntMaze 两个环境上成功率与 GAS 一致，未见 instrumentation 破坏。
- Churn reduction：`cage_contract_commit` 在 AntMaze 中将 global replan 降到 0，优于 `cage_safe_full` 的约 5.6 次。
- Success safety：FAIL。`cage_contract_commit` 在 antmaze-nav 从 GAS 0.64 降到 0.36，在 antmaze-stitch 从 GAS 0.80 降到 0.00。
- Contract validity：INCONCLUSIVE/FAIL for deployment。合同模型能打分并拒绝 target，但当前 gate 明显过保守，尤其 stitch gate reject rate 0.998。
- 当前不支持进入大规模 benchmark。

compact summary：`results/cage_v02_contract_commit/minipilot/compact_summary.md`
