# BARS Stage22R / Stage23 当前实验快照

快照时间：2026-05-19 15:05 Asia/Shanghai

## 当前状态

Stage23 calibrated reachability key-claim medium seed0 复验已经跑完：

- jobs: 12 completed, 0 failed
- eval rows: 1200
- 当前机器无 tmux session，也没有 Stage23 评估进程在跑
- 最新 live summary: `reports/stage23_live_summary.md`

## 已确认

- Stage22 主 pilot 完成，无 failed jobs。
- GAS shortest 在 medium OGBench antmaze 上健康，说明 Stage21 的失败主要来自 full-stack 自研实现退化，而不是 GAS backbone 或 same-backbone 路线不可行。
- Stage22R 决策为 `GO_REACHABILITY_CONFIRM` 和 `HOLD_BOUNDARY`。
- reachability 与 shortest 的 selected-edge overlap 均值约 `0.483`，说明 reachability planner 的确改变了路径选择。
- Stage23 protocol repair 后，official-control adapter 与 official GAS route B 的 gap 在 1.4pp 内。
- Stage23 key-claim 在 medium seed0 的 `fallback=none` 下给出正信号。

## 最新 Key-Claim 结果

```text
antmaze-medium-navigate-v0 / seed0:
  gas_shortest + none:                    0.89
  gas_reachability_budget_calibrated none:0.92
  gas_reachability_soft_calibrated none:  0.90
  gas_shortest + progress_stall_v3:       0.65
  gas_reachability_budget_calibrated v3:  0.69
  gas_reachability_soft_calibrated v3:    0.63

antmaze-medium-stitch-v0 / seed0:
  gas_shortest + none:                    0.86
  gas_reachability_budget_calibrated none:0.87
  gas_reachability_soft_calibrated none:  0.91
  gas_shortest + progress_stall_v3:       0.69
  gas_reachability_budget_calibrated v3:  0.77
  gas_reachability_soft_calibrated v3:    0.73
```

Interpretation:

- `fallback=none` 是最干净的 planner 对比。medium seed0 上 reachability 对 shortest 为正，但幅度仍然不足以宣布最终 claim。
- `progress_stall_v3` 在相对 shortest 的个别格子有增益，但绝对 success 明显低于 `fallback=none`。v3 应作为 fallback protocol repair target，不能作为 planner 主证据。
- Boundary 继续 HOLD。Stage22R boundary budget reject rate 约 `0.985`，当前问题是 feasibility / risk-scale，不应进入主线 claim。

## 当前决策

- `GO_REACHABILITY_SEED_EXPANSION`
- `HOLD_FINAL_REACHABILITY_CLAIM`
- `REPAIR_FALLBACK_V3`
- `HOLD_BOUNDARY`
- `GO_D4RL_PROTOCOL_REPAIR`
- `HOLD_INTEGRATED_BARS_V3`

## 下一步

1. 为 seeds 1/2 和 hard/large env 先生成 GAS shortest baseline 与 risk calibration。
2. 只在有 recommended budget 的 env/seed 上启动 calibrated reachability confirm。
3. 继续使用 same-fallback paired comparison，优先看 `fallback=none`。
4. 对 fallback v3 做 trigger-state causal ablation，不通过前不要把 fallback 收益算入 planner claim。
5. Boundary 需要先修 budget reject、risk scale、virtual start/goal boundary-pair handling，再进入 re-entry。

## Commands

```bash
python scripts/stage23_monitor_and_adjust.py \
  --roots runs_stage23_key_claim_logs,runs_stage23_key_claim \
  --summary-md reports/stage23_live_summary.md
```

```bash
bash scripts/stage23_run_key_claim.sh \
  CONFIG=configs/stage23_key_claim_reachability.json \
  ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 \
  SEEDS=1,2 \
  EPISODES=100 \
  GPUS=0,1,2,3 \
  MAX_PARALLEL_EVAL=4 \
  MAX_PLAN_EDGES=20 \
  REQUIRE_RECOMMENDED_BUDGET=1 \
  WAIT=1
```
