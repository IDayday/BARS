# Stage22 Next Plan

Updated: 2026-05-19 08:45 Asia/Shanghai

## Current Readout

- Pilot completed: 1200 eval episodes, 0 failed jobs.
- GAS baseline is healthy: medium-stitch 0.90/0.92, medium-navigate 0.86/0.84.
- Reachability-budget has local positive signal but not yet a GO:
  - stitch budget3 + progress_stall_v2: 0.94 vs gas_shortest budget3 + progress_stall_v2: 0.88.
  - navigate budget3 + progress_stall_v2: 0.90 vs gas_shortest budget3 + progress_stall_v2: 0.88.
  - aggregate paired delta is only +0.25pp, so confirm is premature.
- progress_stall_v2 is much better than the old fallback. Triggered fallback episodes are high-success, but it can still be neutral/slightly harmful on GAS shortest.
- boundary-budget at budget 2/3 is fully infeasible: 100% budget reject on both medium envs. Treat as HOLD_BOUNDARY until risk calibration is fixed.

## Experiments Launched

Budget calibration sweep is running in tmux session `stage22_budget_sweep`.

Command shape:

```bash
MODE=pilot \
ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 \
SEEDS=0 \
VARIANTS=gas_reachability_budget,gas_boundary_budget \
BUDGETS=5.0,8.0 \
FALLBACK_MODES=none,progress_stall_v2 \
EPISODES=50 \
MAX_PARALLEL_EVAL=4 \
bash scripts/stage22_pipeline.sh
```

Primary questions:

- Does reachability-budget improve consistently at larger budgets, or does it just converge back to GAS shortest?
- Does boundary-budget become feasible at budget 5/8, or does it need risk-scale repair?
- Does progress_stall_v2 improve reachability without hurting a healthy GAS baseline?

## Algorithm Changes To Try Next

1. Reachability calibration:
   - Sweep unsupported edge penalty: 0.05, 0.10, 0.15.
   - Add selected-edge precision diagnostics, not only AUROC/AUPRC.
   - Prefer budget3/5 if budget2 is too restrictive.

2. Boundary repair:
   - Introduce a boundary risk scale, e.g. total risk = exec_risk + alpha_boundary * boundary_cost.
   - Sweep alpha_boundary in 0.25, 0.5, 1.0 with budgets 3/5/8.
   - Keep boundary diagnostic-only until it beats reachability-budget.

3. Fallback repair:
   - Keep progress_stall_v2.
   - Add a stricter v3 gate for GAS-shortest: fallback only if best_goal_dist improved and current/target progress has stalled for K>=5 replans.
   - Report fallback_used success separately from no-fallback episodes.

4. Efficiency:
   - Cache virtual start/goal nearest-node connections per rounded phi/task.
   - Precompute static task-goal paths for GAS shortest and reachability-budget.
   - Keep eval parallelism; default MAX_PARALLEL_EVAL should match available GPUs.

## Decision Gate

- GO_SAME_BACKBONE only if at least two medium env/fallback settings show >=5pp over gas_shortest with the same backbone.
- HOLD_BOUNDARY remains active unless budget5/8 or scaled-boundary beats reachability-budget.
- REPAIR_SCORING if reachability-budget cannot beat gas_shortest after penalty/budget sweep.
