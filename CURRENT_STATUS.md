# BARS Current Status

Updated: 2026-05-19 15:05 Asia/Shanghai

## New Server Bootstrap

```bash
git clone git@github.com:IDayday/BARS.git
cd BARS
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e '.[stage22-gas]'
bash scripts/setup_gas_repo.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Notes:

- `artifacts/`, `runs_stage22*/`, and `runs_stage23*/` are intentionally not tracked.
- `external_src/GAS/` is a pruned vendored GAS source tree; use `scripts/setup_gas_repo.sh` to verify or reapply `third_party/gas_stage22.patch`.
- Existing reports under `reports/` are the lightweight experiment record to start from.
- On a fresh GPU server, install the JAX/JAXLIB build that matches CUDA before long Stage22/23 jobs.

## Code Scope In This Handoff

- `bars/external/`: GAS artifact resolution, official checkpoint download, GAS policy/keygraph loading, and dataset embedding export.
- `bars/gas_bars/`: GAS-aligned BARS planners, reachability/boundary scoring, bridge graph diagnostics, failure atlas, edge execution, oracle bridge, fallback causal analysis, and integrated evaluators.
- `scripts/stage22*`, `scripts/stage22r*`, `scripts/stage23*`: reproducible launch, monitor, analyze, and repair entrypoints.
- `configs/stage22/` and `configs/stage23_*.json`: current experiment matrices and repair protocols.
- `reports/`: current markdown/csv summaries. Raw logs/checkpoints are excluded from git.

## Current Experiment State

No tmux session or Stage23 evaluation process is currently running on this machine. The latest calibrated Stage23 key-claim run completed 12/12 jobs with 0 failures and 1200 eval episodes.

Latest key-claim live summary:

```text
antmaze-medium-navigate-v0 seed0:
  gas_shortest none:                         0.89
  gas_reachability_budget_calibrated none:  0.92
  gas_reachability_soft_calibrated none:    0.90
  gas_shortest progress_stall_v3:            0.65
  gas_reachability_budget_calibrated v3:     0.69
  gas_reachability_soft_calibrated v3:       0.63

antmaze-medium-stitch-v0 seed0:
  gas_shortest none:                         0.86
  gas_reachability_budget_calibrated none:  0.87
  gas_reachability_soft_calibrated none:    0.91
  gas_shortest progress_stall_v3:            0.69
  gas_reachability_budget_calibrated v3:     0.77
  gas_reachability_soft_calibrated v3:       0.73
```

Interpretation:

- Same-fallback `none` comparison is the cleanest planner evidence. Reachability is positive on both medium envs in seed0, but the effect is still small enough that it needs seed/env expansion before a final BARS win claim.
- `progress_stall_v3` improves relative to `gas_shortest` in some cells, but absolute success is below the `none` protocol. Treat fallback v3 as a repair target, not as main planner evidence.
- Boundary remains HOLD because Stage22R budget reject rate was about 0.985 and boundary feasibility/risk scale is not repaired.

## Decisions

- `GO_REACHABILITY_SEED_EXPANSION`: calibrated reachability has a positive medium seed0 signal under `fallback=none`.
- `HOLD_FINAL_REACHABILITY_CLAIM`: need seeds 1/2 and larger envs with calibrated budgets.
- `REPAIR_FALLBACK_V3`: do not include fallback gains in the planner claim until causal trigger-state ablation passes.
- `HOLD_BOUNDARY`: boundary stays diagnostic-only until budget rejects and virtual start/goal boundary handling are fixed.
- `GO_D4RL_PROTOCOL_REPAIR`: official-control adapter route is within 1.4pp of official GAS route B in the current repair report.
- `HOLD_INTEGRATED_BARS_V3`: integrated no-fallback BARS-v3 rows are still pending and oracle headroom was not demonstrated on the tested bridge/oracle setup.

## Resume Commands

Refresh the completed key-claim summary:

```bash
python scripts/stage23_monitor_and_adjust.py \
  --roots runs_stage23_key_claim_logs,runs_stage23_key_claim \
  --summary-md reports/stage23_live_summary.md
```

Run the next calibrated reachability expansion after producing budgets for the target env/seed pairs:

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

Run protocol/bridge diagnostics:

```bash
bash scripts/stage23_pipeline.sh MODE=repro ENVS=antmaze-medium-navigate-v0,antmaze-medium-stitch-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1
bash scripts/stage23_pipeline.sh MODE=bridge ENVS=antmaze-large-explore-v0 SEEDS=0 GPUS=${GPUS:-0} WAIT=1
bash scripts/stage23_pipeline.sh MODE=edge_exec ENVS=antmaze-large-explore-v0 SEEDS=0 EDGE_EXEC_PILOT=1 GPUS=${GPUS:-0} WAIT=1
```

## Key Reports

- `reports/stage23_live_summary.md`: latest 12-job key-claim result.
- `reports/stage22r_decisions.md`: Stage22R reachability/boundary decision.
- `reports/stage22_finalized_summary.md`: Stage22 same-backbone pilot.
- `reports/stage23_summary.md`: Stage23 reproduction, failure atlas, bridge/oracle, p_bridge, and boundary summary.
- `reports/stage23_adapter_protocol_repair.md`: official GAS adapter protocol repair.
- `reports/stage23_failure_atlas.md`: current failure mode summary.
