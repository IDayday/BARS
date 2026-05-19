# BARS Experiment Package

This repository implements a modular experimental scaffold for BARS on D4RL antmaze-style tasks.

## Current handoff

This checkout now contains the Stage22/Stage22R and Stage23 GAS-aligned BARS work:

- `bars/external/`: adapters for official GAS artifacts, policy loading, keygraph loading, and same-backbone evaluation.
- `bars/gas_bars/`: reachability scoring, calibrated GAS/BARS planners, boundary diagnostics, bridge graph tools, failure atlas utilities, and integrated Stage23 evaluators.
- `configs/stage22/` and `configs/stage23_*.json`: experiment configs for pilot, repair, calibrated reachability confirm, boundary re-entry, and D4RL protocol repair.
- `scripts/stage22*`, `scripts/stage22r*`, `scripts/stage23*`: launch, monitor, analyze, and diagnostic entrypoints.
- `reports/`: committed markdown/csv summaries. Raw run directories and checkpoints are intentionally ignored.

Read `CURRENT_STATUS.md` first when resuming the experiments on a new server.

## What is included

- D4RL dataset loading with trajectory reconstruction.
- Goal-conditioned behavior cloning low-level policy.
- Temporal-distance representation training.
- Empirical reachability model over latent pair features.
- Random/FPS/kmeans/spectral/BARS node proposal.
- Reachability-scored graph edges.
- Offline boundary compatibility through local portal-mode histograms.
- Shortest, reachability-aware, and boundary-aware line-graph planning.
- CSV-only logs: `train.csv`, `graph.csv`, `diagnostics.csv`, `eval.csv`, `summary.csv`.
- Per-run automatic `archives/logs_<run>.tar.gz` packaging.
- Multi-GPU memory-aware job controller with status and graceful stop commands.

## Installation

```bash
cd BARS
python -m pip install -e .
```

For Stage22/Stage23 GAS-aligned experiments, install the optional dependencies
that match your CUDA/JAX stack, then verify/apply the GAS compatibility patch:

```bash
python -m pip install -e '.[stage22-gas]'
bash scripts/setup_gas_repo.sh
```

Your D4RL/OGBench/MuJoCo environment should already be available on the server.
The pruned GAS source lives in `external_src/GAS`; the reproducible compatibility
patch is tracked at `third_party/gas_stage22.patch`.

## JAX and Torch in the same environment

This project uses `torch` only. If your shared Conda environment also has GPU `jax` installed, JAX may reserve a large fraction of GPU memory on its first device operation and make Torch runs look unstable or incompatible.

For interactive work in a shared environment, prefer:

```bash
source scripts/gcrlo_torch_safe_env.sh 0
```

That helper:

- disables JAX GPU preallocation
- limits visibility to one GPU when you pass a GPU id
- leaves the scheduler behavior unchanged because scheduled jobs already set `CUDA_VISIBLE_DEVICES`

If you do not want to use the helper, the important manual setting is:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

If JAX is not needed for this project, the cleanest option is to remove it from the environment used for BARS runs.

## Smoke test without D4RL

```bash
bash examples/smoke_toy.sh
```

## Single D4RL run

```bash
bash scripts/run_single.sh antmaze-medium-play-v2 0 full_bars bars
```

Equivalent explicit command:

```bash
python -m bars.cli run \
  --config configs/d4rl_antmaze_quick.json \
  --env antmaze-medium-play-v2 \
  --seed 0 \
  --variant full_bars \
  --node-method bars \
  --log-root runs
```

Enable online evaluation after diagnostics:

```bash
python -m bars.cli run \
  --config configs/d4rl_antmaze.json \
  --env antmaze-large-diverse-v2 \
  --seed 0 \
  --variant full_bars \
  --node-method bars \
  --set eval.enabled=true \
  --set eval.episodes=100
```

## Multi-GPU sweep

```bash
GPUS=0,1,2,3 MAX_JOBS_PER_GPU=2 LOG_ROOT=runs bash scripts/launch_stage1.sh
```

The scheduler queries `nvidia-smi` and only launches a job if the target GPU has at least the job's `mem_mb` free. Set the per-job memory requirement in the sweep JSON through `resources.default_mem_mb` or each task's `mem_mb`.

Dry-run scheduling:

```bash
python -m bars.sched.jobctl launch \
  --sweep configs/sweeps/d4rl_stage1.json \
  --log-root runs \
  --gpus 0,1,2,3 \
  --max-jobs-per-gpu 2 \
  --dry-run
```

## Status and stop

```bash
bash scripts/status.sh
```

Stop one run without affecting other runs:

```bash
bash scripts/stop_one.sh <run_id>
```

Stop all running experiment jobs gracefully:

```bash
bash scripts/stop_all.sh
```

Force stop only if graceful stop hangs:

```bash
python -m bars.sched.jobctl stop --log-root runs --run-id <run_id> --force
```

Graceful stop sends SIGTERM to the selected run's process group. The experiment loop catches the signal, writes final CSV rows, and packages the logs before exiting.

## CSV aggregation

```bash
python scripts/collect_csv.py --log-root runs
```

This writes aggregated files to `runs/_analysis/*_all.csv`.

## Important config knobs

- `tdr.steps`, `policy.steps`, `reachability.steps`: training budget.
- `graph.node_method`: `random`, `fps`, `kmeans`, `spectral`, `bars`.
- `planner.variant`: `shortest`, `reachability`, `old_full_bars` for the original Stage21 planner. Stage22/23 GAS-aligned planner variants are `gas_shortest`, `gas_reachability_budget_calibrated`, and `gas_reachability_soft_calibrated`.
- `diagnostics.planner_variants`: variants evaluated in path diagnostics.
- `boundary.enabled`: turn boundary compatibility on or off.
- `eval.enabled`: run online environment rollouts.

## Stage22/Stage23 quick commands

Prepare GAS backbone artifacts and BARS reachability models:

```bash
bash scripts/stage22_prepare_gas_backbone.sh ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 SEEDS=0 GPUS=0 QUICK=1
bash scripts/stage22_train_reachability.sh ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 SEEDS=0 GPUS=0 QUICK=1
```

Run the current Stage23 calibrated reachability key-claim matrix:

```bash
bash scripts/stage23_run_key_claim.sh \
  CONFIG=configs/stage23_key_claim_reachability.json \
  ENVS=antmaze-medium-stitch-v0,antmaze-medium-navigate-v0 \
  SEEDS=0 EPISODES=100 GPUS=0 WAIT=1
```

Monitor and refresh the live report:

```bash
python scripts/stage23_monitor_and_adjust.py \
  --roots runs_stage23_key_claim_logs,runs_stage23_key_claim \
  --summary-md reports/stage23_live_summary.md
```

## Notes

The first version uses GCBC as the low-level policy. It is intentionally modular: replace `bars/models/policy.py` and `bars/training/policy_train.py` with HIQL or another learned low-level policy without changing the graph/planning/scheduler modules.
