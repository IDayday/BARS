# BARS Experiment Package

This repository implements a modular experimental scaffold for BARS on D4RL antmaze-style tasks.

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
cd bars_experiment_package
python -m pip install -e .
```

Your D4RL environment should already be available. OGBench is not required.

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
- `planner.variant`: `shortest`, `reachability`, `full_bars`.
- `diagnostics.planner_variants`: variants evaluated in path diagnostics.
- `boundary.enabled`: turn boundary compatibility on or off.
- `eval.enabled`: run online environment rollouts.

## Notes

The first version uses GCBC as the low-level policy. It is intentionally modular: replace `bars/models/policy.py` and `bars/training/policy_train.py` with HIQL or another learned low-level policy without changing the graph/planning/scheduler modules.
