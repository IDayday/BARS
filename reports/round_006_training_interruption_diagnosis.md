# Round 006 GAS Training Interruption Diagnosis

Generated: 2026-05-24 Asia/Shanghai.

## Scope

This diagnosis covers the Round006 GAS baseline training interruption only. It does not interpret BARS mechanisms or GAS algorithm performance. Evidence class remains pending completed full-budget evaluation rows.

## Observed Failure

- Affected jobs: 12 initial TDR jobs.
- Affected env/seed set:
  - `antmaze-giant-navigate-v0`, seeds 42-46.
  - `antmaze-giant-stitch-v0`, seeds 42-46.
  - `antmaze-large-explore-v0`, seeds 42-43.
- Failure window: 2026-05-24 01:40:28 to 01:40:47 Asia/Shanghai.
- Phase: `pretrain_tdr`.
- Progress before failure: roughly 125k-165k TDR steps.
- Saved recovery artifact: each affected job has at least `params_100000.pkl`.

## Root Cause

The training process exited because the GAS WandB compatibility shim routed disabled WandB logging to TensorBoard by default. TensorBoard then failed while writing its event file:

```text
FileNotFoundError: ... runs_stage22_tensorboard/.../events.out.tfevents...
```

The exception originated from TensorBoard's async event writer, propagated through `SummaryWriter.flush()`, then surfaced in:

```text
external_src/GAS/O_utils/log_utils.py
```

This made `pretrain_tdr.py` exit non-zero, and the worker recorded a `CalledProcessError`.

This is a logging-path failure, not an algorithm/training/evaluation failure. The core GAS training loop had already saved checkpoints and was not failing due to OOM, dataset corruption, config mismatch, or model code.

## Why It Happened

Round006 set:

```text
WANDB_MODE=disabled
WANDB_DISABLED=true
```

But the local GAS compatibility layer treated disabled WandB as "write TensorBoard instead" and used the relative default:

```text
runs_stage22_tensorboard/...
```

Because GAS subprocesses run with cwd `external_src/GAS`, this resolved under:

```text
external_src/GAS/runs_stage22_tensorboard/...
```

That path is a raw Stage22-style logging directory, not a durable Round006 artifact path. The failing event directories were absent after the interruption, while newly launched jobs recreated the same root later. The most likely trigger is that this relative raw log tree was removed or replaced during concurrent cleanup/pruning, leaving active TensorBoard writers with missing event paths.

## Preventive Fix

Implemented code changes:

- `external_src/GAS/O_utils/log_utils.py`
  - `WANDB_DISABLED=true` now uses a no-op WandB-compatible logger by default, keeping CSV/stdout only.
  - TensorBoard is used only when `BARS_USE_TENSORBOARD=1`.
  - TensorBoard log dirs are normalized to absolute paths.
  - TensorBoard write/flush failures are caught and downgrade to CSV/stdout only instead of killing training.
- `scripts/round006_gas_dynamic_orchestrator.py`
  - Round006 worker environment now forces `BARS_USE_TENSORBOARD=0`.
- `scripts/round006_requeue_logging_failures.py`
  - Requeues only failed jobs whose logs contain both `FileNotFoundError` and `events.out.tfevents`.
  - Preserves failure history and resumes from latest checkpoint instead of deleting artifacts.

These changes do not alter GAS hyperparameters, loss, networks, datasets, step budgets, checkpoints, or evaluation protocol.

## Recovery Action

The 12 affected jobs were changed from `failed` to `retry_pending` and retain their `params_100000.pkl` checkpoints. The active orchestrator will resume them when GPU slots become available.

A guard tmux session is running:

```text
round006_logging_requeue
```

It checks every 5 minutes for the same TensorBoard event-file failure signature and requeues only that non-algorithmic failure class, capped by the script's max requeue count.

## Verification

- `python -m py_compile` passed for the patched logging/orchestrator/requeue scripts.
- `WANDB_DISABLED=true BARS_USE_TENSORBOARD=0` now initializes `_NoOpWandbCompat`.
- `WANDB_DISABLED=true BARS_USE_TENSORBOARD=1` still initializes TensorBoard successfully for explicit diagnostic use.
- Latest monitor snapshot after requeue showed:
  - `failed`: 0
  - `retry_pending`: 12
  - `launched`: 12
  - active jobs still running in TDR

