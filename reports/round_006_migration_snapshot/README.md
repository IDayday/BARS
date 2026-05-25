# Round 006 Migration Snapshot

Generated: `2026-05-25T10:01:35+08:00`

This snapshot is intentionally lightweight and git-trackable. It includes completed `eval.csv`
files, selected `flags.json` files, job tables, and reports. It does not include checkpoint
weights, videos, TensorBoard files, or dataset files.

## Current Snapshot

- Total jobs: `120`
- Completed eval rows: `64`
- Remaining jobs to run on the next server: `56`
- Weights included: `false`

## Restore On A New Server

After cloning the repository and preparing datasets/dependencies, restore completion markers:

```bash
python scripts/round006_restore_migration_snapshot.py \
  --snapshot-root reports/round_006_migration_snapshot \
  --out-root artifacts/gas_selftrain_round006 \
  --run-root runs_round006_gas_dynamic
```

Then start Round006 training with the normal launcher:

```bash
ROUND=006 SEEDS=42,43,44,45,46 GPUS=0,1,2,3,4,5 \
ROUND006_GPU_SLOTS_PER_GPU=2 POLL_SECONDS=60 DOWNLOAD_POLL_SECONDS=30 \
bash scripts/round006_launch_gas_dynamic.sh
```

The restored `eval.csv` markers make the orchestrator skip already completed env/seed pairs.
Interrupted jobs are not marked completed and should be rerun from scratch for comparability.
