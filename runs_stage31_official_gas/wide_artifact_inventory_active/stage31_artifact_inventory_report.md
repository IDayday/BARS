# Stage31 Official GAS Artifact Inventory

Status: OFFICIAL_GAS_ARTIFACT_INVENTORY.
Pre-Stage30 BARS/Stage28/Stage29 evidence: ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE.
READY rows can be evaluated directly with official GAS graph/planner/policy/action outputs unchanged.
Missing checkpoint rows are queued as training candidates, not interpreted as diagnosis evidence.

## Source Lock

- official repo SHA: `UNAVAILABLE_IN_VENDOR_COPY`
- GAS repo path: `external_src/GAS`
- command: `/root/miniconda3/envs/navsim/bin/python scripts/stage31_official_gas_artifact_inventory.py --artifact-root artifacts/gas_ogbench_offline_full_20260522_165138 --dataset-root /mnt/project/offlinerl_datasets --gas-repo-path external_src/GAS --seeds 44,45,46 --out-root runs_stage31_official_gas/wide_artifact_inventory_active`

## Counts

- artifact rows: 102
- ready official GAS rows: 24
- missing checkpoint training pending rows: 75
- local D4RL datasets: 25

| tier | rows | ready | pending_training |
| --- | --- | --- | --- |
| Extra | 60 | 0 | 60 |
| Tier1 | 15 | 9 | 3 |
| Tier2 | 15 | 15 | 0 |
| Tier3 | 12 | 0 | 12 |

## Ready Envs

antmaze-giant-navigate-v0, antmaze-giant-stitch-v0, antmaze-large-explore-v0, antmaze-large-navigate-v0, antmaze-large-stitch-v0, antmaze-medium-explore-v0, antmaze-medium-navigate-v0, antmaze-medium-stitch-v0

## Training Pending Envs

antmaze-teleport-explore-v0, antmaze-teleport-navigate-v0, antmaze-teleport-stitch-v0, humanoidmaze-large-navigate-v0, humanoidmaze-large-stitch-v0, humanoidmaze-medium-navigate-v0, humanoidmaze-medium-stitch-v0, pointmaze-giant-navigate-v0, pointmaze-giant-stitch-v0, pointmaze-large-navigate-v0, pointmaze-large-stitch-v0, pointmaze-medium-navigate-v0, pointmaze-medium-stitch-v0, pointmaze-teleport-navigate-v0, pointmaze-teleport-stitch-v0, scene-play-v0, visual-antmaze-giant-navigate-v0, visual-antmaze-giant-stitch-v0, visual-antmaze-large-explore-v0, visual-antmaze-large-navigate-v0, visual-antmaze-large-stitch-v0, visual-antmaze-medium-explore-v0, visual-antmaze-medium-navigate-v0, visual-antmaze-medium-stitch-v0, visual-scene-play-v0

## Files

- artifact inventory: `runs_stage31_official_gas/wide_artifact_inventory_active/official_gas_artifact_inventory.csv`
- D4RL inventory: `runs_stage31_official_gas/wide_artifact_inventory_active/d4rl_dataset_inventory.csv`
- ready matrix: `runs_stage31_official_gas/wide_artifact_inventory_active/official_gas_ready_matrix.csv`
- missing checkpoint queue: `runs_stage31_official_gas/wide_artifact_inventory_active/official_gas_missing_ckpt_training_queue.csv`
