# Stage 26 Pause And Resume Notes

Generated: 2026-05-28
Branch: `stage26-tdr-factor-tmdcost`
Container project path: `/mnt/project/BARS`

## Pause State

- Active Stage 26/TMD processes: none at pause time.
- GPUs 0-7 were idle at pause time.
- Latest analysis was regenerated with `python scripts/stage26_analyze.py`.
- Tracked reports are under `reports/stage26_tmd_tdr_*`.
- Large run outputs are intentionally local-only and ignored by git:
  - `runs_stage26_tmd_tdr/` about 607M
  - `artifacts/stage26_lowcond/` about 13M
  - `logs/stage26_lowcond_train/` and `logs/stage26_lowcond_eval/`

## Completed Work

Phase A:
- Stage 26 analysis pipeline and reports are in place.
- `stage26_analyze.py` aggregates eval CSVs, graph stats, Phase B comparisons, Phase D lowcond comparisons, task-wise deltas, failure counts, and decision notes.

Phase B:
- TMD soft-cost blend matrix was completed for giant-navigate, giant-stitch, and medium sanity checks.
- Confirmed signal: `antmaze-giant-navigate-v0`, `w=0.25`, 50/task confirm, delta success about +3.7pp with positive normal and stratified bootstrap CI.
- Not confirmed: giant-stitch and medium sanity environments. Medium baselines are near-saturated.

Phase C:
- Universal low-level condition code is implemented under `bars/conditioning/`.
- Unit test: `pytest -q tests/test_low_level_condition.py` passed.

Phase D smoke/ablation:
- Completed 5/task smoke ablations for:
  - `antmaze-medium-stitch-v0`
  - `antmaze-medium-navigate-v0`
  - `antmaze-giant-navigate-v0`
- Results show current BC lowcond actors do not promote. Factor-only fails badly. TDR-only is closest on medium-stitch but still slower than GAS; giant-navigate lowcond variants collapse relative to GAS.

## Latest Phase D Smoke Snapshot

`antmaze-medium-stitch-v0`, seed0/gas_seed42, 5/task:
- GAS: 25/25, mean steps 234.8
- lowcond TDR-only local: 24/25, mean steps 393.0
- lowcond full nearest-task-goal: 22/25, mean steps 424.0
- lowcond factor-only nearest-task-goal: 2/25, mean steps 975.8

`antmaze-medium-navigate-v0`, seed0/gas_seed42, 5/task:
- GAS: 25/25, mean steps 228.7
- lowcond full-localres: 22/25, mean steps 479.2
- lowcond TDR-only local: 19/25, mean steps 511.4
- lowcond factor-only nearest-task-goal: 0/25, mean steps 1000.0

`antmaze-giant-navigate-v0`, seed0/gas_seed42, 5/task:
- GAS: 13/25, mean steps 861.3
- lowcond TDR-only local: 5/25, mean steps 959.4
- lowcond full nearest-task-goal: 2/25, mean steps 977.6
- lowcond factor-only nearest-task-goal: 0/25, mean steps 1000.0

## Quick Resume Commands

From the local machine:

```bash
ssh -A -o PermitLocalCommand=no training-rl-zt
```

Inside the container:

```bash
cd /mnt/project/BARS
git checkout stage26-tdr-factor-tmdcost
git pull --ff-only

source /root/miniconda3/bin/activate gcrlo
export OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench
export BARS_TMD_TEST_DATASET_ROOT=/mnt/project/offlinerl_datasets/ogbench
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin"

pgrep -af "stage26|evaluate_tmd_graph|tmd_test_eval|train_lowcond" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
pytest -q tests/test_low_level_condition.py
python scripts/stage26_analyze.py
```

## Recommended Next Step

Continue Phase D on `antmaze-giant-stitch-v0` only as a smoke/diagnostic, not as a promote attempt:

1. Fit `artifacts/stage26_lowcond/antmaze-giant-stitch-v0/gasseed42/stats_50k.npz`.
2. Train the same lowcond ablations used for the other three envs.
3. Run 5/task eval with fixed GAS graph.
4. Regenerate `reports/stage26_tmd_tdr_*`.

Do not start Phase E combination until lowcond execution is fixed. Current evidence says the route-level TMD soft cost is useful for giant-navigate, while the current lowcond BC actors are not yet reliable.
