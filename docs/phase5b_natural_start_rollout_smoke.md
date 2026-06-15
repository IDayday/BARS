# Phase 5B Natural-Start Rollout Smoke

Phase 5B turns the Phase 3F placeholder into a reset-free online smoke test.
It does not use arbitrary reset-to-state and does not evaluate edge-level
execution from sampled offline states.

## References Checked

- OGBench GitHub documents `ogbench.make_env_and_datasets(...)` as the primary
  API that creates an environment and loads train/validation datasets:
  https://github.com/seohongpark/ogbench
- The OGBench project page describes the benchmark as offline
  goal-conditioned RL across locomotion and manipulation environments:
  https://seohong.me/projects/ogbench/
- Local OGBench source under `external_src/tmd-release/ogbench` confirms that
  AntMaze and Scene `reset(...)` return a Gymnasium-style `(obs, info)` pair
  where `info["goal"]` contains the current task goal observation.

## What Was Implemented

- `phase3f/natural_rollout.py`
  - loads Phase 3/4 GCBC checkpoints;
  - runs Gymnasium-style natural-start episodes;
  - uses `info["goal"]` from `env.reset(...)` as the policy goal;
  - clips actions to `env.action_space`;
  - records per-episode summaries, failure reasons, and JSONL traces.
- `scripts/run_phase3f_natural_rollout.py`
  - supports YAML configs and CLI overrides;
  - gates on Phase 3 preflight status;
  - skips safely when the environment is unavailable;
  - does not require reset-to-state.
- Configs:
  - `configs/phase5b_natural_rollout_antmaze_direct.yaml`
  - `configs/phase5b_natural_rollout_scene_direct.yaml`

## Commands

```bash
MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5b_natural_rollout_antmaze_direct.yaml --device cpu

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5b_natural_rollout_scene_direct.yaml --device cpu
```

## Smoke Results

| dataset | policy | episodes | steps/episode | success rate | mean final goal L2 | failure reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `antmaze-large-stitch-v0` | Phase 3A corebot H10 B120 GCBC | 2 | 100 | 0.0 | 29.7779 | `max_steps_without_success` |
| `scene-play-v0` | Phase 4O Scene H25 `s04` GCBC | 2 | 100 | 0.0 | 6.1759 | `max_steps_without_success` |

Outputs:

- `results/phase3f/antmaze_large_stitch/direct_gcbc_corebot_H10_B120/`
- `results/phase3f/scene_play/direct_gcbc_phase4o_scene_H25_s04/`

## Interpretation

This is useful because it proves the local `gcrlo` stack can run closed-loop
OGBench episodes from natural starts with the trained GCBC checkpoints. The
previous blocker was dependency/environment construction; the remaining blocker
for edge-level execution is arbitrary reset-to-state semantics.

The zero success rate is not yet an algorithm comparison. These policies were
trained on offline option-edge BC samples, not on full natural-start task
episodes with hierarchical switching. Directly feeding the environment task
goal to the edge GCBC is a deliberately weak smoke baseline.

Next online work should add:

1. cluster assignment for the current natural-start observation and task goal;
2. support/repaired graph planning from current cluster to goal cluster;
3. subgoal or termination-state selection for each option edge;
4. option switching when the destination cluster is reached;
5. comparison against direct GCBC, support shortest path, compatibility-aware
   path, and certified path.
