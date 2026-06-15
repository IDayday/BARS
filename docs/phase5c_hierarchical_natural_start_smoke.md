# Phase 5C Hierarchical Natural-Start Rollout Smoke

Phase 5C adds the first reset-free hierarchical online executor. It still does
not use arbitrary reset-to-state and does not claim edge execution success from
offline proxies.

## Implemented

- Reconstructs the Phase 2 cluster model at runtime from the same offline
  dataset slice.
- Plans over support-certified graph edges only.
- Adds optional full support-bank fallback. This is diagnostic and still uses
  only real offline support edges; it does not create kNN/proximity shortcuts.
- Selects each option subgoal from real edge segments.
- Uses initiation-aware subgoal selection:
  - prefer segments whose initiation observation is close to the current online
    observation;
  - also prefer terminations that connect to the next edge initiation or final
    task goal.
- Switches options when the online observation enters the destination cluster.
- Records path diagnostics: `start_cluster`, `goal_cluster`, `planned_edges`,
  `completed_edges`, `replans`, `full_bank_fallbacks`, `subgoal_l2`, and
  `action_norm`.

## Commands

```bash
MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5c_hierarchical_rollout_antmaze_corebot100k.yaml \
  --device cpu

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5c_hierarchical_rollout_antmaze_repaired.yaml \
  --device cpu

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5c_hierarchical_rollout_scene_repaired.yaml \
  --num_episodes 1 --max_steps 40 --device cpu
```

## Results

| dataset | method | episodes | max steps | success | mean completed edges | mean replans | full-bank fallback | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AntMaze | `hierarchical_repaired_corebot100k_H10_B120` | 2 | 200 | 0.0 | 1.0 | 9.0 | 9.0 | `max_replans_exceeded` |
| AntMaze | `hierarchical_repaired_phase4m_s04_H10_B120` | 2 | 200 | 0.0 | 2.5 | 9.0 | 9.0 | `max_replans_exceeded` |
| Scene | `hierarchical_repaired_phase4o_s04_H25_B192` | 1 | 40 | 0.0 | 0.0 | 2.0 | 0.0 | `max_steps_without_success` |

Outputs:

- `results/phase3f/antmaze_large_stitch/hierarchical_repaired_corebot100k_H10_B120/`
- `results/phase3f/antmaze_large_stitch/hierarchical_repaired_phase4m_s04_H10_B120/`
- `results/phase3f/scene_play/hierarchical_repaired_phase4o_s04_H25_B192/`

## Interpretation

The hierarchical executor is a meaningful improvement over the Phase 5B direct
GCBC smoke because it plans, chooses support segment subgoals, and switches
edges online. It still does not solve the tasks.

The useful finding is the failure mode:

- AntMaze needs full support-bank fallback on every replan. The compressed
  repaired graph alone does not cover these natural-start task endpoints well
  enough. Even with fallback, the online path is very long and edge completion
  is unstable.
- Scene can plan a short support path without fallback, but the first option is
  not reliably completed. This points to subgoal/control mismatch more than
  graph coverage.
- Initiation-aware subgoal selection improved AntMaze path progress compared
  with the first hierarchical attempt: the executor now completes some support
  edges and records where it stalls.

## Next Algorithmic Work

The next mature algorithmic step should not be another graph-only coverage
change. The online bottleneck is now policy-aware execution:

1. Cache runtime cluster models so repeated Scene evaluation does not refit
   kmeans.
2. Score candidate edge segments by current-state initiation distance, action
   MSE/policy support, and compatibility, not just graph cost.
3. Add risk-aware replanning that penalizes edges repeatedly failed online.
4. Train longer or train specifically on planner-used edge distributions.
5. Evaluate direct GCBC, support shortest path, repaired path, and certified
   path under the same natural-start protocol.

Phase 5D implements the first version of item 2 and adds failure-penalized
replanning plus cluster-model caching. It improves AntMaze partial progress
slightly but still does not solve natural-start tasks.
