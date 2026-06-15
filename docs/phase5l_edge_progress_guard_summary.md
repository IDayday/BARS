# Phase 5L Edge Progress Guard

Phase 5L tests a runtime execution guard for hierarchical support-graph
rollout. The guard monitors distance to the current option subgoal and aborts
an edge early when the policy is no longer making local progress.

No unsupported graph edges are added. No new policy is trained.

## Design

For each active option edge, the executor records the post-step L2 distance to
the selected offline termination subgoal. After a minimum number of steps, it
marks the current edge as failed and replans if either condition holds:

- distance to the subgoal grows more than `edge_progress_growth_tolerance`
  beyond the best distance seen on this edge attempt;
- recent window improvement is below `edge_progress_min_improvement` and the
  current distance is worse than the best seen distance.

The intent is to stop spending the full edge horizon on a drifting low-level
execution attempt.

## Commands

```bash
conda run -n gcrlo python -m py_compile \
  phase3f/hierarchical_rollout.py \
  scripts/run_phase3f_natural_rollout.py

conda run -n gcrlo pytest -q tests/test_phase3_synthetic.py

MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/project/offlinerl_datasets/ogbench \
PYTHONPATH=/mnt/project/BARS/external_src/tmd-release \
conda run -n gcrlo python scripts/run_phase3f_natural_rollout.py \
  --config configs/phase5l_edge_progress_guard_antmaze_corebot100k.yaml \
  --device cpu
```

## Results

AntMaze natural-start, task id 1, seeds 0/1/2, 120-step cap:

| method | success | mean final L2 | mean L2 improvement | mean completed edges | mean replans | mean failed attempts | mean progress aborts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct GCBC to final goal | 0.0 | 43.4040 | n/a | n/a | n/a | n/a | n/a |
| Phase 5I learned outcome, weight 0.5 | 0.0 | 39.7197 | 4.1465 | 2.333 | 7.000 | 6.000 | n/a |
| Phase 5K preplan mismatch, weight 0.5 | 0.0 | 39.8942 | 3.9258 | 1.667 | 6.333 | 5.333 | n/a |
| Phase 5L edge progress guard, weight 0.5 | 0.0 | 42.8189 | 0.7426 | 0.667 | 13.000 | 12.000 | 12.000 |

The guard fired 12 to 13 times per episode. Most aborts were caused by
`subgoal_distance_growth`; the rest were caused by `subgoal_progress_stalled`.
All three episodes ended with `max_replans_exceeded`.

Output:

- `results/phase3f/antmaze_large_stitch/edge_progress_guard_w0p5_3ep_corebot100k_H10_B120/`

## Analysis

This is a negative result. The guard correctly detects edge-level drift, but
early aborting by itself reduces task progress. It turns many poor edge
attempts into many short poor edge attempts, exhausting the replan budget before
the agent accumulates meaningful progress toward the task goal.

The direct-GCBC audit also matters. Directly aiming the same 100k-step GCBC at
the final OGBench goal gives 0 success and mean final L2 `43.4040`, while the
best Phase 5I/5K hierarchical variants reduce final L2 to about `39.7` to
`39.9`. The support graph therefore provides some useful structure, but it is
not yet converted into successful task completion by the low-level policy.

The useful conclusion is narrower:

- edge-local progress is a real online execution signal;
- the current low-level GCBC policy is not robust enough for planner-selected
  option subgoals;
- graph-side risk tuning and faster replanning are insufficient while the
  policy training distribution remains weakly aligned with the planner's
  runtime subgoals.

The next algorithmic change should be policy-grounded rather than graph-only:
train or adapt the low-level policy on the actual subgoal distribution emitted
by the planner, then evaluate natural-start success directly.

## Success-Protocol Check

A minimal OGBench AntMaze probe in the `gcrlo` environment confirms that env
construction and success reporting are available:

- `env.reset(seed=0, options={"task_id": 1})` returns a 29-dimensional
  observation and a 29-dimensional `goal`;
- `env.step(action)` returns `info["success"]`;
- the current natural-start code reads `goal`/`desired_goal` from reset info
  and `success` from step info.

So the current zero success rate should not be dismissed as a missing success
field. It is evidence that the current policy/executor is not reaching the
environment's task-success region in this smoke setting.
