# Phase 3D Sampling Ablation Summary

Phase 3D is an offline supervised ablation. It compares how GCBC training
sampling changes action-prediction MSE on Phase 2 data-supported option-edge
segments.

It does not run environment rollout and does not measure closed-loop edge
execution success. Phase 3C closed-loop execution remains gated on a reliable
reset-to-state probe. With the current `env_unavailable` reset probe status,
rollout metrics are intentionally skipped.

The ablation compares:

- `uniform_transition`: transition-level supervised sampling.
- `uniform_edge`: edge-balanced sampling.
- `bottleneck_weighted`: edge-balanced sampling weighted by Phase 2 edge
  bottleneck score.

Primary offline readouts:

- final and best validation action MSE.
- bottleneck vs non-bottleneck edge validation MSE.
- high-support vs low-support edge validation MSE.
- short-horizon vs long-horizon edge validation MSE.

Default configs are lightweight so they can be rerun quickly in the current
environment. Increase `num_steps` and `seeds` for stronger statistical
conclusions.

## Current Lightweight Run

Commands run:

- `python scripts/run_phase3_sampling_ablation.py --config configs/phase3_sampling_ablation_antmaze.yaml`
- `python scripts/run_phase3_sampling_ablation.py --config configs/phase3_sampling_ablation_scene.yaml`

The checked-in configs use `num_steps: 200` and `seeds: [0]`, so these numbers
are smoke-scale offline fitting diagnostics.

AntMaze outputs:

- `results/phase3_sampling/antmaze_large_stitch/all_per_seed_metrics.csv`
- `results/phase3_sampling/antmaze_large_stitch/all_sampling_ablation_summary.csv`

Scene outputs:

- `results/phase3_sampling/scene_play/all_per_seed_metrics.csv`
- `results/phase3_sampling/scene_play/all_sampling_ablation_summary.csv`

At this smoke scale, `uniform_transition` has the best final validation action
MSE on the AntMaze `core_plus_bottleneck_budget120_H10` run and on both Scene
runs. For the AntMaze `density_budget120_H10` run, `uniform_transition` also has
the best final validation action MSE, while `bottleneck_weighted` slightly
improves the bottleneck/low-support/long-horizon grouped MSEs relative to
`uniform_edge`.

These are offline supervised action-fitting results only. They should not be
read as edge executability or rollout success until Phase 3C can run with a
reliable reset-to-state environment.
