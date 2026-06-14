# Phase 4H Stronger Scene GCBC Direct Repair Validation Summary

Phase 4H reran Scene direct repair-edge policy evidence with a 10000-step GCBC
model instead of the previous 200-step smoke model. This remains reset-free
offline supervised evidence, not rollout success.

## Command

```bash
python scripts/run_phase4h_scene_gcbc_repair_validation.py --config configs/phase4h_scene_repair_validation_uniform_transition_H5_B192_10000.yaml
```

The run used local OGBench `.npz` fallback because the current environment still
lacks `gymnasium`, so no environment rollout was attempted.

## Training Result

- Dataset: `scene-play-v0`
- Phase 2 graph: `core_plus_bottleneck_budget192_H5`
- Sampling: `uniform_transition`
- Steps: `10000`
- Final validation action MSE: `0.005127`
- Best validation action MSE: `0.005127`
- Prior 200-step smoke final validation action MSE: `0.024553`

## Direct Repair-Edge Evidence

Compared with the previous 200-step Scene Phase 4G run:

- Mean direct repair-edge action MSE improved from `0.038238` to `0.011464`.
- Median direct repair-edge action MSE improved from `0.023925` to `0.005753`.
- Mean direct policy support score improved from `0.561399` to `0.833299`.
- Direct repair-edge certified rate improved from `0.870` to `0.894`.
- Transfer-vs-direct Spearman decreased from `0.547` to `0.498`, reinforcing
  that transfer scores are a coarse fallback, not a precise direct-evidence
  ranker.

## Planner Impact

Coverage did not change because the graph, queries, and compatibility rules were
unchanged:

- `support_shortest_path`: coverage `0.510`, uncertified edge fraction
  `0.356863 -> 0.345425`.
- `calibrated_compat_penalized`: coverage `0.510`, uncertified edge fraction
  `0.196405 -> 0.184967`.
- `compat_threshold`: coverage `0.480`, pair incompatible fraction `0.000`,
  uncertified edge fraction `0.164484 -> 0.159797`.
- `calibrated_compat_threshold`: coverage `0.480`, pair incompatible fraction
  `0.000`, uncertified edge fraction `0.037335 -> 0.033169`.

## Interpretation

The Scene direct repair-edge evidence is not just an artifact of the 200-step
smoke model. A stronger 10000-step GCBC substantially improves supervised
repair-edge fitting while preserving the Phase 4E/4F structural result:
support-only graph repair plus compatibility-aware planning recovers much higher
compatibility-safe Scene coverage than the compressed base graph.

This still does not prove policy execution. The next execution-level claim
requires env availability and closed-loop rollout or a natural-start rollout
protocol.
