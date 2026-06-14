# Phase 4I Stronger GCBC Sampling Study Summary

Phase 4I tested whether edge-balanced sampling improves Scene GCBC supervised
fitting on rare option edges. This is offline action-fitting evidence only, not
rollout success.

## Command

```bash
python scripts/run_phase4i_sampling_study.py --config configs/phase4i_sampling_scene_H5_B192_3000.yaml
```

The run used `scene-play-v0`, Phase 2
`core_plus_bottleneck_budget192_H5`, 3000 training steps, and seeds `[0, 1]`.
The current environment still lacks `gymnasium`, so no rollout was attempted.

## Implementation Result

Two new samplers were added:

- `support_balanced`: samples edges proportional to
  `1 / sqrt(num_unique_starts)`.
- `bottleneck_support_balanced`: combines inverse-sqrt support with a normalized
  bottleneck multiplier.

The sampler hot path was also accelerated by replacing per-sample NumPy
`Generator` construction with deterministic integer-hash sampling. This reduced
the slowdown observed for edge-level samplers during the Phase 4I run.

## Metrics

Baseline is `uniform_transition`.

| sampling_mode | final_val_action_mse | final_ratio | rare_edge_mean_mse | rare_ratio |
| --- | ---: | ---: | ---: | ---: |
| uniform_transition | 0.008046 | 1.000 | 0.009620 | 1.000 |
| uniform_edge | 0.009178 | 1.141 | 0.008850 | 0.920 |
| bottleneck_weighted | 0.009863 | 1.226 | 0.009376 | 0.975 |
| support_balanced | 0.010109 | 1.256 | 0.009088 | 0.945 |
| bottleneck_support_balanced | 0.010770 | 1.339 | 0.009654 | 1.004 |

`rare_edge_mean_mse` is the mean of bottleneck-edge, low-support-edge, and
long-horizon-edge validation MSE.

## Analysis

The simple support-aware samplers did not produce a clean improvement. They can
reduce rare-edge MSE, but the overall validation regression is too large under
the configured 5% regret tolerance. `uniform_edge` is the most interesting
non-default sampler: rare-edge mean MSE improves by about 8%, but final
validation MSE worsens by about 14%.

The current recommendation remains `uniform_transition` for Scene H5 GCBC
training. The useful lesson is negative: naive edge-level oversampling is not
enough. A more promising next direction is a softer mixture or loss-weighted
objective that preserves transition coverage while giving controlled extra
weight to rare/bottleneck edges.

## Claim Boundary

This result only compares offline supervised action fitting. It does not prove
edge execution, path execution, or online task success.
