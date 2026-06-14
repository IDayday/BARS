# Phase 4J Mixed/Loss-Weighted GCBC Summary

Phase 4J tested soft per-edge loss weighting as an alternative to the hard
edge-level oversampling that failed in Phase 4I. This remains offline supervised
action fitting, not rollout success.

## Command

```bash
python scripts/run_phase4j_loss_weighting_study.py --config configs/phase4j_loss_weighting_scene_H5_B192_3000.yaml
```

The run used `scene-play-v0`, Phase 2
`core_plus_bottleneck_budget192_H5`, 3000 training steps, and seeds `[0, 1]`.
The baseline rows are the Phase 4I `uniform_transition` runs with the same graph
and seed set.

## Methods

All weighted methods keep `uniform_transition` sampling. Only the supervised
loss changes:

- `loss_support_s03`: inverse-sqrt support weight, strength `0.3`.
- `loss_bottleneck_s03`: bottleneck-score weight, strength `0.3`.
- `loss_support_bottleneck_s03`: combined support and bottleneck weight,
  strength `0.3`.

Weights are clipped to `[0.7, 1.8]`.

## Metrics

| method | final_val_action_mse | final_ratio | rare_edge_mean_mse | rare_ratio |
| --- | ---: | ---: | ---: | ---: |
| uniform_transition_none | 0.008046 | 1.000 | 0.009620 | 1.000 |
| loss_support_s03 | 0.008233 | 1.023 | 0.009315 | 0.968 |
| loss_bottleneck_s03 | 0.008322 | 1.034 | 0.009200 | 0.956 |
| loss_support_bottleneck_s03 | 0.008177 | 1.016 | 0.009038 | 0.940 |

`loss_support_bottleneck_s03` is the recommended variant under the configured
5% overall-regret rule.

## Analysis

Phase 4J is the first training-side improvement after Phase 4H that gives a
cleaner trade-off than the baseline. The combined support+bottleneck loss weight
reduces rare-edge mean MSE by about 6.0% while increasing final validation MSE
by only about 1.6%. The low-support and long-horizon groups improve by about
7.2% and 8.2%, respectively.

This supports the direction suggested by Phase 4I: do not hard oversample rare
edges; keep transition coverage broad and use small, clipped loss-side weights.

## Claim Boundary

This result does not prove option execution or online task performance. It only
supports a better offline supervised fitting trade-off for Scene H5 GCBC. The
next check should reuse this model in direct repair-edge policy evidence and,
when env dependencies are available, closed-loop rollout.
