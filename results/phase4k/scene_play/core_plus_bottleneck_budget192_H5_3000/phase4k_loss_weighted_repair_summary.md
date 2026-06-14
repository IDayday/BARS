# Phase 4K Loss-Weighted GCBC Direct Repair-Edge Validation

Phase 4K reuses the Phase 4J GCBC checkpoints and evaluates them on
the same Scene repair-bank segments used by Phase 4G/4H. This checks
whether loss-weighted training improves direct repair-edge evidence,
not only ordinary edge validation MSE. No environment rollout is used.

## Recommendations

- `core_plus_bottleneck_budget192_H5`: `loss_support_bottleneck_s03`
  - direct repair MSE: `0.01531421800867139`
  - direct repair MSE ratio vs baseline: `0.9864332939411989`
  - final validation MSE ratio vs baseline: `1.0163051753898613`
  - direct certified rate: `0.89`

## Baseline Comparisons

| method | final_val_action_mse_ratio_vs_baseline | mean_direct_edge_action_mse_ratio_vs_baseline | direct_certified_rate | mean_uncertified_edge_fraction | mean_min_edge_proxy_score |
| --- | --- | --- | --- | --- | --- |
| loss_bottleneck_s03 | 1.03438 | 1.00982 | 0.889 | 0.0342104 | 0.247242 |
| loss_support_bottleneck_s03 | 1.01631 | 0.986433 | 0.89 | 0.0342104 | 0.247276 |
| loss_support_s03 | 1.02324 | 1.00145 | 0.888 | 0.0342104 | 0.247193 |
| uniform_transition_none | 1 | 1 | 0.887 | 0.0352521 | 0.247316 |

## Interpretation Rules

- A useful method should improve direct repair-edge MSE without a large
  ordinary validation-MSE regression.
- Planner coverage is mostly graph-limited here; model changes mainly
  affect repair-edge policy support, certification, and path risk.
- Direct repair-edge action MSE is still offline supervised evidence and
  does not prove option execution or online task success.

Related work reviewed: GCSL, RvS, Class-Balanced Loss, Focal Loss,
and the GCSL reference implementation.
