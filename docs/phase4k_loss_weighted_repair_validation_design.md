# Phase 4K Loss-Weighted Repair Validation Design

Phase 4K tests whether the Phase 4J loss-weighted GCBC improvement transfers
from ordinary edge validation MSE to direct repair-edge evidence.

## Question

Do clipped support/bottleneck loss weights improve supervised action fitting on
the selected support-bank repair edges used by the Phase 4E/4F/4G repaired
planner?

## Scope

- Reuse existing Phase 4I baseline and Phase 4J weighted GCBC checkpoints.
- Reuse Phase 4G direct repair-edge scoring and repaired planner evaluation.
- Do not train a new policy.
- Do not construct environments.
- Do not run rollout.
- Do not interpret direct action MSE as option execution success.

## Compared Methods

- `uniform_transition_none`
- `loss_support_s03`
- `loss_bottleneck_s03`
- `loss_support_bottleneck_s03`

All methods use the same Scene Phase 2 graph:
`core_plus_bottleneck_budget192_H5`.

## Main Metrics

- `mean_direct_edge_action_mse`
- `mean_direct_policy_support_score`
- `direct_certified_rate`
- `final_val_action_mse`
- `mean_min_edge_proxy_score`
- `mean_uncertified_edge_fraction`

The primary recommendation rule is: choose the method with the lowest direct
repair-edge MSE among methods whose ordinary validation MSE is within the
configured regret bound versus `uniform_transition_none`.

## Related Work Reviewed

- Goal-Conditioned Supervised Learning: supervised goal-conditioned policy
  training framing.
- RvS: supervised offline RL framing and claim-boundary caution.
- Class-Balanced Loss: loss-side reweighting under long-tail sample imbalance.
- Focal Loss: loss-side focusing without changing data provenance.
- GCSL reference implementation: simple open-source GCBC-style training
  baseline.

## Claim Boundary

Phase 4K is reset-free offline supervised evidence. A better direct repair-edge
MSE makes the repaired graph's policy proxy more credible, but it still does not
prove closed-loop edge execution or online benchmark success.
