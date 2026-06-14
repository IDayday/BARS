# Phase 4L Direct Repair-Edge Group Diagnostics Design

Phase 4L diagnoses where the Phase 4K loss-weighted repair-edge improvement
comes from.

## Question

Does `loss_support_bottleneck_s03` improve direct repair-edge action fitting on
the intended hard edge groups, or is the aggregate improvement driven by
unstructured noise?

## Scope

- Reuse Phase 4K direct repair-edge outputs.
- Match each candidate checkpoint to the same-seed `uniform_transition_none`
  baseline.
- Compare edge-level MSE deltas on the same repair edge ids.
- Group deltas by support, bottleneck score, horizon, compatibility context,
  planner usage, and repair reason.
- Do not train policies.
- Do not run environment rollout.

## Main Metrics

- `edge_action_mse_delta = candidate_mse - baseline_mse`
- `edge_action_mse_ratio = candidate_mse / baseline_mse`
- `fraction_edges_improved`
- `sample_weighted_mse_delta`
- `planner_usage_rate`

Negative MSE delta means the candidate fits that repair-edge group better than
the matched baseline.

## Evidence Standard

The useful signal is not just an aggregate MSE improvement. A convincing
training-side improvement should help the intended hard groups and preferably
also the repair edges actually used by the repaired planner.

## Claim Boundary

This phase is still reset-free offline supervised diagnostics. It does not prove
closed-loop edge execution or task success.
