# Phase 4G Direct Repair-Edge Policy Evidence Design

Phase 4G strengthens Phase 4F by replacing repair-edge policy-score transfer
with direct GCBC action-fitting evidence on real repair-edge segments. It
remains reset-free and offline-only.

## Target Failure Mode

Phase 4F made repaired graphs planner-compatible by assigning conservative
transfer-proxy certification to repair edges. That was useful for planning, but
it did not directly test whether the trained GCBC policy fits actions on those
repair segments.

Phase 4G evaluates the trained GCBC model on the selected repair-bank segments
and uses the resulting action MSE to update the repair-edge policy component.

## Related Work Review

Reviewed before implementation:

- BCQ, which motivates constraining offline policies toward behavior-supported
  actions instead of trusting extrapolated behavior.
- CQL, which motivates conservative treatment under offline distribution shift.
- Search on the Replay Buffer, where long-horizon planning assumes local policy
  and replay-buffer support.
- Local Phase 3 GCBC edge fitting and Phase 3E policy-likelihood utilities.

Phase 4G is not closed-loop execution. Low action MSE can still fail under
compounding rollout error.

## Method

Inputs:

- Phase 4E `repair_edge_map.csv`;
- Phase 2 repair-bank `option_edges.csv` and `edge_segments.npz`;
- Phase 4F repair-edge certification table;
- trained GCBC `model.pt`;
- OGBench train dataset arrays.

Steps:

1. Map augmented repair edge ids back to support-bank edge ids.
2. Select all bank segment indices belonging to selected repair edges.
3. Expand those segments into one-step GCBC samples.
4. Evaluate the trained GCBC model and compute edge-wise action MSE.
5. Convert MSE to `direct_edge_policy_support_score = exp(-mse / temperature)`.
6. Replace repair-edge transfer policy scores with direct policy scores.
7. Recalibrate edge reliability and re-run repaired-graph planners.

## Evidence Standard

Phase 4G supports:

- whether transfer-proxy repair scores align with direct GCBC action fitting;
- whether direct policy evidence preserves the Phase 4E/4F coverage and
  compatibility gains;
- which repair edges are high-risk under supervised action fitting.

Phase 4G does not support:

- online success;
- reset-to-state edge execution;
- closed-loop option composability under model errors.

