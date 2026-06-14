# Phase 4E Compatibility Graph Repair Design

Phase 4E targets the structural composability gap exposed by Phase 4D. It
remains reset-free and offline-only: no environment rollout, no policy training,
and no online success claim.

## Target Failure Mode

Phase 4D showed that Scene paths can become compatibility-safe only by losing
coverage and roughly doubling path cost. This suggests the compressed
`core_plus_bottleneck` graph is missing support-certified detours that connect
compatible initiation and termination regions.

Phase 4E therefore repairs the compressed graph using a larger Phase 2 support
bank. It never introduces kNN, proximity, random, or learned-distance edges.
Every added edge must already exist in a Phase 2 support-certified option graph.

## Related Work Review

Reviewed before implementation:

- GAS-style subgoal graph planning and local `external_src/GAS` graph-search
  code, where graph connectivity drives long-horizon subgoal planning.
- Search on the Replay Buffer and Test-Time Graph Search, which motivate graph
  search over replay-supported subgoals.
- Phase 2.2 compatibility metrics, especially `termination_bridge_coverage`.
- Phase 4D line-graph planning, which showed that path composition is a
  transition-dependent property.

The Phase 4E repair differs from proximity augmentation: repair edges are not
created from distance. They are selected from a support-certified edge bank.

## Algorithm

Inputs:

- base graph: the compressed Phase 2 run, for example
  `core_plus_bottleneck_budget192_H5`;
- repair bank: a broader Phase 2 support graph at the same horizon, for example
  `all_budget192_H5`;
- base pair compatibility computed from `edge_segments.npz`;
- path queries and optional Phase 4C calibrated edge certification.

Steps:

1. Compute base adjacent-edge compatibility.
2. Score bad junctions using low `termination_bridge_coverage` adjacent pairs.
3. Select support-bank edges not already present in the base graph.
4. Prioritize edges touching bad junctions/endpoints, high segment support,
   diverse starts/episodes, and shorter median horizon.
5. Reassign repair edge ids, merge their real edge segments, and recompute
   adjacent-edge compatibility on the augmented graph.
6. Re-run Phase 4D compatibility-aware planners on identical queries.

## Evidence Standard

Phase 4E can support:

- whether support-only graph augmentation recovers compatibility-safe coverage;
- whether Scene's failure is structural compression rather than absence of
  support in the broader offline graph;
- whether added support-bank edges should be certified before being used by
  calibrated planners.

Phase 4E cannot support:

- closed-loop option execution;
- online benchmark performance;
- rollout success probability for added edges.

