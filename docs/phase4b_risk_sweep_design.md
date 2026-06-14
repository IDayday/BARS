# Phase 4B Calibrated Risk-Aware Planner Sweep Design

Phase 4B extends Phase 4A from one fixed risk penalty to a calibrated offline
sweep. It remains reset-free: no environment rollout, no new policy training,
and no online success claim.

## Target Failure Mode

Phase 4A showed that:

- hard certification can collapse graph connectivity;
- soft risk penalties preserve coverage while reducing path risk;
- one fixed set of weights is not enough evidence for a mature algorithm.

Phase 4B targets that gap by searching for stable coverage/risk/cost trade-offs
over risk weights and mild edge floors.

## Related Work Review

Reviewed before implementation:

- GAS official code in `external_src/GAS/K_utils/keygraph_utils.py`, which uses
  threshold graph edges and shortest-path search.
- Search on the Replay Buffer and TTGS, which motivate replay-buffer graph
  planning but still require careful interpretation of graph distances.
- Constrained and multi-objective shortest-path formulations. The relevant
  lesson for BARS is that cost and risk are separate objectives; reporting one
  scalar path cost is not enough to justify a planner.

Phase 4B therefore reports a Pareto front rather than a single unqualified
"best" planner.

## Planner Family

All Phase 4B candidates are support-only. They do not add kNN/proximity/random
edges.

Candidate method:

- `floor_proxy_penalized`

It first applies mild hard floors:

```text
edge_proxy_score >= proxy_floor
heldout_support_lcb >= heldout_support_lcb_floor
```

Then it uses Phase 4A's scalarized risk-aware edge cost:

```text
planning_cost = base_cost *
  (1
   + risk_weight * (1 - clipped(edge_proxy_score))
   + ood_weight * edge_ood_score
   + incompat_weight * outgoing_incompatible_fraction
   + uncertified_weight * I[not certified])
```

Swept parameters:

- `risk_weight`
- `ood_weight`
- `incompat_weight`
- `uncertified_weight`
- `proxy_floor`
- `heldout_support_lcb_floor`

## Metrics

For each sweep config:

- path coverage.
- mean base path cost.
- mean minimum edge proxy score.
- mean heldout support LCB.
- mean uncertified edge fraction.
- mean high-OOD and high-incompatibility fractions.
- coverage/risk derived score.
- Pareto membership.

Pareto dominance uses:

- maximize `path_coverage`;
- maximize `mean_min_edge_proxy_score`;
- minimize `mean_uncertified_edge_fraction`;
- minimize `mean_base_path_cost`.

## Recommendation Rule

The recommended config is selected only as a practical next-run default:

- preserve at least `min_coverage_ratio` of support-shortest-path coverage;
- keep mean base path cost within `max_base_cost_increase`;
- among eligible configs, maximize a coverage/risk heuristic.

This does not prove global optimality or execution success. It only identifies a
reasonable candidate for the next offline or closed-loop validation stage.

## Evidence Boundary

Phase 4B can support:

- offline planner trade-off conclusions;
- whether risk floors are too aggressive;
- whether a soft+floor planner improves path risk without large coverage loss.

Phase 4B cannot support:

- online task success;
- GCBC execution success;
- calibrated probability of edge success;
- superiority over official GAS in environment return.
