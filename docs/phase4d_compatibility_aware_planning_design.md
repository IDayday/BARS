# Phase 4D Compatibility-Aware Planning Design

Phase 4D targets path composability. It remains reset-free and offline-only:
no environment rollout, no new policy training, and no online success claim.

## Target Failure Mode

Phase 4C improved independent edge reliability, but Scene still showed high
path-level incompatibility exposure. The remaining failure mode is second order:
edge `a -> b` and edge `b -> c` may each have support, yet the termination
samples of the first edge may not bridge to the initiation samples of the next
edge within a short horizon.

Phase 4D therefore treats option paths as sequences of edges and adds
transition-dependent compatibility costs between adjacent edges.

## Related Work Review

Reviewed before implementation:

- GAS-style graph planning, including the local `external_src/GAS` graph and
  shortest-path code, where long-horizon control is decomposed through graph
  subgoals.
- Search on the Replay Buffer, which motivates replay-buffer graph search for
  long-horizon goal-conditioned control.
- Test-Time Graph Search, which improves offline goal-conditioned policies by
  graph search at evaluation time.
- The local Phase 2.2 compatibility metric cleanup, especially
  `termination_bridge_coverage`, which is the probability-like bridge metric
  safe to aggregate.

The Phase 4D difference is that it does not let graph search create unsupported
proximity shortcuts. It keeps Phase 2 support edges as a hard provenance
boundary, then scores whether supported edges compose with their neighbors.

## Planner

The planner uses a line-graph Dijkstra formulation. Search states are option
edges rather than clusters. A transition from edge `e_ab` to edge `e_bc` is
allowed only when `dst(e_ab) == src(e_bc)`.

Implemented methods:

- `support_shortest_path`: Phase 2 support graph shortest path.
- `calibrated_edge_penalized`: Phase 4C calibrated single-edge risk cost.
- `compat_penalized`: support edges plus adjacent-edge bridge penalty.
- `calibrated_compat_penalized`: calibrated edge risk plus bridge penalty.
- `compat_threshold`: rejects adjacent transitions below a bridge coverage
  floor.
- `calibrated_compat_threshold`: calibrated edge risk with bridge threshold.

For pair-penalized methods:

```text
transition_cost(e_next | e_prev) =
  edge_cost(e_next) + pair_weight * (1 - termination_bridge_coverage(e_prev, e_next))
```

For threshold methods, adjacent pairs with
`termination_bridge_coverage < min_pair_coverage` are not traversed.

## Metrics

Phase 4D reports:

- path coverage;
- base path cost and planning cost;
- calibrated edge reliability and original uncertified edge fraction;
- mean and minimum adjacent-pair termination bridge coverage;
- pair strict-compatible rate and incompatible fraction;
- low-pair-coverage and missing-pair fractions.

These are graph-layer and offline segment-composability metrics. They are not
closed-loop execution metrics.

