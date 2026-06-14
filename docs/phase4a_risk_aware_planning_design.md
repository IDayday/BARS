# Phase 4A Risk-Aware Offline Support Planner Design

Phase 4A is a reset-free offline planning experiment. It does not train a new
policy, does not run environment rollout, and does not claim closed-loop
success. It asks whether Phase 3E edge certification can improve graph path
selection inside the existing Phase 2 support-certified option graph.

## Motivation

Phase 2 and Phase 3E show that path coverage alone is misleading. kNN,
proximity, and random graphs can look connected while using unsupported
shortcuts. Phase 3E also shows that even data-supported option edges have
different heldout support, GCBC fitting proxy, compatibility context, and OOD
risk.

The next algorithmic question is therefore narrow:

Can a planner reduce path risk by using offline edge certification scores,
without adding unsupported edges?

## Related Work and Code Review

Reviewed before implementation:

- GAS, Graph-Assisted Stitching for Offline Hierarchical Reinforcement
  Learning: <https://arxiv.org/abs/2506.07744>
- GAS official implementation in this repository under `external_src/GAS/`.
  The keygraph code builds distance-threshold edges and then connects strongly
  connected components before running shortest-path planning.
- Search on the Replay Buffer: Bridging Planning and Reinforcement Learning:
  <https://arxiv.org/abs/1906.05253>
- Google Research SoRB code:
  <https://github.com/google-research/google-research/tree/master/sorb>
- Test-Time Graph Search for Offline Goal-Conditioned Reinforcement Learning:
  <https://openreview.net/forum?id=PDG8COkj7t>
- TTGS code:
  <https://github.com/ktolnos/ttgs>

Key takeaways:

- GAS and related graph-search methods support the idea that graph planning can
  improve long-horizon goal-conditioned behavior.
- The reviewed GAS code uses thresholded graph edges and shortest paths, but
  edge support provenance is not the primary graph constraint.
- SoRB/TTGS-style methods also search over replay-buffer states, but distance or
  value estimates still need careful interpretation when the graph creates
  shortcuts.
- A mature BARS planner should keep support certification as the hard edge
  provenance boundary, then use offline risk scores to choose among supported
  paths.

## Algorithm

Inputs:

- Phase 2 `option_edges.csv`.
- Phase 3E `offline_edge_certification.csv`.
- Phase 2 `path_queries.csv`.

No new edges are created. All planner variants operate only over Phase 2 support
edges.

Planner variants:

- `support_shortest_path`: baseline shortest path with Phase 2 edge cost.
- `certified_only`: hard filter to edges with `certified_offline_binary=true`.
- `proxy_threshold`: hard filter to edges passing `edge_proxy_score` and
  `heldout_support_lcb` thresholds.
- `proxy_penalized`: use all support edges but multiply base edge cost by a
  penalty from proxy risk, OOD score, incompatibility, and uncertified status.

Default proxy-penalized cost:

```text
planning_cost = base_cost *
  (1
   + risk_weight * (1 - clipped(edge_proxy_score))
   + ood_weight * edge_ood_score
   + incompat_weight * outgoing_incompatible_fraction
   + uncertified_weight * I[not certified])
```

This is a scalarized risk-aware shortest path, not a proof of constrained
optimality. Its value is interpretability and direct alignment with the current
offline evidence.

## Evidence Standard

Phase 4A can support claims about offline path selection only. Required metrics:

- path coverage.
- planning cost and base support cost.
- mean and minimum edge proxy score along reachable paths.
- heldout support LCB along paths.
- uncertified, low-proxy, low-support, high-OOD, and high-incompatibility edge
  fractions.
- risk-adjusted path score, reported as a proxy only.
- common-reachable deltas against `support_shortest_path`.

Phase 4A cannot support these claims:

- learned policy execution success.
- calibrated rollout success probability.
- online performance improvement over GAS.
- reset-to-state support or lack of support.

## Expected Interpretation

A useful Phase 4A result is not necessarily higher raw coverage. It should show
whether risk-aware planning can preserve a reasonable fraction of support graph
coverage while choosing paths with higher certification evidence and lower path
risk. If coverage collapses under hard filtering, that is still useful: it
quantifies how much of the current graph connectivity depends on low-certainty
edges.
