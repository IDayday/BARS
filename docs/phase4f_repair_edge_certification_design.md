# Phase 4F Repair-Edge Certification Design

Phase 4F addresses the main caveat from Phase 4E: support-bank repair edges
improved compatibility-safe coverage, but they were not included in Phase 3E/4C
edge certification. This phase remains reset-free and offline-only.

## Target Failure Mode

Phase 4E showed that Scene has many useful support-certified repair edges in the
broader `all` Phase 2 graph. However, planners treated those edges as missing
certification, lowering edge proxy metrics and increasing uncertified fractions.

Phase 4F builds conservative planner-facing certification for repair edges so
that repaired graphs can be evaluated under reliability-aware planning.

## Related Work Review

Reviewed before implementation:

- Offline RL behavior-support and pessimism ideas such as BCQ and CQL, which
  motivate conservative treatment of actions or transitions outside reliable
  behavior support.
- Replay-buffer graph search methods such as SoRB and TTGS, where graph edges
  derive value from observed replay support.
- Phase 4C component-wise edge calibration.
- Phase 4E support-only graph repair outputs.

Because no closed-loop labels are available, Phase 4F does not claim true
calibration or execution probability.

## Certification Transfer

Base edges retain their Phase 4C certification rows.

Repair edges get `certification_source = repair_transfer_proxy`. Their
planner-facing reliability uses:

- support scale from `num_segments`, `num_unique_starts`, and `num_episodes`;
- endpoint-neighbor transfer from base-edge `edge_policy_support_score`;
- behavior reliability from support/diversity and endpoint familiarity;
- augmented incoming/outgoing pair compatibility;
- Phase 4C-style conservative geometric aggregation.

The planner still expects a `heldout_support_lcb` column. For repair edges this
column is a support-transfer proxy, not an actual heldout episode lower
confidence bound.

## Evaluation

Phase 4F evaluates the augmented Phase 4E graph with the new combined
certification table and reports:

- path coverage;
- compatibility metrics;
- current `uncertified_edge_fraction` under repair-transfer certification;
- original uncertified fraction where available;
- repair edge fraction on planned paths;
- repair certified fraction on planned paths.

These are offline graph and supervised-proxy metrics. They do not prove rollout
success.

