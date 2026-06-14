# Phase 4C Edge Risk Calibration Design

Phase 4C improves the edge-risk score used by Phase 4B. It remains reset-free
and offline-only: no environment rollout, no policy training, and no online
success claim.

## Target Failure Mode

Phase 4B showed that direct aggregate `risk_weight` is not consistently the
best planner knob. Recommended configs repeatedly relied on decomposed OOD and
uncertified penalties. Scene also showed that planner weights alone cannot clean
paths when compatibility/certification signals are weak.

Phase 4C therefore separates edge risk into calibrated components before
planning.

## Related Work Review

Reviewed before implementation:

- Calibration of Modern Neural Networks, which motivates separating confidence
  scores from correctness and evaluating calibration rather than assuming raw
  scores are probabilities.
- Offline RL pessimism and behavior-support estimation, which motivates
  conservative treatment of OOD or low-support edges.
- Goal-conditioned graph search methods such as GAS, SoRB, and TTGS, where
  graph search quality depends on whether edge scores represent reliable local
  transitions.

Phase 4C does not claim true probability calibration because we do not have
closed-loop rollout labels. It performs component-wise offline calibration and
reports pseudo-label diagnostics against heldout support.

## Component Scores

Each edge gets five reliability components, all oriented so higher is better:

- `support_reliability`: heldout support lower confidence bound.
- `policy_reliability`: GCBC action-fitting support score.
- `behavior_reliability`: `1 - edge_ood_score`.
- `compatibility_reliability`: geometric combination of termination bridge
  coverage and adjacent-edge compatibility.
- `diversity_reliability`: rank-normalized unique starts and unique episodes.

The calibrated reliability score is a weighted geometric mean:

```text
calibrated_edge_reliability_score =
  geom_mean(
    support_reliability,
    policy_reliability,
    behavior_reliability,
    compatibility_reliability,
    diversity_reliability
  )
```

The geometric mean is intentionally conservative: one weak component lowers the
whole edge score.

## Planner Integration

For Phase 4C sweeps:

- `edge_proxy_score` is replaced by `calibrated_edge_reliability_score`.
- `edge_ood_score` is replaced by `1 - behavior_reliability`.
- `certified_offline_binary` is recomputed from calibrated reliability and
  support/compatibility floors.

The planner then runs the same support-only Phase 4B sweep.

## Evidence Standard

Phase 4C can support:

- whether component-wise calibrated scores align better with heldout support
  pseudo-labels;
- whether calibrated scores improve Phase 4B Pareto fronts;
- whether Scene's bottleneck is score calibration or deeper edge support.

Phase 4C cannot support:

- online execution success;
- calibrated probability of edge success;
- claims that BARS beats GAS in environment return;
- claims about reset-to-state support.
