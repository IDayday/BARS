# Stage63 Adaptive Penalty Results

Date: 2026-06-17

## Goal

Test whether a confidence-weighted offline planner penalty can improve over the
current fixed global `risk_weight` recipe without introducing any online data
into training or calibration.

This stage remains offline-only for score construction. Environment rollouts
are used only for reporting.

## Code

- `scripts/stage63_make_adaptive_risk_edge_scores.py`

The script keeps `contract_prob_edge` fixed and rewrites only `r_exec`:

```text
adaptive_risk
  = base_risk
  * evidence_gate(contract_samples, sequence_evidence_gate)
  * support_relief(local_support, same_traj_support)
```

This lets us test adaptive planner penalties without retraining the scorer.

## Giant Navigate: Negative Result

Run roots:

- `runs_stage63_adaptive_risk/20260617_010810`
- `runs_stage63_adaptive_eval_gpu3/giant_nav_adaptive_w0p25_10ep`

Adaptive candidates:

- `stage45_base_adaptive`
- `stage49_hybrid_adaptive_t5`

10-episode result at `risk_weight=0.25`:

| variant | success |
| --- | ---: |
| original | 0.80 |
| stage45 base adaptive | 0.78 |
| stage49 hybrid adaptive t5 | 0.80 |

Task breakdown:

| variant | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.40 | 1.00 | 1.00 | 0.70 | 0.90 |
| base adaptive | 0.30 | 0.90 | 0.90 | 0.90 | 0.90 |
| hybrid adaptive t5 | 0.50 | 0.90 | 0.90 | 1.00 | 0.70 |

Conclusion:

- current adaptive gating does not solve the giant task tradeoff;
- the gain still moves across tasks instead of improving the full profile;
- this is not enough to replace the current fixed mild penalty recipe.

## Humanoid Candidate Selection

Humanoid actor-conditioned edge scores:

- `runs_stage61_humanoid_confirm_gpu3/actor_conditioned_w0p10_20ep/filtered_actor_conditioned/edge_scores/caplite_edge_scores.csv`

Offline candidate summaries:

| candidate | sample target | evidence floor | mean adaptive risk |
| --- | ---: | ---: | ---: |
| `actor_t2_f0` | 2 | 0.0 | 0.124 |
| `actor_t2_f01` | 2 | 0.1 | 0.202 |
| `actor_t3_f01` | 3 | 0.1 | 0.145 |
| `actor_t2_f01_d75` | 2 | 0.1 | 0.200 |

`actor_t2_f01` was selected because it matched the fixed-penalty scale more
closely while staying concentrated on higher-evidence edges.

## Humanoid Smoke

Run root:

- `runs_stage63_adaptive_eval_gpu3/humanoid_nav_adaptive_w0p50_10ep`

10-episode result at `risk_weight=0.50`:

| variant | success | length |
| --- | ---: | ---: |
| original | 0.80 | 1431.88 |
| actor adapt t2 f01 | 0.86 | 1332.32 |
| actor adapt t3 f01 | 0.82 | 1374.14 |

Task breakdown:

| variant | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1.00 | 0.70 | 1.00 | 0.50 | 0.80 |
| actor adapt t2 f01 | 1.00 | 0.80 | 1.00 | 0.60 | 0.90 |
| actor adapt t3 f01 | 0.90 | 0.60 | 1.00 | 0.60 | 1.00 |

This was strong enough to justify 20-episode confirmation for `actor_t2_f01`.

## Humanoid 20-Episode Confirm

Run root:

- `runs_stage63_adaptive_eval_gpu3/humanoid_nav_adaptive_w0p50_20ep`

Result:

| variant | success | length |
| --- | ---: | ---: |
| original | 0.76 | 1452.93 |
| actor adapt t2 f01 | 0.80 | 1396.01 |

Task breakdown:

| variant | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.90 | 0.65 | 0.95 | 0.50 | 0.80 |
| actor adapt t2 f01 | 0.90 | 0.75 | 0.95 | 0.50 | 0.90 |

Conclusion:

- adaptive penalty is not noise on humanoid;
- it gives a stable `+0.04` over the matched original;
- the gain comes from exactly the desired tasks: `task2` and `task5`.

## Matched Fixed Actor-Controlled Baseline

Run root:

- `runs_stage63_adaptive_eval_gpu3/humanoid_nav_fixed_actor_w0p10_20ep`

Result:

| variant | success | length |
| --- | ---: | ---: |
| original | 0.73 | 1469.73 |
| actor fixed w0p10 | 0.87 | 1433.08 |

Task breakdown:

| variant | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.85 | 0.65 | 1.00 | 0.50 | 0.65 |
| actor fixed w0p10 | 1.00 | 0.75 | 1.00 | 0.65 | 0.95 |

This is the key comparison for Stage63:

- adaptive is better than original;
- adaptive is not better than the current best fixed actor-conditioned recipe.

## Current Position

What Stage63 supports:

1. confidence-weighted offline planner penalties are viable;
2. humanoid gains can be reproduced without relying on a globally saturated
   `r_exec`;
3. adaptive penalty can selectively recover `task2` and `task5`.

What Stage63 does not support:

1. replacing the current `filtered + actor-conditioned + mild fixed penalty`
   mainline;
2. claiming that edge-static adaptive gating already solves the giant tradeoff;
3. claiming that Stage63 is the paper-ready final algorithm.

## Updated Research Judgment

The current strongest mainline remains:

```text
relative endpoint filtering
+ actor-conditioned contract scorer
+ mild fixed planner penalty
```

Stage63 is now a useful supporting branch:

```text
offline confidence-weighted adaptive penalty
```

but it should be treated as a bridge toward the next step, not as the new best
method.

## Next Step

The next improvement should move from static edge-local scaling to path-local
or task-local calibration, for example:

1. planner penalty based on the selected path's risk composition rather than
   one per-edge static scale;
2. edge-type-aware calibration using graph role or long-hop structure;
3. actor-aware adaptive gating that only strengthens penalties when both
   contract risk and policy mismatch agree.
