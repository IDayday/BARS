# Stage47 Actor-Aware Offline Contract Results

Last updated: 2026-06-16

## Offline Boundary

Stage47 remains an offline-RL diagnostic and method-development step.

- Training data: fixed OGBench `.npz` datasets under `/mnt/project/offlinerl_datasets/ogbench`.
- Learned/scored artifacts: GAS keygraphs, dataset embeddings, frozen GAS policies, and derived offline contract rows.
- No online environment interaction is used to build contract rows, fit the scorer, or compute actor-agreement features.
- Environment rollouts are used only for reporting/diagnosis of already fixed graph variants. They must not become training examples or adaptive threshold-tuning data.

## What Was Added

New code:

- `scripts/stage47_add_actor_agreement_features.py`

This script augments Stage45 offline contract rows with frozen-actor agreement features:

- `actor_action_mse`
- `actor_action_l2`
- `actor_pred_action_norm`
- `dataset_action_norm`
- `actor_skill_norm`

The feature computation uses only offline states/actions and a frozen GAS policy. It aligns raw OGBench terminal rows to the GAS/GCDataset protocol by dropping raw terminal rows before indexing.

Stage42 path-local reranking was also extended to accept contract probability/risk columns, but current results indicate that path-local gating is not the right main direction.

## Stage45 Result Summary

Stage45 built offline contract datasets and a non-actor-aware logistic scorer on:

- `antmaze-giant-navigate-v0`
- `antmaze-large-explore-v0`
- `scene-play-v0`

Combined logistic scorer:

| split | AP | ROC-AUC | Brier |
| --- | ---: | ---: | ---: |
| val | 0.509 | 0.961 | 0.076 |
| test | 0.569 | 0.963 | 0.077 |

The scorer is substantially better than single-feature support/distance baselines, but the labels are local offline contract labels rather than environment success. This supports the idea that a learned contract model contains useful signal, but it is not yet a complete algorithmic claim.

## Stage45 Closed-Loop Evaluation

Global CAP-lite soft-risk graph patch:

| env | original | support-only best/representative | CAP-lite global | note |
| --- | ---: | ---: | ---: | --- |
| `antmaze-giant-navigate-v0` | 0.76 | 0.83 (`support w0.10`) | 0.81 (`contract w0.25`) | improves over original, below best support-only |
| `antmaze-large-explore-v0` | 0.94 | 0.98 (`support w0.50`) | 0.95 (`contract w0.25`) | small gain over original, below support-only |
| `scene-play-v0` | 0.73 | 0.70 (`support w5`) | 0.76 (`contract w2`) | avoids support-only regression and improves original |

Path-local CAP-lite was negative:

| env/variant | success |
| --- | ---: |
| `giant_pathlocal_aggressive` | 0.79 |
| `giant_pathlocal_conservative` | 0.77 |
| `large_pathlocal` | 0.93 |
| `scene_pathlocal` | 0.73 |

Interpretation: global soft contract risk is useful; path-local replacement/gating is too brittle in the current form.

## Stage47 Actor-Aware Scorer

Actor-agreement feature outputs:

| env | rows | actor MSE mean | actor MSE median | actor MSE p90 |
| --- | ---: | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 20,000 | 0.087 | 0.050 | 0.191 |
| `antmaze-large-explore-v0` | 20,000 | 0.886 | 0.815 | 1.531 |
| `scene-play-v0` | 20,000 | 0.044 | 0.018 | 0.125 |

The large-explore actor mismatch is much higher than giant/scene, which is a useful diagnostic. However, adding these raw features to the current local-label scorer gives only a small metric gain:

| scorer | val AP | test AP | val ROC-AUC | test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Stage45 contract scorer | 0.509 | 0.569 | 0.961 | 0.963 |
| Stage47 actor-aware scorer | 0.515 | 0.572 | 0.960 | 0.963 |

Dominant coefficients are still distance/support-like features:

- `local_support`: positive and large.
- `phi_dist_pair` / `actor_skill_norm`: strongly negative.
- action agreement terms are smaller and mixed.

Interpretation: raw frozen-actor action agreement is not enough by itself. The current label definition still lets distance/support dominate.

## Actor-Aware Edge Scores And Path Audit

Actor-aware row probabilities were aggregated into edge scores:

| env | scored edges | mean edge prob | median edge prob | supported edge rate |
| --- | ---: | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 6,511 | 0.1085 | 1.07e-05 | 0.1209 |
| `antmaze-large-explore-v0` | 15,727 | 0.0437 | 4.77e-06 | 0.0499 |
| `scene-play-v0` | 17,482 | 0.0297 | 9.77e-09 | 0.0315 |

Compared with Stage45 non-actor-aware scores, these are nearly unchanged. The resulting path changes are also almost identical:

| env | Stage45 path change rate | Stage47 actor-aware path change rate | conclusion |
| --- | ---: | ---: | --- |
| `antmaze-giant-navigate-v0` | 0.5442 | 0.5444 | one extra changed path only |
| `antmaze-large-explore-v0` | 0.7946 | 0.7946 | identical at this audit level |
| `scene-play-v0` | 0.1409 | 0.1409 | identical at this audit level |

This is the main Stage47 conclusion: adding frozen-actor agreement as ordinary row features does not materially change graph decisions. It is a useful negative result and points to the label/objective rather than the feature availability as the current bottleneck.

## Research Implications

The current evidence supports a shift from "support-weight patching" to a fuller algorithm:

```text
BARS-CAP = offline contract learning + actor-conditioned edge reliability +
planner-grounded graph risk + offline-only policy/planner training signals.
```

What is promising:

- Contract scores can improve Scene where support-only regresses.
- Contract probabilities explain some edge reliability beyond support count.
- The same framework works across AntMaze and Scene without online training.

What is weak:

- Stage45 global CAP-lite is not consistently better than the best support-only AntMaze weight.
- Stage42/45 path-local gating underperforms global soft risk.
- Stage47 raw actor features barely change edge decisions.

Therefore the next algorithmic step should not be another weight sweep. It should change the contract target.

## Stage48 Follow-Up: Actor-Conditioned Labels

Stage48 adds a stricter offline label transform:

- preserve the original Stage45 local labels as `label_reach_base`;
- compute the train-positive actor-MSE 75th percentile per environment;
- demote original positive rows with higher actor MSE into hard negatives;
- preserve all original data provenance and keep the output compatible with the Stage45 scorer.

New code:

- `scripts/stage48_build_actor_conditioned_contract_labels.py`

Output:

- `runs_stage48_actor_conditioned_labels/20260616_063500/actor_conditioned_contract_pairs.csv`

Label changes:

| env | base positive rate | actor-conditioned positive rate | demoted positives |
| --- | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.0587 | 0.0435 | 240 / 25.9% |
| `antmaze-large-explore-v0` | 0.0207 | 0.0158 | 75 / 23.6% |
| `scene-play-v0` | 0.0243 | 0.0180 | 90 / 25.9% |

Actor-conditioned scorer:

| split | AP | ROC-AUC | Brier |
| --- | ---: | ---: | ---: |
| val | 0.353 | 0.951 | 0.093 |
| test | 0.470 | 0.957 | 0.091 |

These metrics are lower than Stage47 because the label is stricter and the
positive class is smaller. That is not automatically a failure; the key question
is whether edge scores and graph decisions change.

Stage48 edge scores:

| env | scored edges | mean edge prob | median edge prob | supported edge rate |
| --- | ---: | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 6,511 | 0.1088 | 1.50e-05 | 0.1212 |
| `antmaze-large-explore-v0` | 15,727 | 0.0392 | 2.59e-06 | 0.0453 |
| `scene-play-v0` | 17,482 | 0.0296 | 3.13e-08 | 0.0313 |

Stage48 path audit:

| env | Stage45 change rate | Stage47 change rate | Stage48 change rate |
| --- | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.5442 | 0.5444 | 0.5444 |
| `antmaze-large-explore-v0` | 0.7946 | 0.7946 | 0.7946 |
| `scene-play-v0` | 0.1409 | 0.1409 | 0.1436 |

Stage48 conclusion:

```text
Demoting high actor-MSE positives is a better target than raw actor features,
but this first proxy is still too weak to materially change graph decisions.
It slightly affects large/scene edge probabilities and changes 10 additional
Scene paths, but it does not justify a new closed-loop GPU evaluation while
Stage44 visual training is still occupying GPU2.
```

The stronger next step is sequence-level actor-conditioned reachability, not
single-step action MSE.

## Stage49: Sequence-Level Actor Contract

Stage49 implements the sequence-level version of actor-conditioned labels:

- sample states along fixed offline same-trajectory segments;
- condition the frozen GAS actor on the current latent-to-target skill;
- compare predicted action against the offline action at each sampled state;
- measure latent progress along the offline segment;
- demote positives that fail the sequence actor/progress contract.

New code:

- `scripts/stage49_add_sequence_actor_contract_features.py`

Run:

- `runs_stage49_sequence_actor_contract/20260616_071148`

The full Stage49 feature jobs were launched in parallel on CPU with `setsid`; no
GPU training or online environment interaction was used.

Label changes:

| env | rows with sequence | base positive rate | sequence positive rate | demoted positives | sequence hard negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 3,249 | 0.0587 | 0.0300 | 48.8% | 949 |
| `antmaze-large-explore-v0` | 2,109 | 0.0207 | 0.0108 | 47.8% | 516 |
| `scene-play-v0` | 1,207 | 0.0243 | 0.0141 | 42.1% | 266 |

Sequence-level scorer:

| split | AP | ROC-AUC | Brier | edge AP | edge ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 0.590 | 0.989 | 0.0366 | 0.524 | 0.980 |
| test | 0.644 | 0.987 | 0.0399 | 0.660 | 0.982 |

This is a much stronger offline discriminative signal than Stage48. The main
coefficients now include `seq_has_segment`, `seq_mean_progress_delta`, and
negative weight on `seq_actor_action_mse_mean`, which is the expected direction.

Directly replacing graph risk with Stage49 sequence scores is too conservative:

| env | Stage45 supported edge rate | Stage49 direct supported edge rate |
| --- | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.1209 | 0.0258 |
| `antmaze-large-explore-v0` | 0.0511 | 0.0121 |
| `scene-play-v0` | 0.0316 | 0.0152 |

Path audit confirms the direct replacement is risky:

| env | Stage45 path change rate | Stage49 direct path change rate | key issue |
| --- | ---: | ---: | --- |
| `antmaze-giant-navigate-v0` | 0.5442 | 0.4699 | reverts toward original support profile |
| `antmaze-large-explore-v0` | 0.7946 | 0.7927 | similar change rate but lower same-trajectory support |
| `scene-play-v0` | 0.1409 | 0.1540 | more path churn and lower support profile |

Conclusion: Stage49 is useful as an actor verifier, not as a standalone global
edge prior.

## Stage50: Broad Prior Plus Sequence Verifier

Stage50 composes the broad Stage45 contract prior with the Stage49 sequence
verifier:

```text
hybrid_prob = base_prob * (1 - sequence_penalty_strength * evidence_gate * sequence_risk)
```

New code:

- `scripts/stage50_make_hybrid_contract_edge_scores.py`

Hybrid edge score run:

- `runs_stage49_sequence_actor_contract/20260616_071148/caplite_hybrid_stage45_sequence`

Hybrid edge-score effect:

| env | changed edge rate | mean base prob | mean hybrid prob | supported edge rate |
| --- | ---: | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.341 | 0.1072 | 0.0952 | 0.1201 |
| `antmaze-large-explore-v0` | 0.161 | 0.0460 | 0.0431 | 0.0486 |
| `scene-play-v0` | 0.071 | 0.0299 | 0.0283 | 0.0301 |

Hybrid path audit:

| env | Stage45 change rate | Stage49 direct | Stage50 hybrid | interpretation |
| --- | ---: | ---: | ---: | --- |
| `antmaze-giant-navigate-v0` | 0.5442 | 0.4699 | 0.5514 | hybrid is close to Stage45 but slightly more aggressive |
| `antmaze-large-explore-v0` | 0.7946 | 0.7927 | 0.7946 | hybrid preserves Stage45 paths |
| `scene-play-v0` | 0.1409 | 0.1540 | 0.1415 | hybrid avoids direct sequence over-pruning |

Stage50 is the best current algorithmic shape:

```text
Use Stage45 as a broad offline reachability prior.
Use Stage49 as a sequence-level actor verifier only where sequence evidence
exists.
Avoid replacing the whole graph with the sparse sequence verifier.
```

This is closer to a complete ICLR-level algorithm than the earlier support-only
or single-feature variants, but the closed-loop result is mixed. Stage44 has now
finished, and the predeclared Stage50 closed-loop evaluation completed on GPU3:

| job | env | Stage45 | Stage50 hybrid | result |
| --- | --- | ---: | ---: | --- |
| `gn_hybrid_w0p25` | `antmaze-giant-navigate-v0` | 0.81 | 0.82 | small positive, not decisive |
| `large_explore_hybrid_w0p25` | `antmaze-large-explore-v0` | 0.95 | 0.95 | unchanged |
| `scene_hybrid_w2` | `scene-play-v0` | 0.76 | 0.73 | regression |

Run root:

```text
runs_stage50_hybrid_eval_gpu3/20260616_140855
```

This evaluation is for reporting and diagnosis only. It must not be used to
train labels, tune thresholds, or add online data to the offline-RL method.

Follow-up Stage51 offline diagnosis shows that Scene regression is caused by a
small number of path changes that lower the offline support profile. Conservative
drop caps and positive-only sequence boosts are safe but mostly revert to Stage45
paths. A strict path-safety overlay protects Scene but also rejects all giant
Stage50 candidate path changes under current offline metrics.

Detailed result note:

```text
docs/stage50_hybrid_eval_results.md
```

## Next Plan

P0: Actor-conditioned contract labels.

- Build labels that explicitly ask whether the frozen/offline-trained actor can execute from edge source to edge target, not just whether a same-trajectory local pair exists.
- Use offline-only proxies: action likelihood, value/TD consistency, temporal reachability under matched goal-conditioning, and hard negatives where geometric support exists but actor agreement is poor.
- Keep episode-level train/val/test splits.
- Upgrade Stage48 from single-step action MSE to sequence-level actor-conditioned reachability:
  - aggregate action agreement over offline segments, not only the first state;
  - compare actor skill progress in latent space along the segment;
  - treat high-support but low-progress edges as hard negatives;
  - train an edge-ranking objective that directly changes graph ordering.

P0: Global soft-risk only for short-term evaluation.

- Use actor-aware edge scores as a fixed diagnostic artifact.
- If evaluating, run only predeclared variants that mirror Stage45 weights, not adaptive sweeps from rollout feedback.
- Do not spend GPU time on path-local variants unless the label/objective changes.

P1: Extend to Stage44 Humanoid/Visual.

- Humanoid local GAS baselines are complete:
  `humanoidmaze-large-navigate-v0` success `0.776`,
  `humanoidmaze-large-stitch-v0` success `0.848`.
- Visual local GAS baselines are complete but weak:
  `visual-antmaze-large-explore-v0` success `0.040`,
  `visual-scene-play-v0` success `0.004`.
- Run the same offline contract/actor-feature pipeline on representative
  Humanoid/Visual tasks.
- Treat visual results as local retrain baselines, not official pretrained-weight
  results; use them as stress tests where a stronger algorithm may show clearer
  improvement.

P1: Toward a complete ICLR-level method.

- Move from graph patching to a named algorithm with offline-learned contract reliability.
- Couple planner risk to policy training targets, so the policy sees the same edge/skill distribution the planner will issue.
- Report support-only and CAP-lite as ablations, not as the final contribution.
