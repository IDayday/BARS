# Stage54 Relative-Endpoint Filter Results

Last updated: 2026-06-16

## Offline Boundary

Stage54/55 remain offline-RL method-development steps.

- Contract rows are built only from fixed local OGBench datasets and frozen
  GAS artifacts.
- Actor features use fixed offline states/actions and frozen GAS policies.
- No online environment interaction is used for label construction, scorer
  fitting, or graph patch generation.
- Rollouts are used only for reporting.

## Code Change

Updated:

- `scripts/stage45_build_offline_contract_dataset.py`

New optional flag:

- `--max-endpoint-dist-ratio`

Meaning:

- keep only endpoint-neighbor states satisfying
  `node_dist <= ratio * edge_phi_dist`;
- if the filter would remove all candidates for one endpoint, keep the nearest
  fallback state so the edge remains representable;
- default `<= 0` keeps the old behavior unchanged.

This is an edge-relative support filter. The goal is to reduce false positive
support in dense latent spaces without hard-coding one global radius per
environment family.

## Visual Density Hypothesis

Stage52 showed that the local-retrain visual graph was excessively dense:

| env | support edge rate | scorer test AP | scorer test ROC-AUC |
| --- | ---: | ---: | ---: |
| `visual-antmaze-large-explore-v0` Stage52 baseline | 0.789 | 0.532 | 0.819 |

We tested three endpoint ratios on the same Stage44 local-retrain artifact:

Run root:

```text
runs_stage54_actor_visual_density/20260616_144721/sweep_v3
```

| variant | support edge rate | positive row rate | test AP | test ROC-AUC | test Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| `visual ratio=0.4` | 0.266 | 0.127 | 0.635 | 0.922 | 0.117 |
| `visual ratio=0.5` | 0.324 | 0.150 | 0.730 | 0.924 | 0.119 |
| `visual ratio=0.6` | 0.372 | 0.177 | 0.686 | 0.914 | 0.123 |

Conclusion:

- the dense visual support graph was a real problem, not just a metric artifact;
- edge-relative endpoint filtering improves visual offline discrimination
  substantially;
- `ratio=0.5` is the current sweet spot;
- `ratio=0.4` over-tightens;
- `ratio=0.6` starts drifting back toward the dense baseline.

## Humanoid Check

The same filter did not collapse on humanoid:

| variant | support edge rate | positive row rate | test AP | test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| `humanoid ratio=0.5` | 0.058 | 0.071 | 0.896 | 0.982 |

Interpretation:

- the relative filter is not merely a visual-specific hack;
- it aggressively removes noisy support while preserving a highly learnable
  contract target.

## Actor On Top Of Filtered Visual Data

Run root:

```text
runs_stage55_visual_endpoint_actor/20260616_150357
```

Filtered visual dataset: `ratio=0.5`.

Actor summary:

| env | rows | actor MSE mean | actor MSE median | actor MSE p90 |
| --- | ---: | ---: | ---: | ---: |
| `visual-antmaze-large-explore-v0` filtered `ratio=0.5` | 20,000 | 0.532 | 0.516 | 0.776 |

Scorer comparison on the same filtered dataset:

| scorer | test AP | test ROC-AUC | test Brier |
| --- | ---: | ---: | ---: |
| filtered base | 0.730 | 0.924 | 0.119 |
| filtered + actor raw | 0.747 | 0.929 | 0.115 |
| filtered + actor-conditioned labels | 0.731 | 0.945 | 0.098 |

Interpretation:

- once the graph density problem is addressed, actor features become useful;
- raw actor features give the best AP gain;
- actor-conditioned labels give the best calibration / ROC-AUC / Brier;
- this suggests a cleaner algorithm split:
  1. edge-relative support filtering to repair the candidate graph,
  2. actor-aware contract scoring on top of the repaired graph.

## Current Research Direction

The current mainline is no longer "support count plus weight sweep".

The stronger candidate algorithm is:

```text
BARS-CAP-Local =
  edge-relative offline support filtering
  + learned contract scorer
  + actor-aware contract refinement
  + planner-time soft risk patching
```

This is more coherent than the earlier Stage47 result because actor awareness is
helpful only after the candidate support neighborhood is repaired.

## Visual Rollout Report

Patched visual keygraphs for:

- filtered base
- filtered + actor raw
- filtered + actor-conditioned

were built under:

```text
runs_stage56_visual_filtered_eval_gpu3/20260616_150920
```

Longer 20-episode visual evals on one shared GPU were much slower than expected,
so a shorter detached smoke report was launched under:

```text
runs_stage56_visual_filtered_eval_gpu3/20260616_150920_smoke5_bg
```

Completed outputs:

- `runs_stage56_visual_filtered_eval_gpu3/20260616_150920_smoke5_bg/eval_csv/visual_base_w0p25_smoke5.csv`
- `runs_stage56_visual_filtered_eval_gpu3/20260616_150920_smoke5_bg/eval_csv/visual_actor_raw_w0p25_smoke5.csv`

Result:

| variant | success |
| --- | ---: |
| filtered base `w=0.25` | 0.00 |
| filtered actor raw `w=0.25` | 0.00 |

Interpretation:

- the offline scorer improvement on local visual data did not translate into
  closed-loop success on the weak local visual policy;
- this keeps visual useful as an offline stress test, but not yet as the best
  final reporting environment for graph-patch gains.

## Follow-On

Later humanoid weight-sweep results are summarized in:

- `docs/stage61_humanoid_weight_sweep_results.md`
