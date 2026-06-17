# Stage46 Parallel Research Workplan And Current Status

Last updated: 2026-06-16

## Current Training Snapshot

Stage44 humanoid/visual local GAS retraining is complete:

```text
runs_stage44_humanoid_visual_retrain/20260615_231837
artifacts/gas_ogbench_stage44_humanoid_visual_retrain_20260615_231837
GPU policy: completed on GPU2; new launches are now assigned to GPU3.
```

Final Stage44 eval summary:

| env | local GAS success | mean length | note |
| --- | ---: | ---: | --- |
| `humanoidmaze-large-navigate-v0` | 0.776 | 1457.732 | close to the official reference band |
| `humanoidmaze-large-stitch-v0` | 0.848 | 1357.028 | above the official reference mean |
| `visual-antmaze-large-explore-v0` | 0.040 | 996.580 | very weak local visual baseline |
| `visual-scene-play-v0` | 0.004 | 748.880 | very weak local visual baseline |

Stage44 resolves the missing-artifact issue. The visual results should be
treated as local retrain baselines, not official pretrained-weight results. They
are nevertheless useful stress tests because GAS performance is poor enough to
show room for method-level improvement.

Completed follow-up evaluation:

```text
runs_stage50_hybrid_eval_gpu3/20260616_140855
GPU policy: three Stage50 hybrid closed-loop eval jobs completed in parallel with GPU id 3.
```

This is evaluation only. It is not used to train labels, tune thresholds, or add
online data to the offline-RL method.

Result summary:

| env | Stage45 CAP-lite | Stage50 hybrid | conclusion |
| --- | ---: | ---: | --- |
| `antmaze-giant-navigate-v0` | 0.81 | 0.82 | small positive, not yet decisive |
| `antmaze-large-explore-v0` | 0.95 | 0.95 | unchanged |
| `scene-play-v0` | 0.76 | 0.73 | sequence penalty regresses Scene |

Detailed notes:

```text
docs/stage50_hybrid_eval_results.md
```

## Progress Started Without Waiting

Track A has moved from plan to executable artifact:

```text
scripts/stage45_build_offline_contract_dataset.py
```

The builder uses only fixed local OGBench `.npz` files, GAS keygraphs, and
precomputed dataset embeddings. It does not run online environment interaction.
It also aligns raw OGBench terminal rows to the GAS/GCDataset training protocol by
dropping raw terminal rows and marking the penultimate row as terminal, and
records that alignment in every summary.

Generated smoke/sample outputs:

| env | rows | episode split | edge positive support rate | labeled positive row rate | output |
| --- | ---: | --- | ---: | ---: | --- |
| `humanoidmaze-large-navigate-v0` | 20,000 | 800/100/100 | 0.154 | 0.082 | `runs_stage45_offline_contract_dataset/smoke_humanoid_large_navigate_seed0` |
| `antmaze-giant-navigate-v0` | 20,000 | 400/50/50 | 0.120 | 0.059 | `runs_stage45_offline_contract_dataset/antmaze_giant_navigate_seed0_sample` |
| `antmaze-large-explore-v0` | 20,000 | 8000/1000/1000 | 0.049 | 0.021 | `runs_stage45_offline_contract_dataset/antmaze_large_explore_seed0_sample` |
| `scene-play-v0` | 20,000 | 800/100/100 | 0.026 | 0.024 | `runs_stage45_offline_contract_dataset/scene_play_seed0_sample` |

The low positive-support rates, especially on `scene-play-v0`, reinforce the
current hypothesis: support-only routing is too weak as a complete algorithmic
principle. The next useful step is to train/calibrate an offline contract scorer
and compare it against support count and geometric distance.

First lightweight scorer diagnostic is also available:

```text
scripts/stage45_train_offline_contract_scorer.py
runs_stage45_offline_contract_dataset/combined_antmaze_scene_logistic_scorer
```

It trains only on fixed offline contract rows from:

- `antmaze-giant-navigate-v0`;
- `antmaze-large-explore-v0`;
- `scene-play-v0`.

Current combined logistic results:

| split | AP | ROC-AUC | Brier | strongest simple AP baseline |
| --- | ---: | ---: | ---: | ---: |
| train | 0.515 | 0.983 | 0.048 | 0.297 (`same_traj_support`) |
| val | 0.509 | 0.961 | 0.076 | 0.342 (`same_traj_support`) |
| test | 0.569 | 0.963 | 0.077 | 0.383 (`same_traj_support`) |

This is not yet a final algorithm result because the labels are still local
offline contract labels, not environment success. The useful conclusion is more
limited but important: a small contract model extracts signal beyond a single
support count or distance baseline, so BARS-CAP-lite was worth implementing.

## Latest Stage45/47 Findings

Detailed notes are in `docs/stage47_actor_contract_results.md`.

Stage45 global CAP-lite results:

| env | original | support-only reference | CAP-lite global | conclusion |
| --- | ---: | ---: | ---: | --- |
| `antmaze-giant-navigate-v0` | 0.76 | 0.83 | 0.81 | improves original, below best support-only |
| `antmaze-large-explore-v0` | 0.94 | 0.98 | 0.95 | small gain, below support-only |
| `scene-play-v0` | 0.73 | 0.70 | 0.76 | fixes the support-only regression |

Stage45 path-local CAP-lite was negative:

| variant | success |
| --- | ---: |
| `giant_pathlocal_aggressive` | 0.79 |
| `giant_pathlocal_conservative` | 0.77 |
| `large_pathlocal` | 0.93 |
| `scene_pathlocal` | 0.73 |

Stage47 added frozen-actor agreement features. It produced useful diagnostics
but only a small offline scorer gain:

| scorer | val AP | test AP | val ROC-AUC | test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Stage45 contract scorer | 0.509 | 0.569 | 0.961 | 0.963 |
| Stage47 actor-aware scorer | 0.515 | 0.572 | 0.960 | 0.963 |

Actor-aware edge scores also produced almost the same path changes as Stage45:

| env | Stage45 path change rate | Stage47 actor-aware path change rate |
| --- | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.5442 | 0.5444 |
| `antmaze-large-explore-v0` | 0.7946 | 0.7946 |
| `scene-play-v0` | 0.1409 | 0.1409 |

Current conclusion:

```text
The promising direction is global soft contract risk, not path-local replacement.
Raw frozen-actor action agreement is diagnostic, but it does not materially
change graph decisions under the current local-label objective. The next
research step should change the contract target/label, not just add another
feature or sweep another graph weight.
```

Stage48 implemented the first actor-conditioned label transform:

```text
scripts/stage48_build_actor_conditioned_contract_labels.py
runs_stage48_actor_conditioned_labels/20260616_063500
```

It demotes original positive rows with high frozen-actor MSE into hard negatives.
This changes roughly one quarter of the original positive rows:

| env | base positive rate | actor-conditioned positive rate | demoted positive rate |
| --- | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.0587 | 0.0435 | 0.259 |
| `antmaze-large-explore-v0` | 0.0207 | 0.0158 | 0.236 |
| `scene-play-v0` | 0.0243 | 0.0180 | 0.259 |

However, graph decisions still barely move:

| env | Stage47 path change rate | Stage48 path change rate |
| --- | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.5444 | 0.5444 |
| `antmaze-large-explore-v0` | 0.7946 | 0.7946 |
| `scene-play-v0` | 0.1409 | 0.1436 |

Stage48 conclusion:

```text
Single-step actor MSE is a useful offline diagnostic, but not a strong enough
contract target. The next step should be sequence-level actor-conditioned
reachability and an edge-ranking objective that directly affects graph ordering.
```

Stage49/50 advanced that plan:

```text
Stage49: sequence-level actor-conditioned labels and scorer.
Stage50: hybrid broad-prior + sequence-verifier edge scores.
```

The Stage49 sequence label is much stronger than the Stage48 single-step proxy:

| env | base positive rate | sequence positive rate | demoted positives |
| --- | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.0587 | 0.0300 | 48.8% |
| `antmaze-large-explore-v0` | 0.0207 | 0.0108 | 47.8% |
| `scene-play-v0` | 0.0243 | 0.0141 | 42.1% |

The Stage49 scorer gives strong offline sequence-contract metrics:

| split | AP | ROC-AUC | edge AP | edge ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| val | 0.590 | 0.989 | 0.524 | 0.980 |
| test | 0.644 | 0.987 | 0.660 | 0.982 |

Direct Stage49 graph replacement is too sparse. Stage50 instead preserves the
Stage45 broad reachability prior and uses Stage49 only as an actor verifier where
sequence evidence exists. Path audit shows this is more stable:

| env | Stage45 path change | Stage49 direct | Stage50 hybrid |
| --- | ---: | ---: | ---: |
| `antmaze-giant-navigate-v0` | 0.5442 | 0.4699 | 0.5514 |
| `antmaze-large-explore-v0` | 0.7946 | 0.7927 | 0.7946 |
| `scene-play-v0` | 0.1409 | 0.1540 | 0.1415 |

Current algorithm shape:

```text
BARS-CAP-Seq =
  broad offline reachability prior
  + sequence-level frozen-actor verifier
  + soft graph-risk composition
  + strictly offline contract learning.
```

The Stage50 hybrid graph has been evaluated with the same 20-episode closed-loop
protocol used by Stage45:

| job | env | graph variant |
| --- | --- | --- |
| `gn_hybrid_w0p25` | `antmaze-giant-navigate-v0` | Stage50 hybrid `w0.25` |
| `large_explore_hybrid_w0p25` | `antmaze-large-explore-v0` | Stage50 hybrid `w0.25` |
| `scene_hybrid_w2` | `scene-play-v0` | Stage50 hybrid `w2` |

Run root:

```text
runs_stage50_hybrid_eval_gpu3/20260616_140855
```

The result is mixed: giant slightly improves over Stage45, large is unchanged,
and Scene regresses. This points to the next algorithmic question: low sequence
probability is not yet calibrated enough to be used as a direct negative planner
penalty across environment families.

## Principle

This project is offline RL. All trainable components for the main method must
use only fixed offline datasets and labels/features derived from those datasets.
Evaluation rollouts are for reporting and diagnosis only; they must not become
training data or threshold-tuning feedback for the main claim.

## What Can Start Immediately

### Track A: Offline Contract Dataset Design

This is the highest-priority parallel task. It does not depend on Stage44.

Goal:

```text
Build D_contract from fixed offline OGBench data and existing graph artifacts.
```

Inputs available now:

- existing official GAS AntMaze/Scene artifacts;
- BARS support edge scores and path audits;
- local OGBench datasets under `/mnt/project/offlinerl_datasets/ogbench`;
- existing graph export utilities under `bars/gas_bars/`.

Immediate deliverables:

1. Define row schema for offline contract examples.
2. Implement dataset builder for GAS-latent graph first.
3. Add hard-negative sampling:
   - reversed edges;
   - cross-trajectory near targets;
   - planner-proposed unsupported alternatives;
   - high actor-disagreement pairs computed on offline states.
4. Produce small sample files on:
   - `antmaze-giant-navigate-v0`;
   - `antmaze-large-explore-v0`;
   - `scene-play-v0`.

Success criterion:

The builder produces reproducible train/val/test episode splits and reports
positive/negative balance, support counts, temporal distance, and actor-feature
coverage.

### Track B: BARS-CAP-lite On Existing GAS Bridge

This has now produced a useful but incomplete result. It should remain as a
bridge baseline/ablation, not the final algorithm.

Goal:

```text
Replace support-only path selection with offline contract-aware path selection.
```

Immediate deliverables:

1. Keep the lightweight contract scorer on fixed offline data only.
2. Add contract score to GAS edge/path costs.
3. Evaluate only predeclared variants on the Stage43 primary set:
   - `antmaze-giant-navigate-v0`;
   - `antmaze-large-explore-v0`;
   - `antmaze-giant-stitch-v0`;
   - `scene-play-v0`.

Success criterion:

- Preserve at least two existing AntMaze gains.
- Avoid the Stage42 Scene regression.
- Show contract calibration explains success/failure better than support count
  alone.

### Track C: BARS-native Policy V2 Design

This should start as design and data audit before launching new GPU training.

Goal:

```text
Move from raw planner GCBC/source heads toward offline planner-grounded option
execution.
```

Immediate deliverables:

1. Audit Phase5N/5P target-family datasets.
2. Decide whether V2 should use:
   - separate direct/option policies;
   - option-conditioned policy with edge embedding;
   - learned skill-space target representation.
3. Define validation groups:
   - final-goal;
   - support-edge;
   - planner-issued subgoal;
   - low-contract edge;
   - long-horizon edge.

Success criterion:

Before training, the design must show how every training example is derived
from fixed offline data and how runtime planner targets are represented in
training.

### Track D: Evaluation/Inventory Automation

This engineering work now supports the completed Stage44 artifact set and the
new Stage50 GPU3 evaluation.

Immediate deliverables:

1. Stage44 artifact inventory/report:
   - checkpoint exists;
   - keygraph exists;
   - flags exist;
   - dataset embeddings exist;
   - manifest exists;
   - eval CSV exists.
2. Summary table generator for:
   - original GAS baseline;
   - BARS support-only;
   - BARS-CAP-lite;
   - final BARS-native method.

Success criterion:

Results can be audited in minutes rather than by manual file inspection.

## Humanoid/Visual Expansion

Stage44 complete artifacts are now available. The next step is no longer waiting;
it is deciding which of these environments become development stress tests versus
claim-set environments.

Start now:

- build offline contract rows for `humanoidmaze-large-navigate-v0`;
- build offline contract rows for `visual-antmaze-large-explore-v0` if visual
  embedding/state alignment is clean;
- audit local retrain quality against official expected scores;
- keep visual results labeled as local retrain baselines.

Still wait:

- any Humanoid/Visual main-table claim;
- multi-seed expansion beyond seed 0 for these families;
- claims that compare visual local retraining directly to official pretrained
  weights.

Current Stage52 contract-dataset expansion:

```text
runs_stage52_stage44_contract_datasets/20260616_142400
```

| env | status | key diagnostic |
| --- | --- | --- |
| `humanoidmaze-large-navigate-v0` | complete | 20k rows; positive labeled row rate `0.197`; edge positive support rate `0.381` |
| `visual-antmaze-large-explore-v0` | running CPU-only | graph tables written; contract rows still building |

## Work Queue

| priority | task | dependency | start now? |
| --- | --- | --- | --- |
| P0 | Offline contract dataset schema and builder | existing AntMaze/Scene data | yes |
| P0 | BARS-CAP-lite contract-aware path reranking | existing GAS bridge artifacts | yes |
| P0 | Stage44 artifact summary and Humanoid/Visual contract inventory | completed Stage44 run | yes |
| P1 | BARS-native policy V2 design | Phase5N/5P artifacts | yes |
| P1 | Scene actor-compatibility failure analysis | Stage41/42 outputs | yes |
| P2 | Humanoid/Visual BARS variants | Stage44 artifact audit | partial |
| P2 | multi-seed Humanoid/Visual claims | baseline/inventory pass | not yet |

## Proposed Immediate Sequence

### Day 0: No-wait actions

1. Write offline contract dataset schema.
2. Implement first builder on existing GAS latent artifacts.
3. Generate sample contract tables for one AntMaze and one Scene environment.
4. Train a simple logistic/MLP scorer or at least run calibration diagnostics.

### Day 1: BARS-CAP-lite

1. Patch path selection with contract-aware cost.
2. Run small episode-count A/B on:
   - `antmaze-giant-navigate-v0`;
   - `scene-play-v0`.
3. Decide whether compatibility signal fixes the Scene support-only regression.

### After Stage44 Completion

1. Run inventory on Stage44 artifacts.
2. Treat the completed local GAS evals as baseline repair evidence.
3. Decide whether Humanoid/Visual enter the development set or claim set.

## Current Research Plan Summary

The immediate research plan is:

```text
Do not wait for Stage50 eval results.
Use existing AntMaze/Scene artifacts to build and test offline actor-compatible
contracts.
Use Stage44 outputs to expand the stress-test environment set.
```

The next algorithmic milestone is no longer another support-weight or path-local
sweep. It is an actor-conditioned contract target that changes edge reliability
labels enough to affect graph decisions while staying strictly offline.

## Immediate Next Actions

1. Treat Stage50 hybrid as the current main branch for offline graph-risk
   development, but keep Stage45 broad prior as the stable bridge baseline.
2. Add an edge-ranking/calibration objective next:
   - broad prior should preserve coverage;
   - sequence verifier should only penalize edges when held-out offline evidence
     separates true actor-incompatibility from false negatives;
   - avoid direct global replacement by sparse sequence probabilities.
3. Keep global soft-risk graph patching as the short-term evaluation bridge.
4. Stop expanding path-local reranking until the label/objective changes again.
5. Run the same offline contract pipeline on representative Humanoid/Visual
   artifacts, starting with the completed local Stage44 baselines.
