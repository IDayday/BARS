# BARS Algorithm Improvement Attempt Log

Last updated: 2026-06-14

This document is the running ledger for BARS algorithm improvement attempts. It
records what was tried, what the evidence says, what remains unproven, and what
must be reviewed before starting another algorithmic direction.

The goal is to reduce repeated trial-and-error. Future work should review this
document first, then review relevant papers and open-source implementations
before adding new experiments. Experimental conclusions are only useful when the
algorithm design, implementation, and metric semantics are mature enough to
support them.

## Ground Rules

- Do not claim online success from offline graph, compatibility, or supervised
  action-MSE metrics.
- Do not treat kNN, proximity, latent-threshold, or random graph edges as
  executable unless they are support-certified or policy/rollout validated.
- Do not implement or compare TDR/TMD/MQE-style claims without first reading the
  corresponding papers and available official or widely used code.
- Keep environment availability separate from reset capability. Current
  `env_unavailable` results are dependency blockers, not evidence that AntMaze
  or Scene lack reset-to-state support.
- Every new algorithm attempt must produce a short design note, scoped tests,
  result artifacts, and an update to this log.

## Completed Attempts

### Phase 2: Support-Only Compressed Option Graph

Question:

Can we construct a compressed directed option graph using only offline
trajectory support, without learned policies or environment rollout?

Implemented:

- Cluster/support/bottleneck diagnostics from Phase 1 were reused.
- Nodes were selected by `density`, `bottleneck`, `core_plus_bottleneck`, and
  `all`.
- Edges were included only when real trajectory segments supported
  `src -> dst` within horizon `H`.
- The graph reports edge support scale, unique starts, episodes, costs,
  coverage, bottleneck removal utility, and compatibility.

Evidence:

- AntMaze and Scene sweeps completed across multiple `H`, budgets, and node
  selections.
- `all` upper bound gives high graph coverage, showing the offline support graph
  contains broad connectivity.
- Compressed methods expose a coverage, edge-scale, bottleneck-sensitivity, and
  compatibility trade-off.
- kNN/random baselines can have high path coverage while relying on unsupported
  edges.

Analysis:

Phase 2 establishes a trustworthy graph-layer baseline: if an edge exists, it
has observed trajectory support. This directly targets the failure mode where
proximity or latent distance creates unsupported shortcuts.

Remaining gap:

The graph still does not prove low-level executability. A data-supported
segment means the behavior policy did it in the dataset, not that a learned
option policy can reproduce it from arbitrary starts.

### Phase 2.2: Metrics Cleanup

Question:

Which graph metrics were easy to misread, and how should they be corrected
before feeding Phase 3?

Implemented:

- Separated strict query selection rate from strict coverage over all queries.
- Added reachable-only virtual edge usage metrics.
- Split reverse support into raw and certified support.
- Replaced compatibility support interpretation with termination bridge
  coverage as the probability-like main metric.
- Changed bottleneck removal cost delta to common-reachable query cost delta.

Evidence:

- Synthetic tests pass for metric edge cases.
- Sweep summaries now preserve both all-query and reachable-only path usage.

Analysis:

This phase reduced metric ambiguity. In particular, match-count compatibility
can exceed a probability interpretation, while termination bridge coverage stays
in `[0, 1]` and is safer for aggregate analysis.

Remaining gap:

Even cleaned compatibility is still cluster/segment evidence. It is not a
closed-loop execution guarantee.

### Phase 3A: GCBC Supervised Edge Fitting

Question:

Can a state-based goal-conditioned BC policy fit held-out one-step actions on
Phase 2 support-certified edge segments?

Implemented:

- Built edge BC samples from option edge segments.
- Trained GCBC on observed transitions with goals set to segment termination
  observations.
- Reported train/validation MSE and edge/grouped validation metrics.

Evidence:

- Existing 100000-step AntMaze run has final `val_action_mse = 0.0426389`.
- Grouped metrics are written under
  `results/phase3/antmaze_large_stitch/core_plus_bottleneck/`.

Analysis:

This shows the GCBC model can fit held-out edge BC samples in the offline
supervised sense. It gives a useful policy-fitting proxy for edge risk.

Remaining gap:

Offline action MSE is not edge execution success. It does not measure compounding
closed-loop error, reset sensitivity, or whether the final state enters the
destination cluster.

### Phase 3B/3C: Environment Preflight and Reset Probe

Question:

Can the current Python environment construct OGBench/Gym environments and probe
reset-to-state support?

Implemented:

- Added environment preflight.
- Added reset probe statuses: `env_unavailable`, `reset_supported`,
  `reset_unsupported`, and `reset_uncertain`.
- Rollout evaluation skips safely when env construction is unavailable.

Evidence:

- The original active environment reported `env_unavailable` due missing
  `gymnasium`/`gym`/MuJoCo dependencies.
- The `gcrlo` conda environment now constructs both
  `antmaze-large-stitch-v0` and `scene-play-v0` through local OGBench
  (`external_src/tmd-release`) with `PYTHONPATH` set.
- `gcrlo` preflight status is `env_available` for both datasets, with
  `reset_probe_status: reset_unsupported`.
- Natural-start `env.reset(seed=0)` and one-step `env.step(action)` smoke tests
  succeed for both AntMaze and Scene.

Analysis:

The blocker has moved from environment construction to arbitrary reset
semantics. This should not be interpreted as AntMaze/Scene lacking online
closed-loop evaluation support. It means edge-level reset-to-state rollout
needs exact simulator state references, while natural-start task rollout is now
the right next evaluation path.

Remaining gap:

Phase 3C edge-level arbitrary reset execution is still blocked by missing exact
state refs (`qpos/qvel`, and Scene button state). Natural-start closed-loop task
rollout should be implemented next in `gcrlo`.

### Phase 3D: Offline Sampling Ablation

Question:

Do `uniform_transition`, `uniform_edge`, or `bottleneck_weighted` sampling
improve supervised fitting on bottleneck, low-support, or long-horizon edges?

Implemented:

- Added reset-free sampling ablation script and configs.
- Compared validation MSE groups by sampling strategy.

Evidence:

- Current checked-in configs are smoke-scale: `num_steps: 200`, `seeds: [0]`.
- At smoke scale, `uniform_transition` has the best final validation MSE on the
  AntMaze core-plus-bottleneck run and on both Scene runs.
- On AntMaze density, `bottleneck_weighted` slightly improves some grouped MSEs
  relative to `uniform_edge`.

Analysis:

The current ablation is useful as an implementation smoke test and a weak
directional signal. It is not strong enough for algorithmic claims because it is
short-horizon in optimization and single-seed.

Remaining gap:

Rerun with longer training, more seeds, and identical validation splits before
using sampling choice as a research conclusion.

### Phase 3E: Reset-Free Offline Edge Certification and GAS Graph Audit

Question:

Can reset-free offline evidence identify high-risk option edges and audit
whether proximity/GAS-style graphs overestimate connectivity?

Implemented:

- Heldout episode support split.
- Edge-level GCBC action-fitting proxy.
- Simple OOD/support risk proxy.
- Compatibility context aggregation.
- Offline edge proxy score and binary certification.
- GAS-style diagnostic approximation audit against support, kNN, and random
  graphs.

Evidence:

- AntMaze `core_plus_bottleneck_budget120_H10`: 40 / 582 edges certified offline
  at the current threshold, rate `0.068729`.
- Scene `core_plus_bottleneck_budget192_H5`: 209 / 1897 edges certified offline
  at the current threshold, rate `0.110174`.
- In both datasets, random graphs show highest path coverage and strongest
  unsupported shortcut reliance, while support graphs have the lowest
  unsupported edge rate.

Analysis:

This supports the central graph-layer critique: high path connectivity can be
created by unsupported shortcuts. Support-certified graphs reduce this risk by
construction, and offline certification provides a reset-free edge risk filter.

Remaining gap:

The certification score is a proxy composed of offline terms. It is not a
rollout success probability and should not be calibrated as one without
closed-loop validation.

### Phase 3F: Natural-Start Rollout Scaffold

Question:

If arbitrary reset remains unavailable but env construction works later, can we
evaluate from natural starts?

Implemented:

- Added scaffold for natural-start hierarchical rollout.
- It skips safely under `env_unavailable`.

Evidence:

- Current AntMaze and Scene scaffold outputs are skipped due env construction
  blockers.

Analysis:

This is useful infrastructure, not an experimental result.

Remaining gap:

Needs env availability, task definitions, and robust runtime planning before it
can support claims.

### Phase 4A: Risk-Aware Offline Support Planner

Question:

Can Phase 3E edge certification improve path selection inside the Phase 2
support graph without adding unsupported kNN/proximity/random edges?

Reviewed before implementation:

- GAS paper and official code path for keygraph construction and shortest-path
  planning.
- Search on the Replay Buffer and Google Research SoRB code.
- Test-Time Graph Search paper and code.
- The local Phase 3E graph audit and offline certification outputs.

Implemented:

- `support_shortest_path` baseline over Phase 2 support edges.
- `certified_only` hard filter.
- `proxy_threshold` hard proxy/support lower-bound filter.
- `proxy_penalized` scalarized risk-aware cost over all support edges.
- Synthetic tests for low-proxy shortcut avoidance, hard-filter coverage loss,
  and reachable-path risk metrics.

Evidence:

- AntMaze `core_plus_bottleneck_budget120_H10`: `proxy_penalized` preserves
  baseline coverage 0.566 while improving mean minimum edge proxy from 0.060 to
  0.128 and reducing uncertified edge fraction from 0.924 to 0.874.
- Scene `core_plus_bottleneck_budget192_H5`: `proxy_penalized` preserves
  baseline coverage 0.160 while improving mean minimum edge proxy from 0.065 to
  0.104 and reducing uncertified edge fraction from 0.984 to 0.855.
- Hard certification is too sparse for AntMaze at the current threshold and
  collapses coverage to 0.000.

Analysis:

This supports soft risk-penalized planning as the next default offline planner.
Hard filtering is useful as a diagnostic but can destroy connectivity. The
planner should continue to use Phase 2 support as a hard provenance boundary,
then use Phase 3E risk scores to choose safer paths among supported edges.

Remaining gap:

The result is still reset-free offline path selection. It does not prove that
the GCBC policy can execute the selected path, and the edge proxy score is not
calibrated to rollout success probability.

### Phase 4B: Calibrated Risk-Aware Planner Sweep

Question:

Can sweeping risk weights and mild edge floors find a better coverage/risk/cost
trade-off than the fixed Phase 4A planner?

Reviewed before implementation:

- GAS graph construction and shortest-path code.
- Replay-buffer graph search methods such as SoRB and TTGS.
- Constrained and multi-objective shortest-path framing, motivating Pareto
  reporting rather than a single unqualified scalar metric.

Implemented:

- `floor_proxy_penalized`: support-only planning with optional proxy/support-LCB
  floors plus risk-aware edge cost.
- 480-config sweeps for AntMaze and Scene.
- Pareto-front extraction over coverage, minimum edge proxy, uncertified edge
  fraction, and base path cost.
- Recommendation heuristic constrained against support-shortest-path coverage
  and base cost.

Evidence:

- AntMaze recommended config keeps coverage at 0.544 versus baseline 0.566,
  raises mean minimum edge proxy from 0.060 to 0.217, and lowers uncertified
  edge fraction from 0.924 to 0.826.
- Scene recommended config keeps coverage at 0.160, raises mean minimum edge
  proxy from 0.065 to 0.092, and lowers uncertified edge fraction from 0.984 to
  0.733.
- AntMaze has many strong Pareto choices, including full-coverage configs with
  mean minimum edge proxy around 0.19. Scene has a harsher compatibility and
  certification bottleneck.

Analysis:

The useful planner pattern is not simply increasing aggregate `risk_weight`.
The recommended configs emphasize decomposed OOD and uncertified penalties, and
AntMaze also benefits from a mild proxy floor. This suggests the aggregate
Phase 3E proxy score is useful for ranking but is not yet calibrated enough to
serve directly as an execution probability.

Remaining gap:

This remains offline path selection. The next algorithmic bottleneck is edge
risk calibration, especially compatibility/certification reliability on Scene,
not another blind planner-weight sweep.

### Phase 4C: Edge Risk Calibration

Question:

Can separating heldout support, policy fit, OOD, compatibility, and diversity
signals produce a better planner-facing edge reliability score than the Phase 3E
aggregate proxy?

Reviewed before implementation:

- Reliability calibration for confidence scores.
- Offline RL behavior-support and pessimism motivation.
- GAS, SoRB, and TTGS graph-search code paths where local edge confidence drives
  global path quality.

Implemented:

- Component-wise reliability scores for support, policy, behavior, compatibility,
  and diversity.
- Conservative weighted geometric aggregation into
  `calibrated_edge_reliability_score`.
- Planner-compatible calibrated certification table preserving original
  certification columns.
- Phase 4B-style sweep over calibrated scores for AntMaze and Scene.
- Synthetic tests for conservative component aggregation, planner input
  replacement, and pseudo-label diagnostics.

Evidence:

- AntMaze Brier score against heldout-support pseudo-label improves from 0.490
  to 0.388. Recommended calibrated planning restores coverage to 0.566 and
  lowers original uncertified path-edge fraction from Phase 4B's 0.826 to 0.013.
- Scene Brier score improves from 0.404 to 0.350. Recommended calibrated
  planning keeps coverage at 0.160 and lowers original uncertified path-edge
  fraction from 0.733 to 0.156.
- Scene high-incompatibility exposure remains high, so compatibility is still a
  separate bottleneck.

Analysis:

Phase 4C is the strongest offline graph-layer improvement so far. The calibrated
score does not prove executability, but it makes support-only planner paths much
cleaner under original Phase 3E certification evidence. The remaining failure
mode is path composability: independent edge reliability does not fully handle
adjacent-edge mismatch, especially in Scene.

Remaining gap:

The calibrated score is not an execution probability. The next offline
algorithmic target should be compatibility-aware path planning that models
adjacent edge pairs, not only independent edge scores.

### Phase 4D: Compatibility-Aware Planning

Question:

Can transition-dependent adjacent-edge compatibility costs reduce option-path
composability risk beyond independent calibrated edge reliability?

Reviewed before implementation:

- GAS-style graph planning and the local `external_src/GAS` graph-search code.
- Replay-buffer graph search methods such as SoRB and TTGS.
- Phase 2.2 compatibility metric semantics, especially
  `termination_bridge_coverage` as the probability-like bridge metric.
- Phase 4C calibrated edge certification outputs.

Implemented:

- Line-graph Dijkstra where search states are option edges.
- Pair-penalized planning with cost
  `edge_cost + pair_weight * (1 - termination_bridge_coverage)`.
- Pair-threshold planning that rejects adjacent option transitions below a
  bridge coverage floor.
- Comparisons against support shortest path and Phase 4C calibrated edge
  penalty on identical path queries.
- Synthetic tests for pair-penalty route selection, pair-threshold blocking, and
  recomputing pair compatibility from `edge_segments.npz`.

Evidence:

- AntMaze `core_plus_bottleneck_budget120_H10`: `calibrated_compat_penalized`
  preserves coverage at 0.566 while reducing pair incompatible fraction from
  0.161 to 0.033 and improving mean minimum pair bridge coverage from 0.043 to
  0.113.
- AntMaze `calibrated_compat_threshold` removes incompatible adjacent pairs
  under the configured bridge floor, with coverage 0.544 and mean minimum pair
  bridge coverage 0.146.
- Scene `core_plus_bottleneck_budget192_H5`: support shortest paths have pair
  incompatible fraction 0.906. Threshold methods reduce that to 0.000, but
  coverage drops from 0.160 to 0.150 and base cost roughly doubles.
- Scene pair diagnostics show a strict-compatible rate of only 0.337 and median
  termination bridge coverage 0.000, explaining why compatibility-aware
  planning is costly there.

Analysis:

Phase 4D confirms that independent edge reliability is not enough. Path
composition must account for adjacent-edge bridge evidence. AntMaze has enough
alternative supported routes for soft compatibility penalties to improve path
risk without losing coverage. Scene has a structural composability problem:
clean paths exist for some queries, but enforcing compatibility sharply
increases path length and cost.

Remaining gap:

This is still offline graph-layer evidence. It does not prove GCBC execution,
and Scene likely needs structural graph repair or compatibility-aware node/edge
selection before closed-loop evaluation would be meaningful.

### Phase 4E: Compatibility Graph Repair

Question:

Can a compressed support graph recover compatibility-safe coverage by adding
only support-certified repair edges from a broader Phase 2 support bank?

Reviewed before implementation:

- GAS-style graph planning and local `external_src/GAS` graph-search code.
- Replay-buffer graph search methods such as SoRB and TTGS.
- Phase 2.2 termination bridge coverage and Phase 4D line-graph planning.
- Phase 4D evidence that Scene's strict compatibility threshold sharply reduced
  coverage.

Implemented:

- Support-only repair edge selection from an `all` Phase 2 edge bank.
- Bad-junction scoring from low `termination_bridge_coverage` adjacent pairs.
- Repair edge scoring using bad-junction incidence, base-graph connectivity,
  support scale, start/episode diversity, and short horizon.
- Segment-level merge and edge-id remapping for selected repair edges.
- Recomputed augmented pair compatibility and reran Phase 4D planners on the
  same queries.
- Synthetic tests for duplicate avoidance, edge-id/segment remapping, and
  threshold coverage restoration.

Evidence:

- AntMaze repair adds 200 support-bank edges and improves strict
  `compat_threshold` coverage from 0.544 to 0.620 while keeping pair
  incompatible fraction at 0.000.
- AntMaze repaired support shortest-path coverage improves from 0.566 to 0.642.
- Scene repair adds 500 support-bank edges and improves strict
  `compat_threshold` coverage from 0.150 to 0.480 while keeping pair
  incompatible fraction at 0.000.
- Scene repaired support shortest-path coverage improves from 0.160 to 0.510,
  showing that the broader support bank contains useful connectivity omitted by
  the compressed graph.
- Scene H10 repair adds 500 support-bank edges and improves repaired support
  shortest-path coverage from 0.160 to 0.550. Strict `compat_threshold`
  coverage improves from 0.160 to 0.510 while keeping pair incompatible
  fraction at 0.000.

Analysis:

This is the strongest structural graph-layer result so far. Scene's poor
compatibility-safe coverage was not simply absence of data support; it was a
compression/selection problem. Adding only support-certified repair edges
recovers many compatible paths without proximity shortcuts. However, the added
edges are not yet Phase 3E/4C certified, so repaired paths have lower edge proxy
scores and higher original uncertified fractions.

Remaining gap:

Repair-bank edges need offline certification and calibration before they can be
used as planner-default edges. Phase 4E improves graph structure, but it does
not prove policy execution or online benchmark gains.

### Phase 4F: Repair-Edge Certification and Joint Planning

Question:

Can conservative offline certification for selected repair-bank edges preserve
Phase 4E's coverage gains while restoring planner-facing edge reliability?

Reviewed before implementation:

- Offline RL behavior-support and pessimism motivation such as BCQ/CQL.
- Replay-buffer graph search methods such as SoRB and TTGS.
- Phase 4C component-wise edge calibration.
- Phase 4E support-only repair graph outputs.

Implemented:

- Base edges retain Phase 4C certification.
- Repair edges receive `repair_transfer_proxy` certification from support
  scale, endpoint-neighbor policy-score transfer, behavior support,
  compatibility context, and conservative geometric aggregation.
- Combined base+repair planner certification table.
- Re-evaluation of Phase 4E augmented graphs with repair-edge-aware
  compatibility planners.
- Path metrics for repair edge fraction and repair certified fraction.
- Synthetic tests for certification provenance, score ordering, and repair-edge
  path usage.

Evidence:

- AntMaze: 182 / 200 repair edges are transfer-certified. On the repaired graph,
  `calibrated_compat_threshold` keeps coverage at 0.620, keeps pair
  incompatible fraction at 0.000, improves mean minimum edge proxy from Phase
  4E's 0.229 to 0.277, and lowers current uncertified edge fraction from 0.059
  to 0.014.
- Scene: 397 / 500 repair edges are transfer-certified. On the repaired graph,
  `calibrated_compat_threshold` keeps coverage at 0.480, keeps pair
  incompatible fraction at 0.000, improves mean minimum edge proxy from 0.072 to
  0.230, and lowers current uncertified edge fraction from 0.250 to 0.044.

Analysis:

Phase 4F closes the immediate Phase 4E reliability gap. The repaired graph no
longer needs to choose between compatibility-safe coverage and planner-facing
edge reliability. Scene remains the most important result: compatibility-safe
coverage stays far above the compressed base graph while uncertified exposure is
much lower under transfer certification.

Remaining gap:

Repair-edge certification is still a transfer proxy. It is not direct heldout
GCBC likelihood and not rollout success. The next step should compute direct
policy-likelihood / action-fitting evidence for selected repair edges and then
run closed-loop evaluation once environment dependencies are available.

### Phase 4G: Direct Repair-Edge Policy Evidence

Question:

Does direct GCBC action fitting on selected repair-bank segments support or
contradict Phase 4F's transfer-proxy repair-edge certification?

Reviewed before implementation:

- BCQ/CQL-style offline RL behavior-support and pessimism motivation.
- Replay-buffer graph search methods such as SoRB and TTGS.
- Phase 3 GCBC edge fitting and Phase 3E policy-likelihood utilities.
- Phase 4F repair-transfer certification outputs.

Implemented:

- Mapping from Phase 4E augmented repair edge ids back to support-bank edge ids.
- Direct GCBC action-MSE evaluation on all selected repair-bank edge segments.
- Conversion of direct MSE to policy support scores.
- Replacement of repair-edge transfer policy scores with direct policy scores.
- Recalibration and repaired-graph planner re-evaluation.
- Synthetic tests for segment selection, direct-score replacement,
  transfer-vs-direct diagnostics, and repair-edge path usage.

Evidence:

- AntMaze uses the 100000-step `core_plus_bottleneck` GCBC model. All 200
  repair edges are direct-scored. Mean direct repair-edge MSE is 0.0554, direct
  certified rate is 0.905, and `calibrated_compat_threshold` keeps coverage at
  0.620 with pair incompatible fraction 0.000 and uncertified fraction 0.014.
- AntMaze transfer-to-direct policy score delta is -0.134 and Spearman
  transfer-vs-direct policy score is only 0.055, showing that the transfer proxy
  is not a precise repair-edge ranker when direct policy evidence is available.
- Scene uses the available 200-step smoke GCBC model. All 500 repair edges are
  direct-scored. Mean direct repair-edge MSE is 0.0382, direct certified rate is
  0.870, and `calibrated_compat_threshold` keeps coverage at 0.480 with pair
  incompatible fraction 0.000 and uncertified fraction 0.037.

Analysis:

Phase 4G strengthens the repair pipeline by adding direct supervised policy
evidence. The main algorithmic finding survives: support-only graph repair plus
compatibility-aware planning recovers much higher compatibility-safe coverage,
especially on Scene. Direct policy evidence also shows that repair-transfer
scores should be treated as coarse fallback estimates, not precise ranking
scores.

Remaining gap:

Direct action MSE is not closed-loop execution. Scene's direct evidence is also
limited by the short 200-step smoke model. Phase 4H addresses this for one
stronger Scene model; closed-loop edge execution still requires environment
dependencies to be unblocked.

### Phase 4H: Stronger Scene GCBC Direct Repair Validation

Question:

Does Scene's Phase 4G direct repair-edge evidence survive a GCBC model trained
substantially longer than the 200-step smoke run?

Reviewed before implementation:

- Goal-Conditioned Supervised Learning (GCSL) and its reference implementation.
- RvS-style offline reinforcement learning via supervised learning.
- Phase 3 GCBC training and Phase 4G direct repair-edge policy-evidence code.
- The Phase 4G Scene limitation that the available direct evidence came from a
  200-step smoke model.

Implemented:

- A 10000-step `uniform_transition` Scene GCBC config for
  `core_plus_bottleneck_budget192_H5`.
- A Phase 4H driver script that trains GCBC, reruns Phase 4G direct repair-edge
  evidence, and writes candidate-minus-baseline deltas against the 200-step
  Scene baseline.
- A summary module and synthetic tests for diagnostic and planner-delta
  reporting.

Evidence:

- The 10000-step Scene GCBC reaches final validation action MSE `0.005127`,
  compared with `0.024553` for the previous 200-step smoke run.
- Mean direct repair-edge action MSE improves from `0.038238` to `0.011464`.
- Mean direct policy support score improves from `0.561399` to `0.833299`.
- Direct repair-edge certified rate improves from `0.870` to `0.894`.
- `calibrated_compat_threshold` coverage remains `0.480`, pair incompatible
  fraction remains `0.000`, and uncertified edge fraction improves from
  `0.037335` to `0.033169`.

Analysis:

Phase 4H addresses the main weakness in the Scene Phase 4G evidence. The Scene
repair-edge direct policy proxy remains positive under a much stronger
supervised model, and planner risk metrics improve slightly without changing
coverage. This supports the offline conclusion that the repaired Scene graph is
not merely relying on repair edges that the GCBC cannot fit in the supervised
sense.

Remaining gap:

This is still reset-free offline supervised evidence. It is not rollout success,
and it is single-seed/single-sampling-mode. Stronger sampling studies and
closed-loop evaluation remain separate requirements before making execution or
benchmark-improvement claims.

### Phase 4I: Stronger GCBC Sampling Study

Question:

Can edge-balanced or bottleneck/support-balanced sampling improve rare-edge
GCBC fitting without unacceptable overall validation loss?

Reviewed before implementation:

- Goal-Conditioned Supervised Learning and RvS-style supervised offline RL.
- Class-balanced and focal-loss long-tail learning motivation.
- Existing Phase 3D smoke ablation and Phase 4H stronger Scene GCBC results.

Implemented:

- Added `support_balanced` sampling with edge probability proportional to
  `1 / sqrt(num_unique_starts)`.
- Added `bottleneck_support_balanced`, combining inverse-sqrt support with a
  normalized bottleneck multiplier.
- Replaced per-sample NumPy `Generator` construction in the hot sampling path
  with deterministic integer-hash sampling, speeding edge-level samplers while
  preserving reproducibility.
- Ran Scene H5 `core_plus_bottleneck_budget192_H5` for 3000 steps across five
  sampling modes and two seeds.

Evidence:

- `uniform_transition`: final validation MSE `0.008046`, rare-edge mean MSE
  `0.009620`.
- `uniform_edge`: rare-edge mean MSE improves to `0.008850` but final validation
  MSE worsens to `0.009178` (1.141x baseline).
- `support_balanced`: rare-edge mean MSE improves to `0.009088`, but final
  validation MSE worsens to `0.010109` (1.256x baseline).
- `bottleneck_support_balanced`: final validation MSE worsens to `0.010770` and
  rare-edge mean MSE is essentially unchanged at `0.009654`.

Analysis:

The simple edge-level rebalancing hypothesis is not supported as a default
training strategy. It can improve rare-edge metrics, especially `uniform_edge`,
but it gives up too much overall heldout action fitting under a 5% regret
tolerance. This suggests the next sampling direction should not be hard
oversampling. A softer mixture, curriculum, or loss-weighted objective may be
better because it can preserve transition coverage while giving controlled
extra weight to rare/bottleneck edges.

Remaining gap:

Phase 4I is still offline supervised evidence on Scene H5 with two seeds. It
does not prove rollout success, and it does not settle AntMaze or H10/H25
sampling choices. The next training-side attempt should test mixture/loss
weighting rather than repeating naive edge oversampling.

### Phase 4J: Mixed/Loss-Weighted GCBC Training

Question:

Can small, clipped per-edge loss weights improve rare-edge GCBC fitting while
preserving transition-uniform coverage and avoiding the overall MSE regression
seen under hard oversampling?

Reviewed before implementation:

- GCSL and RvS supervised offline RL framing.
- Class-balanced and focal-loss long-tail learning motivation.
- Phase 4I evidence that naive hard edge-level oversampling improves rare-edge
  MSE only by damaging overall heldout action fitting.

Implemented:

- Added per-edge supervised loss weighting to Phase 3 GCBC training.
- Supported `support`, `bottleneck`, and `support_bottleneck` loss-weight modes.
- Kept validation unweighted, so metrics remain comparable to previous GCBC
  runs.
- Ran Scene H5 `core_plus_bottleneck_budget192_H5` for 3000 steps, seeds
  `[0, 1]`, using Phase 4I `uniform_transition` as the baseline reference.

Evidence:

- `uniform_transition_none`: final validation MSE `0.008046`, rare-edge mean
  MSE `0.009620`.
- `loss_support_s03`: final validation MSE `0.008233` (1.023x baseline),
  rare-edge mean MSE `0.009315` (0.968x baseline).
- `loss_bottleneck_s03`: final validation MSE `0.008322` (1.034x baseline),
  rare-edge mean MSE `0.009200` (0.956x baseline).
- `loss_support_bottleneck_s03`: final validation MSE `0.008177` (1.016x
  baseline), rare-edge mean MSE `0.009038` (0.940x baseline).
- The combined support+bottleneck weighting also improves low-support-edge MSE
  to 0.928x baseline and long-horizon-edge MSE to 0.918x baseline.

Analysis:

Phase 4J gives the first clean training-side improvement after Phase 4H. The
result matches the pattern from Phase 4I: hard oversampling is too disruptive,
but soft loss weighting can improve rare-edge fitting while keeping overall
validation loss within a small regret budget. This makes
`loss_support_bottleneck_s03` the current preferred Scene H5 supervised
training variant for downstream direct repair-edge evidence.

Remaining gap:

This is still offline supervised evidence. It needs Phase 4G/4H-style direct
repair-edge re-evaluation and, eventually, closed-loop execution once
environment dependencies are available. It also needs AntMaze and Scene H10
replication before becoming a general default.

### Phase 4K: Loss-Weighted GCBC Direct Repair-Edge Validation

Question:

Does the Phase 4J clipped loss-weighting gain transfer from ordinary edge
validation groups to the more relevant Phase 4G/4H direct repair-edge evidence?

Reviewed before implementation:

- Phase 4G direct repair-edge policy evidence and repaired planner evaluation.
- Phase 4H stronger Scene GCBC validation.
- GCSL and RvS supervised offline RL framing.
- Class-balanced and focal-loss long-tail learning motivation.
- GCSL reference implementation as a simple GCBC-style code baseline.

Implemented:

- Added a Phase 4K batch validation script that reuses Phase 4G direct
  repair-edge scoring for arbitrary GCBC checkpoints.
- Compared Phase 4I `uniform_transition` baseline checkpoints against Phase 4J
  `loss_support_s03`, `loss_bottleneck_s03`, and
  `loss_support_bottleneck_s03`.
- Kept the repaired graph, repair-bank segments, path queries, compatibility
  planner, and calibration settings fixed.
- Ran Scene H5 `core_plus_bottleneck_budget192_H5` for two seeds per method.

Evidence:

- `uniform_transition_none`: direct repair-edge MSE `0.015525`, direct
  certified rate `0.887`, `calibrated_compat_threshold` uncertified edge
  fraction `0.035252`.
- `loss_support_s03`: direct repair-edge MSE `0.015547` (1.001x baseline),
  direct certified rate `0.888`.
- `loss_bottleneck_s03`: direct repair-edge MSE `0.015677` (1.010x baseline),
  direct certified rate `0.889`.
- `loss_support_bottleneck_s03`: direct repair-edge MSE `0.015314` (0.986x
  baseline), direct certified rate `0.890`, planner uncertified edge fraction
  `0.034210`.

Analysis:

Phase 4K strengthens but also narrows the Phase 4J conclusion. The combined
support+bottleneck loss weight still looks useful under direct repair-edge
evidence, improving repair-edge supervised MSE by about 1.4% with only about
1.6% ordinary validation-MSE regret. Single-component weights do not clearly
transfer: support-only is essentially flat and bottleneck-only worsens direct
repair MSE despite small certification-rate changes. The current training-side
default should therefore be conservative: small clipped combined weights, not
hard oversampling or single-signal weighting.

Remaining gap:

This is still reset-free offline supervised evidence. It does not prove
closed-loop repair-edge execution. The result also needs AntMaze and longer
Scene-H10/H25 replication before becoming a general cross-environment default.

### Phase 4L: Direct Repair-Edge Group Diagnostics

Question:

Where does the Phase 4K `loss_support_bottleneck_s03` repair-edge gain come
from, and are those improved edges actually used by the repaired planner?

Reviewed before implementation:

- Phase 4K per-edge direct repair scores and planner paths.
- Phase 4J/4K loss-weighting evidence.
- GCSL and RvS supervised offline RL framing.
- Class-balanced and focal-loss long-tail learning motivation.

Implemented:

- Matched every loss-weighted checkpoint against the same-seed
  `uniform_transition_none` baseline on identical repair edge ids.
- Computed per-edge MSE deltas and policy-support deltas.
- Grouped deltas by support, bottleneck score, horizon, compatibility context,
  planner usage, and repair reason.
- Reported both unweighted and sample-weighted group MSE deltas.

Evidence:

- Method-level `loss_support_bottleneck_s03` direct repair-edge MSE delta is
  `-0.000211`, while `loss_support_s03` is roughly flat at `+0.000023` and
  `loss_bottleneck_s03` worsens by `+0.000152`.
- The combined loss helps the intended hard groups:
  - low support: MSE delta `-0.000421`, sample-weighted delta `-0.000251`;
  - long horizon: MSE delta `-0.000417`, sample-weighted delta `-0.000126`;
  - high bottleneck: MSE delta `-0.000309`, sample-weighted delta `-0.000082`.
- The planner-used repair-edge group has only a small unweighted improvement
  (`-0.000067`) and a slightly worse sample-weighted delta (`+0.000060`).

Analysis:

Phase 4L explains the Phase 4K gain. Combined support+bottleneck weighting is
doing what it was intended to do: it shifts supervised fitting toward sparse,
longer-horizon, high-bottleneck repair edges. However, the current
compatibility-threshold planner does not use many of those improved repair
edges. This creates a training/planning mismatch: training improves plausible
hard repair edges, but planner risk constraints still route through a smaller
subset where the gain is weaker.

Remaining gap:

The result is still offline supervised diagnostics. The next algorithmic step
should either replicate combined weighting across AntMaze and Scene H10/H25 or
make the weighting planner-aware, so the low-support/high-bottleneck edges that
matter to selected paths get targeted directly.

### Phase 4M: Planner-Relevant Repair Loss Weighting

Question:

Can supervised loss weighting target repair edges that are both hard and
actually used by the compatibility-aware repaired planner, instead of weighting
all rare/high-bottleneck edges equally?

Reviewed before implementation:

- Phase 4L per-edge repair diagnostics and planner-used mismatch.
- GCSL and RvS supervised offline RL framing.
- Prioritized Experience Replay as priority-signal inspiration, while keeping
  Phase 4M distinct from TD-error replay.
- Class-balanced long-tail loss weighting.

Implemented:

- Added an external per-edge loss-weight path to Phase 3 GCBC training.
- Reconstructed Phase 4E augmented support-certified edge segments for training.
- Built clipped planner-relevant repair loss weights from Phase 4E repaired
  planner path usage plus hard repair-edge signals.
- Compared against the same augmented graph with ordinary
  `support_bottleneck` loss weighting.
- Reported final validation MSE, direct repair-edge MSE, planner-used
  repair-edge MSE, and repair policy support score.

Evidence:

- Scene H5 `core_plus_bottleneck_budget192_H5`, two seeds, 3000 steps:
  final validation MSE ratio `0.979`, direct repair-edge MSE ratio `0.970`,
  planner-used repair-edge MSE ratio `0.981`, and policy support score ratio
  `1.006` versus the same augmented graph with ordinary support+bottleneck loss.
- AntMaze H10 `core_plus_bottleneck_budget120_H10`, two seeds, 3000 steps:
  final validation MSE ratio `0.995`, direct repair-edge MSE ratio `0.995`,
  planner-used repair-edge MSE ratio `0.986`, and policy support score ratio
  `1.004`.
- Scene H10 `core_plus_bottleneck_budget192_H10`, two seeds, 3000 steps:
  final validation MSE ratio `1.009`, direct repair-edge MSE ratio `0.993`,
  planner-used repair-edge MSE ratio `0.962`, and policy support score ratio
  `1.000`.
- Across the three repaired graph runs, mean final validation MSE ratio is
  `0.994`, mean direct repair-edge MSE ratio is `0.986`, and mean planner-used
  repair-edge MSE ratio is `0.977`.
- Scene's 48 planner-used repair edges receive the highest mean loss weight
  (`1.217`). AntMaze's 21 planner-used repair edges receive the highest
  repair-subgroup mean loss weight (`1.151`). Weights remain clipped below
  `2.1`.

Analysis:

This is the cleanest training-side response to the Phase 4L mismatch so far,
and it now has first replication across Scene and AntMaze. The result suggests
planner relevance can be added as a mild supervised weighting signal that
consistently improves planner-used repair-edge MSE. However, Scene H10 shows a
small overall validation-MSE regression, so planner relevance should be treated
as a targeted repair-edge objective with an overall-regret guard rather than an
unqualified default. It still avoids the Phase 4I failure mode of hard
oversampling because transition coverage is unchanged and only the loss is
reweighted.

Remaining gap:

This is still reset-free offline supervised evidence. It is not rollout success.
It still needs Scene H25 and longer-training replication before becoming a
general default.

### Phase 4N: Planner-Relevance Regret Guard

Question:

Can the Scene H10 overall validation-MSE regression from Phase 4M be reduced by
using a weaker planner-relevant loss-weight strength, while preserving the
planner-used repair-edge MSE gain?

Reviewed before implementation:

- Phase 4M Scene H10 result, where aggressive `s04` planner-relevant weighting
  improved planner-used repair-edge MSE but regressed final validation MSE.
- Phase 4J/4K/4L evidence that hard oversampling is too disruptive and clipped
  supervised loss weighting is safer.
- The same GCSL/RvS supervised offline RL framing, long-tail loss weighting
  motivation, and priority-signal caveats already reviewed for Phase 4M.

Implemented:

- Added a Scene H10 regret sweep config over `planner_relevant_repair_s04`,
  `planner_relevant_repair_s02`, and `planner_relevant_repair_s01`.
- Kept the repaired support-certified graph, dataset, seeds, model, planner
  method, validation split size, and direct repair-edge scoring fixed.
- Reused existing baseline and `s04` checkpoints, then trained only the weaker
  `s02/s01` settings.

Evidence:

- Scene H10 `s04`: final validation MSE ratio `1.008751`, direct repair-edge
  MSE ratio `0.993475`, planner-used repair-edge MSE ratio `0.962466`.
- Scene H10 `s02`: final validation MSE ratio `1.005790`, direct repair-edge
  MSE ratio `0.988326`, planner-used repair-edge MSE ratio `0.960478`, and
  policy support score ratio `1.002015`.
- Scene H10 `s01`: final validation MSE ratio `1.006709`, direct repair-edge
  MSE ratio `1.000639`, and planner-used repair-edge MSE ratio `0.994857`.

Analysis:

`planner_relevant_repair_s02` is the best guarded Scene H10 setting from this
sweep. It keeps the planner-used repair-edge gain, improves direct repair-edge
MSE more than `s04`, and reduces overall validation-MSE regret from about
`0.875%` to about `0.579%`. The very weak `s01` setting is too weak for the
targeted repair-edge objective. Planner relevance should therefore be used with
an explicit regret guard or strength schedule rather than as an aggressive
default.

Remaining gap:

The Scene H10 trade-off is controlled but not eliminated. This is still
reset-free offline supervised evidence, not rollout success. Scene H25,
longer-training replication, and eventual env-available execution remain open.

### Phase 4O: Regret-Guard Selector

Question:

Can the Phase 4N manual choice be turned into a reusable candidate-selection
rule so later H25 and longer-training runs do not rely on ad hoc table reading?

Reviewed before implementation:

- Phase 4M/4N planner-relevant loss-weighting results and their explicit
  offline-supervised evidence boundary.
- Phase 4J/4I lesson that rare-edge fitting should improve without excessive
  overall validation-MSE regret.
- The required evidence standard in this document: offline supervised policy
  results require heldout MSE, grouped edge metrics, and explicit
  non-rollout-success language.

Implemented:

- Added a Phase 4O selector that reads existing `phase4m_vs_baseline.csv`
  tables.
- Annotates each candidate with final-validation, direct-repair,
  planner-used-repair, and policy-support guards.
- Recommends a strict non-baseline candidate when every guard passes.
- Adds a relaxed improvement fallback for candidates that improve direct
  repair-edge MSE, planner-used repair-edge MSE, and policy-support score while
  staying inside the final validation-MSE regret budget.
- Falls back to the same augmented-graph support+bottleneck baseline when no
  strict or relaxed planner-relevant candidate passes.
- Added synthetic tests for Scene H10 `s02` selection, fallback behavior, and
  violation reason reporting.

Evidence:

- AntMaze H10 B120 selects `planner_relevant_repair_s04`: final validation MSE
  ratio `0.995124`, direct repair-edge MSE ratio `0.995081`, planner-used
  repair-edge MSE ratio `0.986447`, policy support score ratio `1.003763`.
- Scene H10 B192 selects `planner_relevant_repair_s02`: final validation MSE
  ratio `1.005790`, direct repair-edge MSE ratio `0.988326`, planner-used
  repair-edge MSE ratio `0.960478`, policy support score ratio `1.002015`.
- Scene H5 B192 selects `planner_relevant_repair_s04`: final validation MSE
  ratio `0.979038`, direct repair-edge MSE ratio `0.969635`, planner-used
  repair-edge MSE ratio `0.980885`, policy support score ratio `1.005549`.
- Scene H25 B192 selects `planner_relevant_repair_s04` under the relaxed
  fallback: final validation MSE ratio `0.992440`, direct repair-edge MSE ratio
  `0.983579`, planner-used repair-edge MSE ratio `0.991416`, policy support
  score ratio `1.004670`.
- Across selected candidates, mean final validation MSE ratio is `0.993098`,
  mean direct repair-edge MSE ratio is `0.984155`, and mean planner-used
  repair-edge MSE ratio is `0.979806`.

Analysis:

Phase 4O makes the training-side selection rule more mature. It preserves the
targeted planner-used repair-edge improvement, but refuses candidates that
damage direct repair-edge fitting, policy-support score, or overall validation
MSE beyond the configured regret budget. This reduces manual cherry-picking
risk before expanding to longer training and additional environments.

Remaining gap:

The selector is only as good as the offline supervised metrics it reads. It is
not a rollout-success validator, and the guard thresholds are evidence-based
engineering defaults rather than calibrated execution probabilities.

### Phase 4P: Scene H25 Replication and Planner Scaling

Question:

Does the repaired support graph plus planner-relevant loss-weighting pattern
replicate at the larger Scene H25 horizon, and can the compatibility planner
scale to that graph size?

Reviewed before implementation:

- Phase 4E Scene H5/H10 repair results showing that support-bank repair fixes
  compressed graph coverage while compatibility constraints are still needed.
- Phase 4M/4N/4O planner-relevant weighting and guard-selection results.
- The Phase 4O evidence rule that selected training candidates must be guarded
  by ordinary validation MSE, direct repair-edge MSE, planner-used repair-edge
  MSE, and policy-support score.

Implemented:

- Added Scene H25 Phase 4E repair config using
  `core_plus_bottleneck_budget192_H25` as base graph and `all_budget192_H25` as
  the support-certified repair bank.
- Optimized compatibility-aware planning by reusing line-graph indices, method
  edge-cost maps, and pair-coverage lookups across query evaluations.
- Ran Scene H25 Phase 4E repair and Phase 4M planner-relevant GCBC training for
  baseline, `planner_relevant_repair_s04`, and `planner_relevant_repair_s02`.
- Reran the Phase 4O selector including Scene H25.

Evidence:

- Scene H25 strict `compat_threshold` coverage improves from `0.17` on the base
  compressed graph to `0.64` after adding 500 support-certified repair edges,
  while pair incompatible fraction remains `0.000`.
- Repaired support-shortest-path coverage reaches `0.65`, but pair incompatible
  fraction remains high at `0.791923`, confirming that graph repair alone is
  not enough without compatibility-aware planning.
- Scene H25 `planner_relevant_repair_s04`: final validation MSE ratio
  `0.992440`, direct repair-edge MSE ratio `0.983579`, planner-used repair-edge
  MSE ratio `0.991416`, policy support score ratio `1.004670`.
- Scene H25 `planner_relevant_repair_s02`: final validation MSE ratio
  `0.994574`, direct repair-edge MSE ratio `0.992829`, planner-used repair-edge
  MSE ratio `1.009024`, policy support score ratio `1.001577`.

Analysis:

Scene H25 strengthens the structural repair conclusion: the broader
support-certified bank contains useful compatible connectivity that the
compressed graph omitted. It also refines the weighting conclusion: weaker
planner relevance is not always safer for the actual planner-used repair group.
For H25, `s04` improves all broad supervised proxies and mildly improves
planner-used repair-edge MSE, while `s02` worsens planner-used repair-edge MSE.
This supports using Phase 4O selection guards rather than a fixed global
strength.

Remaining gap:

This remains reset-free offline evidence. The H25 training run is two seeds and
3000 steps; longer training and closed-loop evaluation are still required before
claiming execution or benchmark improvement.

## Lessons So Far

- Support certification is the strongest current distinction between BARS and
  proximity-style graph baselines.
- Graph path coverage alone is not a trustworthy metric; unsupported edge rate
  and path-level shortcut reliance must be reported beside coverage.
- Cluster-level connectivity does not imply option-level composability.
  Compatibility and termination-bridge metrics are required.
- GCBC validation MSE is a policy-fitting proxy, not an execution metric.
- Random or kNN graphs can look strong under path coverage while being weak
  under edge provenance.
- The current environment blocker prevents online rollout claims; reset-free
  analysis should be labeled accordingly.

## Required Review Gate for New Algorithm Attempts

Before implementing a new algorithmic direction, complete this checklist in a
short design note or in this document.

1. Define the target failure mode.
   - Which observed failure does the new idea address: unsupported shortcuts,
     poor bottleneck retention, low compatibility, GCBC fitting risk, or rollout
     dependency?
   - Which previous attempt above is closest, and why is the new idea not a
     repeat?

2. Review relevant papers.
   - Read the original paper or most authoritative paper for the method family.
   - Record the assumptions: reset access, online interaction, reward/task
     supervision, learned distance, graph construction, and evaluation protocol.
   - Record what metric in the paper actually supports the claimed conclusion.
   - Identify at least one limitation that matters for BARS.

3. Review open-source code when available.
   - Prefer official repositories, then widely used reproductions.
   - Inspect how graphs, edge labels, thresholds, rollouts, resets, and
     validation splits are implemented.
   - Check whether reported success depends on environment resets, privileged
     state, oracle distances, dense rewards, or task-specific heuristics.
   - Note dependency and reproducibility constraints before coding.

4. Specify the evidence standard.
   - Offline graph result: requires edge provenance, unsupported rate, and
     path-risk metrics.
   - Offline supervised policy result: requires heldout action MSE, grouped edge
     metrics, and explicit "not rollout success" language.
   - Closed-loop execution result: requires env preflight, reset/natural-start
     semantics, rollout success definition, seeds, and failure summaries.

5. Design the comparison.
   - Include a support-certified baseline.
   - Include kNN/proximity/random only as graph-risk controls unless their edges
     are support-certified.
   - Keep costs comparable, or explicitly label non-comparable costs such as hop
     count.
   - Report coverage together with unsupported edge rate, compatibility, and
     path-level risk.

6. Implement with mature engineering controls.
   - Add or update focused synthetic tests for new metric semantics.
   - Save resolved configs and exact commands.
   - Keep outputs under a phase-specific result directory.
   - Avoid broad refactors while testing an algorithmic hypothesis.

7. Update this log after the run.
   - Add the question, implementation, evidence, analysis, and remaining gap.
   - State which claims are now supported and which remain unproven.

## Candidate Next Attempts

### Phase 4M Loss-Weighted Replication

Hypothesis:

The Phase 4J/4K/4L `loss_support_bottleneck_s03` gain generalizes beyond Scene
H5 3000-step checkpoints to AntMaze and Scene H10/H25, and remains stable under
longer training.

Required review before implementation:

- Phase 4J loss-weighting aggregate metrics.
- Phase 4K direct repair-edge validation metrics.
- Phase 4L group diagnostics, especially the planner-used mismatch.
- Related loss reweighting and goal-conditioned supervised policy references.

Evidence required:

- Run the same clipped combined loss weighting across AntMaze and Scene H10/H25.
- Include at least matched baseline checkpoints and multiple seeds.
- Compare ordinary edge validation, direct repair-edge MSE, certification rate,
  and planner risk metrics.
- Keep offline proxy language unless closed-loop rollout is available.

### Planner-Relevant Loss Weighting

Status:

Completed as Phase 4M on Scene H5, Scene H10, Scene H25, and AntMaze H10, with
Phase 4N adding a Scene H10 regret guard, Phase 4O turning that guard into a
reusable selector, and Phase 4P extending the replication to Scene H25. The
remaining work is longer training and additional environments.

### Closed-Loop Edge Execution

Hypothesis:

Support-certified edges are more executable by GCBC than unsupported kNN,
proximity, or random edges.

Required review before implementation:

- OGBench environment construction and reset semantics.
- Official or reliable examples of resetting AntMaze/Scene-like environments.
- Success definitions used in related graph-option and goal-conditioned policy
  papers.

Evidence required:

- Env preflight passes.
- Reset or natural-start semantics are documented.
- Rollout success is reported separately from offline MSE.

### Natural-Start Online Smoke

Status:

Implemented as Phase 5B. The `gcrlo` environment can run natural-start
Gymnasium loops for AntMaze and Scene using OGBench `info["goal"]` as the
goal-conditioned policy input. Direct GCBC smoke with 2 episodes x 100 steps
completed but produced zero task success on both datasets, so the result is an
interface validation and a negative weak-baseline result, not evidence against
the graph method.

Lessons:

- The online blocker is no longer package construction in `gcrlo`.
- Arbitrary reset-to-state remains unsupported with the current offline dataset
  state references, so edge-level reset rollout should remain gated.
- Direct edge-BC GCBC is not a complete algorithm for natural-start tasks; the
  next mature algorithmic unit needs graph planning, option subgoal selection,
  and switching.

## Claims Currently Supported

- BARS Phase 2 can construct support-certified compressed option graphs from
  offline data.
- kNN/proximity/random graphs can overestimate path connectivity via unsupported
  shortcuts.
- Compatibility and bottleneck removal metrics reveal graph risks that coverage
  alone hides.
- GCBC can fit heldout edge BC samples in the offline supervised sense.
- Reset-free offline edge certification can rank support edges by proxy risk.
- Risk-penalized planning can preserve support-graph coverage while reducing
  offline path risk on the tested AntMaze and Scene runs.
- Phase 4B sweeps identify support-only Pareto planner configs and show that
  decomposed OOD/uncertified penalties are more stable than blindly increasing
  aggregate proxy risk weight.
- Phase 4C calibrated edge reliability improves heldout-support pseudo-label
  Brier and sharply reduces original uncertified path-edge exposure in
  recommended support-only paths.
- Phase 4D compatibility-aware planning reduces adjacent-edge composability risk
  without adding unsupported edges.
- Phase 4E support-bank graph repair recovers compatibility-safe Scene coverage
  using only support-certified edges.
- Phase 4F repair-edge certification restores planner-facing reliability on the
  repaired graph using conservative offline evidence.
- Phase 4G/4H direct repair-edge GCBC fitting supports the repaired Scene graph
  under both smoke and 10000-step supervised models.
- Phase 4I shows naive hard edge-level rebalancing is not a clean default:
  rare-edge MSE can improve, but overall validation MSE regresses too much on
  the tested Scene H5 setup.
- Phase 4J shows clipped support+bottleneck loss weighting improves Scene H5
  rare-edge supervised fitting with only small overall validation regret.
- Phase 4K shows the same clipped support+bottleneck weighting also improves
  Scene H5 direct repair-edge supervised MSE slightly under matched two-seed
  3000-step checkpoints.
- Phase 4L shows that the Phase 4K gain is concentrated on low-support,
  long-horizon, and high-bottleneck repair edges, while planner-used repair
  edges improve much less.
- Phase 4M shows that mild planner-relevant repair loss weighting improves
  direct repair-edge MSE and planner-used repair-edge MSE on Scene H5, Scene
  H10, and AntMaze H10, while Scene H10 exposes a small overall validation-MSE
  trade-off.
- Phase 4N shows that a weaker Scene H10 planner-relevant strength
  (`planner_relevant_repair_s02`) preserves the planner-used repair-edge gain
  while reducing overall validation-MSE regret relative to the more aggressive
  Phase 4M `s04` setting.
- Phase 4O provides a reusable offline supervised guard selector that chooses
  AntMaze H10 `s04`, Scene H10 `s02`, and Scene H5 `s04` under fixed regret and
  repair-edge improvement constraints.
- Phase 4P shows Scene H25 graph repair improves strict compatibility-safe
  coverage from `0.17` to `0.64`, and guarded planner-relevant weighting selects
  `s04` with improved final validation MSE, direct repair-edge MSE, and
  policy-support score.
- Phase 5B shows the local `gcrlo` stack can run reset-free natural-start
  OGBench closed-loop episodes with trained GCBC checkpoints.

## Claims Not Yet Supported

- Support-certified option edges are closed-loop executable by the trained GCBC.
- Any offline proxy score is calibrated to rollout success probability.
- BARS improves online task success over GAS/TDR/TMD/MQE.
- AntMaze or Scene lacks reset-to-state support.
- Sampling ablation results are statistically strong enough to choose a final
  training strategy.
- Phase 4A risk-aware paths are executable by the current GCBC policy.
- Phase 4B recommended configs are online-optimal or calibrated to execution
  success.
- Phase 4C calibrated reliability is a true probability of edge execution
  success.
- Phase 4H single-seed Scene supervised evidence is sufficient to choose a final
  GCBC sampling strategy.
- Naive support-balanced or bottleneck-support-balanced sampling is a better
  default than `uniform_transition`.
- Phase 4K loss weighting improves closed-loop repair-edge execution or online
  task success.
- The Phase 4K Scene H5 result generalizes to AntMaze or Scene H10/H25.
- Phase 4M planner-relevant loss weighting improves closed-loop repair-edge
  execution or online task success.
- The Phase 4M/4P result generalizes to longer training or additional
  environments.
- The Phase 4N guarded strength fully eliminates Scene H10 validation-MSE regret
  or generalizes as a fixed best strength across horizons.
- Phase 4O guard thresholds are calibrated to closed-loop execution success or
  should be treated as final hyperparameter-selection rules.
- Phase 5B direct GCBC smoke success rate is not evidence that support graphs
  are ineffective; it does not include hierarchical planning or option
  switching.
