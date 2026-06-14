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

- Current probes report `env_unavailable` due missing `gymnasium`/`gym`/MuJoCo
  dependencies in the active environment.

Analysis:

The blocker is environment construction. It should not be interpreted as a
benchmark-level reset limitation.

Remaining gap:

Phase 3C closed-loop edge execution is pending until env dependencies are
available and reset semantics can be probed reliably.

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

### Phase 4K Direct Evidence for Loss-Weighted GCBC

Hypothesis:

The Phase 4J `loss_support_bottleneck_s03` model improves direct repair-edge
policy evidence relative to the 10000-step uniform-transition Scene model,
especially on repair edges with low support or high bottleneck score.

Required review before implementation:

- Phase 4G direct repair-edge policy evidence.
- Phase 4H stronger Scene GCBC validation.
- Direct repair-edge grouped diagnostics by support, bottleneck, and horizon.

Evidence required:

- Train a longer loss-weighted Scene model or reuse a matched-step comparison.
- Re-run direct repair-edge policy evidence.
- Compare direct repair-edge MSE, certification rate, and planner risk metrics.
- Keep offline proxy language unless closed-loop rollout is available.

### Mixed/Loss-Weighted Replication

Hypothesis:

The Phase 4J loss-weighting gain generalizes to AntMaze and Scene H10/H25.

Evidence required:

- Same weighting scheme across multiple graphs and seeds.
- Compare rare-edge gains and overall validation regret.
- Report cases where weighting hurts.

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
- Phase 4J loss weighting improves direct repair-edge evidence or online
  execution; that still needs the next validation step.
