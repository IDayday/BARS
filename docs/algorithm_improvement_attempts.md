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

### Phase 4D Compatibility-Aware Planning

Hypothesis:

Adding adjacent-edge compatibility penalties or constraints to calibrated
support-only planning will reduce path composability risk, especially on Scene.

Required review before implementation:

- Constrained shortest path with transition-dependent costs.
- Hierarchical planning methods that score option composability.
- Existing GAS/CAGE code paths for path drift, subgoal switching, and contract
  ranking.

Evidence required:

- Use Phase 2.2 edge-pair compatibility or termination bridge coverage as an
  adjacent-edge path cost.
- Compare against Phase 4C calibrated planner on identical queries.
- Report coverage, calibrated reliability, original uncertified edge fraction,
  and path-level compatibility.
- Keep offline proxy language unless closed-loop rollout is available.
- Explicitly label the result as reset-free offline planning, not execution
  success.

### Stronger GCBC Sampling Study

Hypothesis:

Edge-balanced or bottleneck-weighted sampling improves fitting on rare,
bottleneck, or low-support edges without unacceptable overall MSE degradation.

Required review before implementation:

- Goal-conditioned behavioral cloning sampling schemes.
- Imbalanced supervised learning and per-group risk minimization.
- Prior hierarchical offline RL implementations using subgoal-conditioned BC.

Evidence required:

- Multiple seeds and longer training than the current smoke run.
- Same validation split across sampling modes.
- Edge-wise metrics with confidence intervals or seed variance.

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
