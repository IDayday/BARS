# Stage36 GAS Support-Keygraph Bridge

## Purpose

Stage36 is the lowest-risk route for testing whether BARS graph evidence can
improve final task success when the low-level policy is not the bottleneck.
It keeps the official GAS actor unchanged and only patches the official GAS
keygraph with BARS support evidence.

This is intentionally different from replacing GAS with the BARS cluster graph.
GAS trains its actor on normalized TDR skills, approximately
`normalize(phi_goal - phi_obs)`. BARS option nodes and edge IDs do not have the
same target semantics. Directly feeding BARS clusters to the GAS actor would
mix incompatible interfaces and would not be a clean success-rate test.

## What Transfers Directly

- Edge provenance: penalize or prune GAS keygraph edges that lack offline
  same-trajectory support under the available edge-score table.
- Risk-aware edge cost: optionally add a learned or heuristic edge risk column
  such as `r_exec` to the GAS edge weight.
- Success-rate A/B protocol: use the same GAS policy checkpoint and compare the
  original keygraph against the patched keygraph.

## What Does Not Transfer Directly

- BARS cluster IDs are not GAS actor targets.
- BARS option-edge IDs are not GAS skills.
- BARS GCBC checkpoints are not interchangeable with GAS TDR actors.
- Lower offline action MSE remains only a policy-fitting proxy, not closed-loop
  task success.

## Implementation

New files:

- `bars/gas_bars/support_keygraph.py`
- `scripts/stage36_patch_official_gas_keygraph_support.py`
- `scripts/stage36_score_gas_support_only.py`
- `configs/stage36_gas_support_keygraph_patch_antmaze.json`
- `tests/test_gas_support_keygraph_patch.py`

The patcher loads a GAS `keygraph.pkl` and an `edge_scores.csv` table with
`u`, `v`, and a support column such as `local_support`. It writes a patched
keygraph plus:

- `<stem>_edge_audit.csv`
- `<stem>_summary.json`

Important detail: GAS stores cached shortest paths to task targets inside the
keygraph pickle. The patcher recomputes `task_paths_dict` and
`task_paths_dist_dict` after changing edge weights or pruning edges. Without
this, `evaluate_gas.py` would silently execute old paths.

## Example Command Chain

First produce edge scores for an official GAS keygraph:

```bash
PYTHONPATH=/mnt/project/BARS conda run -n gcrlo python -m bars.gas_bars.score_edges \
  --env antmaze-large-stitch-v0 \
  --seed 0 \
  --gas-artifact-root artifacts/gas \
  --gas-repo-path external_src/GAS \
  --out runs_stage36_gas_support_patch/antmaze_large_stitch/seed0/edge_scores \
  --quick 1 \
  --gpu 0
```

Then patch the keygraph:

```bash
PYTHONPATH=/mnt/project/BARS conda run -n gcrlo python scripts/stage36_patch_official_gas_keygraph_support.py \
  --keygraph-path artifacts/gas/antmaze-large-stitch-v0/seed0/graph/keygraph.pkl \
  --edge-scores-csv runs_stage36_gas_support_patch/antmaze_large_stitch/seed0/edge_scores/edge_scores.csv \
  --out-keygraph-path runs_stage36_gas_support_patch/antmaze_large_stitch/seed0/keygraph_support_penalized.pkl \
  --mode penalize \
  --support-column local_support \
  --min-support 1 \
  --unsupported-penalty 8.0 \
  --risk-column r_exec \
  --risk-weight 0.0
```

Finally evaluate with the unchanged GAS policy:

```bash
PYTHONPATH=/mnt/project/BARS/external_src/GAS conda run -n gcrlo python external_src/GAS/evaluate_gas.py \
  --env_name antmaze-large-stitch-v0 \
  --keygraph_path runs_stage36_gas_support_patch/antmaze_large_stitch/seed0/keygraph_support_penalized.pkl \
  --policy_path artifacts/gas/antmaze-large-stitch-v0/seed0/policy/params_1000000.pkl \
  --eval_episodes 49 \
  --gpu 0
```

The valid comparison is original GAS keygraph vs patched keygraph with the
same policy checkpoint, task set, seeds, and evaluation hyperparameters.

## Smoke A/B Result On Official GAS Actor

Available complete large-stitch artifacts were not present in the current
workspace, but Stage30 retained an official GAS smoke artifact for
`antmaze-medium-navigate-v0`, seed `44`. I copied only its keygraph and policy
into `runs_stage36_gas_support_patch/artifact_root`, kept the policy checkpoint
unchanged, exported dataset embeddings through the GAS policy's `agent.get_phi`,
and generated support-only edge scores in the same GAS latent space.

Support audit:

- graph edges: `12332`
- goal connector edges: `150`
- non-goal local-supported edge rate: `0.1879`
- effective unsupported edge rate after protecting goal connectors: `0.8022`

Closed-loop A/B, same official GAS actor:

| method | episodes/task | unsupported penalty | success | mean episode length |
| --- | ---: | ---: | ---: | ---: |
| original keygraph | 5 | 0 | `0.96` | `260.48` |
| support penalty | 5 | 2 | `1.00` | `288.92` |
| support penalty | 5 | 8 | `0.92` | `331.68` |
| original keygraph | 20 | 0 | `0.97` | `248.70` |
| support penalty | 20 | 2 | `0.98` | `261.95` |

This is the first direct evidence that BARS graph evidence can be applied to a
mature GAS actor and change final closed-loop success without retraining the
policy. The effect is still small and from a near-saturated smoke environment:
penalty `2` gives a marginal success gain but longer paths, while penalty `8`
hurts success. The useful transfer direction is calibrated soft edge risk, not
hard pruning or large unsupported-edge penalties.

Result files:

- `runs_stage36_gas_support_patch/antmaze-medium-navigate-v0/seed44/stage36_gas_support_patch_ablation.csv`
- `runs_stage36_gas_support_patch/antmaze-medium-navigate-v0/seed44/stage36_gas_support_patch_ablation_summary.json`
- `runs_stage36_gas_support_patch/antmaze-medium-navigate-v0/seed44/support_only_edge_scores/support_only_metrics.json`

## Non-Saturated GAS Artifact Check

I then pulled the official Hugging Face GAS artifact for
`antmaze-giant-stitch-v0`, seed `0`, and repeated the same procedure. This is
a better stress test because the original GAS baseline is not saturated.

Support audit:

- graph edges: `31986`
- goal connector edges: `154`
- SCC connector edges: `112`
- non-goal local-supported edge rate: `0.2941`
- all-edge unsupported rate after protecting goal connectors: `0.7025`
- SCC-only unsupported rate after protecting ordinary distance edges: `0.0028`

The first giant run exposed an important implementation issue. Official GAS
computes cached task paths with target-rooted Dijkstra and reverses the path.
That is harmless when weights are approximately symmetric geometric distances,
but BARS support penalties are directional: `u -> v` can be unsupported even
when `v -> u` is supported. The patcher now recomputes cached task paths on
`graph.reverse(copy=False)`, so the stored path minimizes the original forward
execution cost under directional support weights.

The pre-fix giant table below should be treated as evidence of this direction
bug, not as the final support-penalty conclusion:

| method | support column | episodes/task | unsupported penalty | success | mean episode length |
| --- | --- | ---: | ---: | ---: | ---: |
| original keygraph | none | 10 | 0 | `0.90` | `690.42` |
| all-edge support penalty | `local_support` | 10 | 0.5 | `0.90` | `706.28` |
| all-edge support penalty | `local_support` | 10 | 2 | `0.86` | `757.26` |
| SCC-only support guard | `scc_only_support` | 10 | 8 | `0.90` | `681.30` |

After the forward-cost recompute fix, the authoritative giant smoke is:

| method | support column | episodes/task | unsupported penalty | success | mean episode length | path change rate | mean unsupported edge fraction | mean same-trajectory support |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original keygraph | none | 10 | 0 | `0.90` | `690.42` | `0.000` | `0.670` | `9.60` |
| all-edge support penalty | `local_support` | 10 | 0.5 | `0.92` | `689.16` | `0.892` | `0.366` | `17.37` |
| all-edge support penalty | `local_support` | 10 | 2 | `0.90` | `679.10` | `0.958` | `0.169` | `24.18` |
| SCC-only support guard | `scc_only_support` | 10 | 8 | `0.86` | `694.54` | `0.000` | `0.670` | `9.60` |

The corrected result changes the interpretation. A mild all-edge support
penalty is the only setting that improves success in this 50-episode smoke,
while the stronger all-edge penalty makes paths much more support-backed but
does not improve success. The SCC-only guard barely changes cached paths and
drops success in this sample, so it should not be treated as the current
winner.

The useful algorithmic signal is calibrated soft support risk. Too much support
pressure can trade unsupported shortcuts for longer paths without raising
success, and structural-only guarding may be too sparse to affect the planner.

Result files:

- `runs_stage36_gas_support_patch/antmaze-giant-stitch-v0/seed0/stage36_giant_support_patch_ablation.csv`
- `runs_stage36_gas_support_patch/antmaze-giant-stitch-v0/seed0/stage36_giant_support_patch_ablation_summary.json`
- `runs_stage36_gas_support_patch/antmaze-giant-stitch-v0/seed0/stage36_giant_forward_support_patch_ablation.csv`
- `runs_stage36_gas_support_patch/antmaze-giant-stitch-v0/seed0/stage36_giant_forward_support_patch_ablation_summary.json`
- `runs_stage36_gas_support_patch/antmaze-giant-stitch-v0/seed0/path_audit_forward/path_summary.csv`
- `runs_stage36_gas_support_patch/antmaze-giant-stitch-v0/seed0/support_only_edge_scores/support_only_metrics.json`

## Current Interpretation

Phase5P source-head GCBC did not solve the BARS policy bottleneck. Stage36 is
therefore the right bridge for near-term success-rate evidence: test graph
evidence on a mature official GAS actor while preserving GAS target semantics.

The smoke A/B says the bridge is viable but not yet a complete algorithmic win.
BARS support evidence can directly improve a mature GAS actor's final success
in the mild `p0.5` giant smoke, but the effect is small and needs multi-seed /
larger-episode confirmation. It should be used as a calibrated risk/cost prior
over GAS edges, not as a hard replacement for GAS's graph.

The strongest concrete rule so far is:

- recompute cached task paths with forward execution costs whenever edge costs
  are directional;
- keep support pressure mild unless closed-loop evaluation justifies stronger
  penalties;
- move from constant penalties to calibrated edge/path risk models.

The next decisive experiment is the same patch on non-saturated official GAS
artifacts with calibrated risk rather than constant penalties. `scene-play-v0`
is also a useful next check because its way-step scale and manipulation
dynamics differ from AntMaze.

## Stage37 Calibrated Support Risk

Stage37 tests the next step: replace fixed binary unsupported-edge penalties
with a continuous support-risk column computed from `same_traj_support`.

The generated calibrated columns are:

- `risk_unsupported_binary`: `1` for unsupported non-goal edges, else `0`.
- `risk_inv_sqrt_support`: `1 / sqrt(1 + same_traj_support)`.
- `risk_low_support_target`: linear shortfall to a target support count.
- `risk_hybrid_support`: average of inverse-sqrt risk and target shortfall.

Goal connectors are protected with zero calibrated risk. The keygraph patcher
then uses `unsupported_penalty=0` and adds
`risk_weight * risk_hybrid_support` to the GAS edge cost.

Giant stitch, same official GAS actor, 10 episodes/task:

| method | risk signal | weight | success | mean episode length | mean unsupported path-edge fraction | mean same-trajectory support |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original keygraph | none | 0 | `0.90` | `690.42` | `0.670` | `9.60` |
| binary support | `local_support == 0` | 0.25 | `0.90` | `693.44` | `0.502` | `14.73` |
| binary support | `local_support == 0` | 0.5 | `0.92` | `689.16` | `0.366` | `17.37` |
| binary support | `local_support == 0` | 2.0 | `0.90` | `679.10` | `0.169` | `24.18` |
| hybrid support risk | `risk_hybrid_support` | 0.25 | `0.96` | `675.66` | `0.523` | `14.84` |
| hybrid support risk | `risk_hybrid_support` | 0.5 | `0.94` | `694.10` | `0.435` | `17.31` |
| SCC-only support | `scc_only_support` | 8.0 | `0.86` | `694.54` | `0.670` | `9.60` |

The best setting, `risk_hybrid_support` with weight `0.25`, was rerun with
20 episodes/task:

| method | episodes/task | success | mean episode length | task1 | task2 | task3 | task4 | task5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original keygraph | 20 | `0.91` | `687.01` | `0.80` | `1.00` | `0.95` | `0.85` | `0.95` |
| hybrid support risk | 20 | `0.94` | `677.69` | `0.95` | `1.00` | `0.95` | `0.85` | `0.95` |

This is the strongest final-success signal so far. It is still one seed and
one environment, but it shows that a calibrated BARS support-risk prior can
improve a mature GAS actor's closed-loop success and shorten trajectories
without changing the actor.

Engineering note: keygraph patching should run with `JAX_PLATFORMS=cpu` when
not evaluating, because official GAS keygraph pickles may contain JAX arrays
and can otherwise initialize CUDA during unpickling. Closed-loop evaluation can
use the GPU normally.

## Stage38 Lower-Score Slice Replication

The first Stage37 validation used all five giant-stitch tasks and showed a
clear aggregate gain, but the task scores are partly saturated. Stage38
therefore rechecked the same calibrated patch on a lower-score task slice and
on the near-saturated medium sanity environment.

The patch is unchanged: official GAS actor, original GAS node/target semantics,
`risk_hybrid_support`, `risk_weight=0.25`, no hard unsupported penalty, and
forward-cost cached path recomputation.

| dataset / slice | episodes | method | success | mean length | mean unsupported path-edge fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| giant stitch, all tasks | 20/task | original | `0.91` | `687.01` | `0.670` |
| giant stitch, all tasks | 20/task | hybrid support risk | `0.94` | `677.69` | `0.523` |
| giant stitch, task1 only | 50 | original | `0.88` | `832.92` | `0.670` |
| giant stitch, task1 only | 50 | hybrid support risk | `0.88` | `826.98` | `0.523` |
| medium navigate, all tasks | 20/task | original | `0.97` | `247.12` | `0.723` |
| medium navigate, all tasks | 20/task | hybrid support risk | `0.98` | `238.27` | `0.641` |

This changes the interpretation in an important way. The all-task giant result
is still positive, and the support-risk patch consistently reduces unsupported
path-edge usage. However, the lower-score task1-only slice does not improve
success when evaluated with 50 episodes; it only shortens trajectories. The
previous 20-episode task1 gain should therefore be treated as suggestive but
not stable enough on its own.

Medium navigate remains useful as a sanity check, but it is close to saturated:
`0.97 -> 0.98` leaves too little headroom for decisive algorithm evidence.
Future replications should prioritize lower-baseline official GAS artifacts,
especially `antmaze-giant-navigate-v0`, `antmaze-large-explore-v0`, and
`scene-play-v0`.

A quick inventory of local historical GAS evaluations confirms this priority:
`antmaze-giant-navigate-v0` has many all-task runs around `0.54-0.60` success
and a median around `0.64`, so it has substantially more visible headroom than
medium navigate.

I then prepared the official `antmaze-giant-navigate-v0` GAS artifact and ran a
small hybrid-risk weight sweep. The policy checkpoint and GAS node semantics
were unchanged. Only the keygraph edge cost changed.

| dataset / slice | episodes | method | success | mean length | unsupported path-edge fraction | mean same-traj support |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| giant navigate, all tasks | 20/task | original | `0.76` | `763.02` | `0.694` | `6.93` |
| giant navigate, all tasks | 20/task | hybrid risk `w=0.05` | `0.75` | `755.25` | `0.669` | `7.95` |
| giant navigate, all tasks | 20/task | hybrid risk `w=0.10` | `0.83` | `722.13` | `0.638` | `8.87` |
| giant navigate, all tasks | 20/task | hybrid risk `w=0.25` | `0.75` | `751.98` | `0.555` | `11.03` |

This is the cleanest lower-baseline success result so far: a mild calibrated
support-risk prior improves official GAS success by `+0.07` and shortens mean
episode length by `40.89` steps. The dose response is also informative.
Path-risk metrics improve monotonically with weight, but closed-loop success
does not. On this sparser graph, `w=0.25` over-regularizes planning even though
it most strongly reduces unsupported path usage. The practical rule is to
calibrate support pressure to graph support density instead of reusing a fixed
weight across environments.

Result files:

- `runs_stage36_gas_support_patch/stage38_cross_scenario/stage38_cross_scenario_summary.csv`
- `runs_stage36_gas_support_patch/stage38_cross_scenario/stage38_cross_scenario_summary.json`
- `runs_stage36_gas_support_patch/stage38_low_baseline_candidates/antmaze_giant_navigate_candidate_scores.csv`
- `runs_stage36_gas_support_patch/stage38_low_baseline_candidates/antmaze_giant_navigate_candidate_scores_summary.json`
- `runs_stage36_gas_support_patch/antmaze-giant-navigate-v0/seed0/stage38_giant_navigate_hybrid_weight_sweep_summary.csv`
- `runs_stage36_gas_support_patch/antmaze-giant-navigate-v0/seed0/stage38_giant_navigate_hybrid_weight_sweep_summary.json`

## Stage39 Large-Explore Replication

Stage39 repeats the calibrated support-risk bridge on official
`antmaze-large-explore-v0`, seed `0`. The official GAS actor and GAS node
semantics are unchanged. The only intervention is a BARS-derived
`risk_hybrid_support` edge-cost prior in the GAS keygraph, with forward-cost
cached path recomputation.

This graph is much sparser than giant-navigate under the same support target:
non-goal supported edge rate is only `0.1599`, and calibrated hybrid risk has
median and 90th percentile `1.0`. That means support risk is close to
saturated, so the useful question is whether a stronger weight can still help
or whether it over-regularizes execution.

Closed-loop results:

| dataset | episodes | method | success | mean length | unsupported path-edge fraction | mean same-traj support | path change rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| large explore | 20/task | original | `0.940` | `419.690` | `0.852` | `1.287` | `0.000` |
| large explore | 20/task | hybrid risk `w=0.10` | `0.960` | `405.840` | `0.850` | `1.326` | `0.091` |
| large explore | 20/task | hybrid risk `w=0.25` | `0.970` | `399.350` | `0.846` | `1.369` | `0.204` |
| large explore | 20/task | hybrid risk `w=0.50` | `0.980` | `394.020` | `0.834` | `1.485` | `0.489` |
| large explore | 50/task | original | `0.944` | `415.884` | `0.852` | `1.287` | `0.000` |
| large explore | 50/task | hybrid risk `w=0.10` | `0.952` | `409.168` | `0.850` | `1.326` | `0.091` |
| large explore | 50/task | hybrid risk `w=0.25` | `0.964` | `403.524` | `0.846` | `1.369` | `0.204` |
| large explore | 50/task | hybrid risk `w=0.50` | `0.976` | `400.676` | `0.834` | `1.485` | `0.489` |

The ep50 result is the important one: `w=0.50` improves success by `+0.032`
and shortens mean episode length by `15.208` steps over the unchanged official
GAS baseline. This is now a second official AntMaze artifact where BARS
support-risk evidence improves final closed-loop GAS success without retraining
the actor.

The interpretation differs from giant-navigate. On giant-navigate, `w=0.25`
over-regularized and `w=0.10` was best. On large-explore, stronger support
pressure continues to help up to `w=0.50`, despite only modest graph-layer
support improvement. The current rule is therefore not a fixed global weight.
Support-risk strength must be calibrated to the environment's support density,
baseline policy behavior, and closed-loop validation.

Result files:

- `runs_stage39_gas_support_patch/antmaze-large-explore-v0/seed0/stage39_large_explore_hybrid_weight_sweep_summary.csv`
- `runs_stage39_gas_support_patch/antmaze-large-explore-v0/seed0/stage39_large_explore_hybrid_weight_sweep_summary.json`
- `runs_stage39_gas_support_patch/antmaze-large-explore-v0/seed0/support_only_edge_scores/support_only_metrics.json`
- `runs_stage39_gas_support_patch/antmaze-large-explore-v0/seed0/calibrated_support_scores/edge_scores_calibrated_target8_summary.json`
- `runs_stage39_gas_support_patch/antmaze-large-explore-v0/seed0/path_audit_hybrid_weight_sweep_extended/path_summary.csv`
- `runs_stage39_gas_support_patch/antmaze-large-explore-v0/seed0/path_audit_hybrid_weight_sweep_extended/path_diff_summary.csv`

## Stage40 Scene-Play Cost-Scale Check

Stage40 tests whether the same support-risk bridge transfers to the
manipulation-style `scene-play-v0` artifact. The setup is still clean: official
GAS actor unchanged, official GAS keygraph nodes unchanged, and only
`risk_hybrid_support` is added to keygraph edge cost.

The result is not a simple AntMaze-style win. Scene's support audit is much
sparser in local binary support and much less sensitive to small weights:
non-goal supported edge rate is `0.0098`, `risk_hybrid_support` p50/p90 are
both `1.0`, and AntMaze-scale weights barely change cached paths. A higher
weight is needed just to affect planning.

Closed-loop results:

| dataset | episodes | method | success | mean length | unsupported path-edge fraction | mean same-traj support | path change rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| scene play | 20/task | original | `0.730` | `369.140` | `0.973` | `11.240` | `0.000` |
| scene play | 20/task | hybrid risk `w=5` | `0.700` | `388.180` | `0.963` | `16.367` | `0.059` |
| scene play | 20/task | hybrid risk `w=10` | `0.690` | `385.170` | `0.952` | `19.524` | `0.132` |
| scene play | 20/task | hybrid risk `w=20` | `0.720` | `374.770` | `0.941` | `23.575` | `0.245` |
| scene play | 50/task | original | `0.756` | `343.468` | `0.973` | `11.240` | `0.000` |
| scene play | 50/task | hybrid risk `w=2` | `0.736` | `352.736` | `0.970` | `12.935` | `0.023` |
| scene play | 50/task | hybrid risk `w=5` | `0.764` | `348.896` | `0.963` | `16.367` | `0.059` |
| scene play | 50/task | hybrid risk `w=10` | `0.724` | `359.820` | `0.952` | `19.524` | `0.132` |
| scene play | 50/task | hybrid risk `w=20` | `0.728` | `360.108` | `0.941` | `23.575` | `0.245` |

The only positive all-task setting is `w=5`, and the gain is small:
`0.756 -> 0.764`, with longer episodes. Stronger weights improve graph support
metrics but reduce final success. Per-task deltas also show conflicting effects:
some easier/opening tasks improve under stronger risk, while rearrangement/hard
tasks degrade.

The Scene lesson is important for the algorithm. A global support-risk scalar
is not enough across task families. Manipulation tasks need risk to be
normalized to the base edge-cost scale and conditioned on task/path context;
otherwise support pressure can remove useful shortcuts for hard subtasks even
when graph-level support metrics look better.

Result files:

- `runs_stage40_gas_support_patch/scene-play-v0/seed0/stage40_scene_hybrid_weight_sweep_summary.csv`
- `runs_stage40_gas_support_patch/scene-play-v0/seed0/stage40_scene_hybrid_weight_sweep_summary.json`
- `runs_stage40_gas_support_patch/scene-play-v0/seed0/support_only_edge_scores/support_only_metrics.json`
- `runs_stage40_gas_support_patch/scene-play-v0/seed0/calibrated_support_scores/edge_scores_calibrated_target8_summary.json`
- `runs_stage40_gas_support_patch/scene-play-v0/seed0/path_audit_hybrid_weight_sweep_extended/path_summary.csv`
- `runs_stage40_gas_support_patch/scene-play-v0/seed0/path_audit_hybrid_weight_sweep_extended/path_diff_summary.csv`
