# Phase 5M Policy Backbone Audit

Phase 5M reframes the next step around final success rate. Recent BARS phases
improved graph diagnostics, support certification, and partial final-goal
progress, but AntMaze natural-start success is still `0.0` with the current
Phase 3 GCBC policy.

The important design constraint is that a low-level policy is not an isolated
part. GAS's actor is trained with GAS's TDR representation, graph target
distribution, and skill semantics. Any other SOTA policy has similar coupling.
Therefore cross-policy reuse can only be a diagnostic ablation. The final BARS
algorithm must jointly define graph construction, goal/skill representation,
policy training data, loss, and execution protocol.

## Local Evidence

The local official GAS inventory already contains ready backbones for
`antmaze-large-stitch-v0`:

- ready GAS artifact seeds: `3`
- keygraph, TDR, policy, and dataset embeddings exist for seeds `44`, `45`,
  and `46`
- official GAS wide-atlas success rate on `antmaze-large-stitch-v0`: `0.9520`
  over `750` episodes

The inventory readiness is historical metadata. Before any cross-policy probe,
the current machine must verify that the corresponding `keygraph.pkl`,
`params_1000000.pkl`, and `dataset_embeddings.npy` files exist locally.

Current live check result:

- historical ready GAS backbones: `3`
- live local GAS backbones at recorded paths: `0`
- target-distribution audit status: `blocked_missing_gas_dataset_embeddings`
- BARS target samples collected for audit: `50000`

Current BARS natural-start smoke on the same task family:

| method | policy | planner | success | mean final distance |
| --- | --- | --- | ---: | ---: |
| direct GCBC final goal | BARS GCBC | none | 0.0 | 43.4040 |
| Phase 5I state-outcome | BARS GCBC | BARS support graph | 0.0 | 39.7197 |
| Phase 5K preplan mismatch | BARS GCBC | BARS support graph | 0.0 | 39.8942 |
| Phase 5L progress guard | BARS GCBC | BARS support graph | 0.0 | 42.8189 |
| official GAS | GAS actor | GAS keygraph | 0.9520 | n/a |

This makes the bottleneck concrete: BARS graph planning can improve partial
distance relative to direct GCBC, but the current BARS policy/executor is not
strong enough to reach task success.

## Planner-Policy Matrix

| experiment | status | purpose |
| --- | --- | --- |
| `official_gas_policy_official_gas_graph` | ready | Baseline success-protocol lock: official graph, official policy, official eval. |
| `bars_support_graph_bars_gcbc` | completed smoke | Current BARS end-to-end path; useful but not success evidence. |
| `bars_support_graph_gas_actor` | diagnostic only after distribution audit | Fast bottleneck probe, not a final BARS algorithm. |
| `official_gas_graph_bars_gcbc` | not direct | GAS planner targets are TDR phi/skill targets; BARS GCBC consumes raw goals. |
| `bars_planner_subgoal_replay_policy` | main training path | Train BARS policy on the same graph-derived targets used at execution time. |

## Technical Constraint

Official GAS does not pass raw observation goals directly to the actor. Its
evaluation computes:

```text
phi_obs = get_phi(observation)
cur_target_phi = selected graph/final-goal target in TDR space
skill = normalize(cur_target_phi - phi_obs)
action = GAS actor(observation, skill)
```

Therefore the easy cross-composition is not "raw BARS subgoal -> GAS actor".
The correct probe is:

```text
BARS selected termination observation
  -> GAS get_phi(termination_obs)
  -> normalized skill from current phi
  -> GAS actor
```

The reverse composition, GAS graph with BARS GCBC, needs either nearest raw
observation reconstruction from each GAS keygraph phi target or a learned phi to
raw-goal decoder. It is lower priority.

## GAS-Policy Compatibility Risk

Even with live GAS artifacts, directly using the GAS actor is not automatically
valid. GAS actor inputs are coupled to:

- GAS TDR `phi` representation;
- GAS keygraph node distribution;
- GAS policy-training skill distribution;
- GAS subgoal threshold and final-goal switching rule.

BARS option-graph nodes are clusters of raw observations, and BARS edge targets
are sampled real trajectory terminations. Mapping a BARS termination observation
through `GAS.get_phi` gives a vector in the same coordinate system, but it does
not prove that the GAS actor was trained to execute that target from the current
state. The mapped BARS target could still be outside the GAS keygraph/policy
target distribution.

Therefore the next step is a target-distribution feasibility audit:

```text
BARS edge termination obs -> GAS get_phi(target_obs)
compare against:
  - GAS keygraph node phi distribution
  - GAS dataset embedding distribution
  - official GAS path-edge target distances
```

Only if this audit looks compatible should we run `bars_support_graph_gas_actor`.
On the current machine this audit is blocked because the recorded GAS
`dataset_embeddings.npy` path is not present, so GAS actor reuse is not an
immediate execution path.

## Diagnostic Rollout Experiment

Implement `bars_support_graph_gas_actor` as a diagnostic success-rate probe:

1. Load official GAS agent and TDR using existing `params_1000000.pkl`.
2. Run the same OGBench natural-start task protocol as official GAS.
3. Replace only the target-selection source:
   - planner: BARS support-certified option graph;
   - target: selected edge termination observation;
   - mapping: `target_phi = GAS.get_phi(target_obs)`;
   - policy: unchanged GAS actor with normalized skill.
4. Compare against official GAS and current BARS GCBC under matched env/task
   seeds.

Interpretation:

- If BARS+GAS-policy approaches or beats official GAS, the graph construction
  has potential under a strong compatible control representation. This still
  does not prove a complete BARS algorithm.
- If BARS+GAS-policy stays low, BARS graph targets are not yet compatible with a
  strong low-level policy, so further graph certification or target sequencing
  is required before training more policies.

## Main Algorithm Path

The durable algorithmic route is `bars_planner_subgoal_replay_policy`:

1. Build a BARS support-certified graph.
2. Run the BARS planner offline or in natural-start rollouts to collect the
   actual target/subgoal distribution.
3. Train a BARS low-level policy on that same distribution, including final
   goals, edge-local goals, and planner-issued subgoals.
4. Keep the goal/skill representation consistent between training and
   execution.
5. Evaluate natural-start success against GAS under matched official protocol.

This is the route that can support a SOTA claim. Borrowed GAS/SOTA policies can
only shorten diagnosis and isolate whether the failure is mainly graph target
quality or BARS policy training. Since the current live GAS artifacts are not
available at the recorded paths, Phase 5N should proceed in parallel instead of
waiting on cross-policy reuse.

## Outputs

- `results/phase5m/policy_backbone_audit/antmaze_large_stitch/gas_backbone_inventory.csv`
- `results/phase5m/policy_backbone_audit/antmaze_large_stitch/bars_policy_smoke_summary.csv`
- `results/phase5m/policy_backbone_audit/antmaze_large_stitch/planner_policy_matrix.csv`
- `results/phase5m/policy_backbone_audit/antmaze_large_stitch/phase5m_policy_backbone_audit_summary.md`
- `results/phase5m/target_distribution_audit/antmaze_large_stitch_seed44/target_distribution_audit_summary.json`

Command:

```bash
conda run -n gcrlo python scripts/run_phase5m_policy_backbone_audit.py \
  --config configs/phase5m_policy_backbone_audit_antmaze.yaml

conda run -n gcrlo python scripts/run_phase5m_target_distribution_audit.py \
  --config configs/phase5m_target_distribution_audit_antmaze.yaml
```
