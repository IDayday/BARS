# CAGE-GP0 Graph-Policy Alignment Audit

## Scope

Repository root: `/mnt/project/BARS`

GAS code: `external_src/GAS`

Checkpoint root:
`/mnt/project/BARS/artifacts/gas_ogbench_offline_full_20260522_165138`

Environments and seeds:

| env | seed |
| --- | ---: |
| antmaze-giant-navigate-v0 | 42 |
| antmaze-giant-stitch-v0 | 42 |
| humanoidmaze-large-navigate-v0 | 44 |

No TDR retraining, keygraph reconstruction, policy retraining, threshold tuning, risk-aware path search, or 8-env benchmark was run.

## Implementation

Added:

- `scripts/extract_graph_planned_goal_distribution.py`
- `scripts/extract_policy_training_goal_distribution.py`
- `scripts/compare_graph_policy_distributions.py`
- `scripts/probe_policy_edge_success.py`
- `scripts/cage_gp0_common.py`
- `external_src/GAS/cage/graph_policy_dataset.py`

The q_train extractor mirrors the policy training logic in `external_src/GAS/O_utils/datasets.py`:

- `GASDataset.process_features()` stores TDR embeddings.
- `GASDataset.build_waysteps_idx_by_distance()` finds the first same-trajectory future state whose phi distance reaches `way_steps`.
- `GASDataset.sample()` samples a geometric offset and caps it at that waypoint.

The q_G extractor uses the saved keygraph and dataset embeddings only. It does not execute the environment or policy.

## Artifacts

Graph-planned distribution:

- `results/cage_gp0/focused/antmaze_nav/qG.jsonl`
- `results/cage_gp0/focused/antmaze_stitch/qG.jsonl`
- `results/cage_gp0/focused/humanoid_large_nav/qG.jsonl`

Policy-training distribution:

- `results/cage_gp0/focused/antmaze_nav/qtrain.jsonl`
- `results/cage_gp0/focused/antmaze_stitch/qtrain.jsonl`
- `results/cage_gp0/focused/humanoid_large_nav/qtrain.jsonl`

Comparison:

- `results/cage_gp0/focused/compare/graph_policy_compare.md`
- `results/cage_gp0/focused/compare/graph_policy_compare.json`
- `results/cage_gp0/focused/compare/qG_pair_support.jsonl`

Execution proxy:

- `results/cage_gp0/focused/probe/policy_edge_trace_proxy.md`
- `results/cage_gp0/focused/probe/policy_edge_trace_proxy.json`
- `results/cage_gp0/focused/probe/policy_edge_execute_blocked.md`

## q_G And q_train Summary

| env | q_G pairs | q_G d_phi mean | q_G path length mean | q_train pairs | q_train d_phi mean | q_train temporal gap mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-giant-navigate-v0 | 11916 | 5.77 | 23.83 | 8000 | 5.91 | 38.99 |
| antmaze-giant-stitch-v0 | 12458 | 6.31 | 24.92 | 8000 | 5.99 | 26.42 |
| humanoidmaze-large-navigate-v0 | 6987 | 25.28 | 13.97 | 8000 | 24.65 | 45.47 |

The graph planner asks the policy to execute TDR-space targets at roughly the same average `d_phi` scale as the policy training relabeling distribution. Humanoid edges are much larger in absolute phi distance, but that is also true in its q_train replay.

## Distribution Comparison

| env | d_phi JS divergence | NN support mean | coverage rate | final-phase coverage | graph d_phi mean | q_train d_phi mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-giant-navigate-v0 | 0.3932 | 0.6348 | 0.9882 | 1.0000 | 5.77 | 5.91 |
| antmaze-giant-stitch-v0 | 0.3225 | 0.4951 | 0.9114 | 0.7837 | 6.30 | 5.99 |
| humanoidmaze-large-navigate-v0 | 0.3536 | 0.5248 | 0.9798 | 0.9920 | 25.24 | 24.65 |

Coverage here means nearest graph pair to q_train pair distance is within the automatically selected support radius. Support score is `exp(-nn_pair_distance / tau)`.

Path-position coverage:

| env | weakest segment | coverage | support score |
| --- | --- | ---: | ---: |
| antmaze-giant-navigate-v0 | initial | 0.9499 | 0.6045 |
| antmaze-giant-stitch-v0 | final | 0.7837 | 0.4931 |
| humanoidmaze-large-navigate-v0 | initial | 0.9487 | 0.5151 |

Recovery target coverage is `NA` in this run because Repair-0 traces were episode-level only and did not contain step-level recovery target vectors. The comparison script is prepared for recovery-labeled rows when debug traces include target vectors.

## Execution Proxy

The probe script joined q_G support with Repair-0 trace metrics. This is a trace proxy, not a new edge rollout.

Correlations:

| metric | correlation |
| --- | ---: |
| segment reach vs q_train support | 0.1825 |
| segment reach vs graph d_phi | -0.3829 |
| success vs q_train support | 0.0467 |
| success vs graph d_phi | -0.3430 |

The current trace proxy does not show that q_train support predicts execution success better than graph distance. It suggests graph distance and environment/task difficulty still explain more variance at this coarse resolution. A true edge-level policy probe is blocked because q_G rows are in phi space and do not include raw simulator reset states.

## Empirical Mismatch Law

GP0 does not support a simple claim that GAS fails because q_G is globally outside q_train support.

The more precise law is:

1. q_G and q_train overlap at the coarse TDR-distance scale.
2. Despite this overlap, closed-loop execution can still fail because the policy is asked to execute chained targets under drift, commitment changes, recovery retries, and final-goal switching.
3. Humanoid failure is not explained by missing graph paths and not fully explained by q_G/q_train distance support. It is primarily an interface-side compounding error: drift detection and recovery convert execution mismatch into replan churn.
4. AntMaze stitch has weaker final-phase support than navigate, so final-goal interface mismatch remains a plausible contributor there.
5. Fixed commitment likely helps because it reduces temporal target distribution shift and removes recovery/replan churn, not because it creates a better graph path.

## Failure Attribution

Graph-side:

- Not the dominant failure in Repair-0. `no_path` was not the observed problem.
- q_G path lengths and edge distances are plausible under the keygraph.

Policy-side:

- The low-level policy was trained on same-trajectory geometric waypoint targets, not graph-induced multi-segment targets.
- Coarse NN support is high in humanoid, but that does not prove executability. It only says phi-space pair support is not obviously absent.
- True policy edge success still needs raw reset-state probes or step-level target traces.

Interface-side:

- Dominant current failure mode.
- Full CAGE on humanoid had zero segment reach, zero recovery success, and thousands of global replans.
- Safe full reduced replan storm but did not restore success, so guardrails fix churn but not the underlying graph-policy execution alignment.

## Graph-Induced Policy Training Targets

Recommended target construction for the next training milestone:

1. Mix q_train with q_G path-edge pairs from the same saved keygraph.
2. Oversample initial planner targets and late/final path positions.
3. Add recovery-candidate targets from suffix path nodes only after debug traces record actual recovery target vectors.
4. Weight targets by q_train support score: keep high-support graph targets for conservative adaptation and separately track low-support graph targets as hard negatives or curriculum targets.
5. Preserve the original low-level policy objective and add graph-induced actor-goal sampling as an explicit opt-in dataset mode.

Do not start this training until a preregistered CAGE-v0.2 protocol is written.

## Validation

Commands run:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH /root/miniconda3/envs/gcrlo/bin/python -m py_compile scripts/cage_gp0_common.py scripts/extract_graph_planned_goal_distribution.py scripts/extract_policy_training_goal_distribution.py scripts/compare_graph_policy_distributions.py scripts/probe_policy_edge_success.py external_src/GAS/cage/graph_policy_dataset.py
```

Result: return code 0.

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH PYTHONPATH=/mnt/project/BARS/external_src/GAS:$PYTHONPATH /root/miniconda3/envs/gcrlo/bin/python - <<'PY'
from cage.graph_policy_dataset import GraphPolicyDataset
d = GraphPolicyDataset.from_jsonl('results/cage_gp0/smoke/qG.jsonl')
b = d.sample(4)
print({'size': d.size, 'phi_obs_shape': b['phi_obs'].shape, 'skills_shape': b['actor_skills'].shape})
PY
```

Result: `{'size': 604, 'phi_obs_shape': (4, 32), 'skills_shape': (4, 32)}`.

## Blockers

- Real frozen-policy edge rollout is blocked until q_G rows include raw reset states or the environment exposes a safe arbitrary-state reset API.
- Recovery target coverage is blocked until CAGE debug traces include actual selected/recovery target vectors.
- Current support metric is nearest-neighbor in phi-pair space. It is a coverage audit, not a calibrated reachability certificate.

## Next Command

Recommended next step is to collect a tiny debug trace with step-level selected target vectors before training anything:

```bash
PATH=/root/miniconda3/envs/gcrlo/bin:$PATH WANDB_MODE=disabled /root/miniconda3/envs/gcrlo/bin/python scripts/run_cage_manifest.py --manifest_path results/cage_repair0/minipilot_humanoid_large_nav/manifests/minipilot_manifest.jsonl --max_jobs 1 --dry_run
```

After adding vector logging as a trace-only instrumentation patch, rerun GP0 compare with recovery-labeled target rows.

## CLP0 Follow-Up

CAGE-CLP0 added exact `StateRef` auditing and true frozen-policy closed-loop probes. AntMaze dataset observations can be converted to exact `qpos/qvel` state refs, but existing HumanoidMaze dataset observations cannot. See `docs/cage_clp0_closed_loop_report.md` for the current closed-loop contract results and blocker.
