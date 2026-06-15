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

## Current Interpretation

Phase5P source-head GCBC did not solve the BARS policy bottleneck. Stage36 is
therefore the right bridge for near-term success-rate evidence: test graph
evidence on a mature official GAS actor while preserving GAS target semantics.

The smoke A/B says the bridge is viable but not yet a complete algorithmic win.
BARS support evidence should be used as a calibrated risk/cost prior over GAS
edges. It should not directly replace GAS's graph or aggressively remove
distance edges, because the trained GAS actor and planner were optimized around
that graph's target distribution.

The next decisive experiment is the same patch on non-saturated official GAS
artifacts such as `antmaze-large-stitch-v0` or `antmaze-giant-stitch-v0`, where
there is enough baseline failure rate to measure whether support-calibrated
graph costs improve success rather than merely reshuffle already solved tasks.
