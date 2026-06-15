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

## Current Interpretation

Phase5P source-head GCBC is still training and was around validation MSE
`0.055` near step 79000, worse than the earlier Phase3A `0.0426` and Phase5N
`0.0444` references. That reinforces the current experimental priority:
measure BARS graph improvements with a mature GAS actor before spending more
GPU time on another isolated BARS low-level policy variant.

Stage36 can support a strong conclusion only after online evaluation. If
success improves with the patched keygraph and unchanged policy, the graph
evidence is useful beyond our weaker GCBC actor. If it does not, the likely
bottleneck is graph-policy target switching, over-pruning, or the support proxy
itself, not just BARS policy weakness.
