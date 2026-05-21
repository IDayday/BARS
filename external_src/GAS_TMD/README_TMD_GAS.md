# TMD-GAS Representation Backbone

This directory is a GAS copy with experimental TMD integration.  The original
`external_src/GAS/` tree is not modified.

## Modes

### `tmd_graph_gas_policy`

Build a directed key graph with TMD psi embeddings and TMD quasimetric edge
weights, then execute graph subgoals with a trained GAS low-level policy.  The
GAS actor receives the original GAS-TDR direction skill computed from raw
medoid observations.

### `tmd_graph_tmd_actor`

Build the same TMD directed key graph, then execute graph subgoals with the
official TMD actor.  The actor receives raw medoid observations as goals, not
TMD embeddings.

### `tmd_full_gas_low`

Build the TMD graph and execute it with a low-level GAS-style policy conditioned
on precomputed TMD direction and distance features.  The TMD network stays
outside the low-level policy; `TMDGASDataset` provides `psi_obs`,
`psi_next_obs`, `psi_actor_goals`, and `tmd_actor_dist`.

## New Entry Points

- `pretrain_tmd.py`: trains the official TMD agent using GAS-compatible
  environment and checkpoint layout.
- `construct_graph_tmd.py`: calibrates TMD scales, constructs TMD medoid
  keynodes, builds a directed graph, and precomputes task paths.
- `evaluate_gas_tmd.py`: evaluates `tmd_graph_tmd_actor` and
  `tmd_graph_gas_policy`, plus `tmd_full_gas_low`.
- `train_policy_tmd_low.py`: trains the TMD-conditioned GAS low-level policy.
- `scripts/run_tmd_gas_smoke.sh`: smoke test for TMD training, graph
  construction, TMD actor evaluation, TMD-low training, and C-mode evaluation.
- `scripts/launch_tmd_gas_full_experiment.sh`: full multi-GPU experiment
  launcher.  Defaults are official-scale runs: 1,000,000 TMD steps and
  1,000,000 TMD-low policy steps on
  `antmaze-{medium,large}-{stitch,navigate}-v0` with seeds `0 1 2`.

All output run groups should use names such as `tmd_graph_*` or `tmd_actor_*`
to avoid overwriting GAS baselines.
