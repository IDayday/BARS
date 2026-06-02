# Stage30 Official GAS Reset Summary

Status: `OFFICIAL_GAS_FIRST_RESET`.

The graph-method research pipeline is reset to official GAS first. BARS, Stage28, and Stage29 results are archived as internal exploration only and are not evidence about GAS.

## Baseline Certification

Evidence file: `runs_stage30_official_gas/reproduction/official_gas_eval.csv`

Report file: `runs_stage30_official_gas/reproduction/reproduction_report.md`

Local official GAS artifacts, seeds 44-46:

| env_name | official GAS success mean | 95% CI |
| --- | ---: | --- |
| antmaze-medium-navigate-v0 | 0.9827 | [0.9688, 0.9965] |
| antmaze-medium-stitch-v0 | 0.9613 | [0.9509, 0.9718] |
| antmaze-large-navigate-v0 | 0.9413 | [0.9319, 0.9508] |
| antmaze-large-stitch-v0 | 0.9440 | [0.9304, 0.9576] |
| antmaze-large-explore-v0 | 0.9627 | [0.9287, 0.9966] |
| antmaze-medium-explore-v0 | 0.9693 | [0.9399, 0.9988] |
| antmaze-giant-navigate-v0 | 0.8533 | [0.8260, 0.8806] |
| antmaze-giant-stitch-v0 | 0.8840 | [0.8284, 0.9396] |

`evaluate_gas.py` run-smoke also passes on a copied artifact for `antmaze-medium-navigate-v0`, seed 44, 1 episode per official task, with success 1.0 and returncode 0.

## Identity Checks

Artifact identity file: `runs_stage30_official_gas/reproduction/artifact_identity.csv`

The vendored GAS source has no `.git` directory, so the official repo SHA is recorded as `UNAVAILABLE_IN_VENDOR_COPY`. The source tree hash is recorded in the identity CSV. Keygraph, policy, and TDR hashes are recorded per env/seed.

Official `evaluate_gas.py` uses OGBench's hard-coded default dataset directory. Stage30 run mode creates symlinks from `~/.ogbench/data/<env>.npz` and `-val.npz` to `/mnt/project/offlinerl_datasets/ogbench` before running, so no dataset download is required and official GAS source remains unchanged.

## Instrumentation Smoke

Evidence files:

- `runs_stage30_official_gas/instrumentation_smoke/official_gas_episode_traces.csv`
- `runs_stage30_official_gas/instrumentation_smoke/official_gas_path_edges.csv`
- `runs_stage30_official_gas/instrumentation_smoke/instrumentation_report.md`

The smoke trace mirrors official `evaluate_with_graph` and logs path nodes/edges, subgoal progress, no-path, timeout, stuck, divergence, and final goal distance. It does not modify official planner, policy, graph, or action outputs.

## Edge Probe Smoke

Strict node-to-dataset exact mapping is unavailable for official keygraph nodes. Nearest-state diagnostic mapping is possible only with explicit tolerance and logs embedding match distance. This probe remains diagnostic, not promotion evidence.

Evidence files:

- `runs_stage30_official_gas/edge_probe_smoke/edge_probe_report.md`
- `runs_stage30_official_gas/edge_probe_smoke_nearest/edge_probe_report.md`

## Next Gate

Run full official-GAS-only instrumentation and probe before any algorithm change:

- official instrumentation over supported official state-based envs and seeds
- at least 200 valid edges per meaningful category where recoverable
- failure taxonomy only when episode evidence and edge probe evidence both support the label

No new algorithm should be implemented until official GAS audit identifies a dominant failure mode.
