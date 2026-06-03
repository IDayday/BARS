# Stage31 Wide Official GAS Behavior Atlas

Status: OFFICIAL_GAS_WIDE_BEHAVIOR_ATLAS.
Pre-Stage30 BARS/Stage28/Stage29 evidence: ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE.
Official GAS graph, planner, policy, subgoal selection, and action outputs are unchanged.
This wide pass uses episode-level instrumentation only; no same/cross/dt semantics are inferred without exact official provenance, and no edge-probe labels are assigned here.

## Aggregate

- episodes: 9000
- failed episodes: 2304
- unresolved failures / failed episodes: 0/2304 (0.0000)

## Failure-Dense Envs

| env_name | episodes | success_rate | no_path_rate | timeout_rate | stuck_rate | divergence_rate | mean_subgoal_reach_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| humanoidmaze-large-navigate-v0 | 750 | 0.2013 | 0.0000 | 0.7987 | 0.2507 | 0.1507 | 0.0491 |
| antmaze-teleport-explore-v0 | 750 | 0.3600 | 0.0000 | 0.6400 | 0.0107 | 0.0080 | 0.1281 |
| antmaze-teleport-navigate-v0 | 750 | 0.3853 | 0.0000 | 0.6147 | 0.2947 | 0.2693 | 0.1445 |
| antmaze-teleport-stitch-v0 | 750 | 0.4667 | 0.0000 | 0.5333 | 0.1093 | 0.0987 | 0.0784 |
| antmaze-giant-navigate-v0 | 750 | 0.8493 | 0.0000 | 0.1520 | 0.0307 | 0.0240 | 0.0241 |
| antmaze-giant-stitch-v0 | 750 | 0.9053 | 0.0000 | 0.0947 | 0.0213 | 0.0173 | 0.0223 |
| antmaze-large-explore-v0 | 750 | 0.9507 | 0.0000 | 0.0493 | 0.0307 | 0.0280 | 0.0269 |
| antmaze-large-stitch-v0 | 750 | 0.9520 | 0.0000 | 0.0480 | 0.0200 | 0.0187 | 0.0344 |
| antmaze-medium-stitch-v0 | 750 | 0.9547 | 0.0000 | 0.0453 | 0.0200 | 0.0173 | 0.0608 |
| antmaze-large-navigate-v0 | 750 | 0.9573 | 0.0000 | 0.0427 | 0.0213 | 0.0173 | 0.0350 |

## Failure Phase Counts

- keygraph_subgoal: 1615
- final_goal_phase: 689

## Failure Label Counts

- SUBGOAL_SEQUENCE_DRIFT: 1184
- GOAL_INTERFACE_FAILURE: 689
- POLICY_LOCAL_FAILURE: 431

## Files

- all episodes: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_all_episode_traces.csv`
- all path edges: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_all_path_edges.csv`
- summary by env: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_success_by_env.csv`
- failure phases: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_failure_phase_summary.csv`
- failure labels: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_failure_label_summary.csv`
- path dynamics: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_path_dynamics_by_env.csv`
- task sensitivity: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_task_sensitivity.csv`
- compute manifest: `runs_stage31_official_gas/combined_wide_plus_incremental_active/global/stage31_compute_manifest.csv`
