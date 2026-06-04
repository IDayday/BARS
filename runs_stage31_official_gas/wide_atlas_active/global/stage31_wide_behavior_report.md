# Stage31 Wide Official GAS Behavior Atlas

Status: OFFICIAL_GAS_WIDE_BEHAVIOR_ATLAS.
Pre-Stage30 BARS/Stage28/Stage29 evidence: ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE.
Official GAS graph, planner, policy, subgoal selection, and action outputs are unchanged.
This wide pass uses episode-level instrumentation only; no same/cross/dt semantics are inferred without exact official provenance, and no edge-probe labels are assigned here.

## Aggregate

- episodes: 6000
- failed episodes: 364
- unresolved failures / failed episodes: 0/364 (0.0000)

## Failure-Dense Envs

| env_name | episodes | success_rate | no_path_rate | timeout_rate | stuck_rate | divergence_rate | mean_subgoal_reach_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-giant-navigate-v0 | 750 | 0.8493 | 0.0000 | 0.1520 | 0.0307 | 0.0240 | 0.0241 |
| antmaze-giant-stitch-v0 | 750 | 0.9053 | 0.0000 | 0.0947 | 0.0213 | 0.0173 | 0.0223 |
| antmaze-large-explore-v0 | 750 | 0.9507 | 0.0000 | 0.0493 | 0.0307 | 0.0280 | 0.0269 |
| antmaze-large-stitch-v0 | 750 | 0.9520 | 0.0000 | 0.0480 | 0.0200 | 0.0187 | 0.0344 |
| antmaze-medium-stitch-v0 | 750 | 0.9547 | 0.0000 | 0.0453 | 0.0200 | 0.0173 | 0.0608 |
| antmaze-large-navigate-v0 | 750 | 0.9573 | 0.0000 | 0.0427 | 0.0213 | 0.0173 | 0.0350 |
| antmaze-medium-explore-v0 | 750 | 0.9707 | 0.0000 | 0.0293 | 0.0160 | 0.0120 | 0.0498 |
| antmaze-medium-navigate-v0 | 750 | 0.9747 | 0.0000 | 0.0253 | 0.0093 | 0.0093 | 0.0605 |

## Failure Phase Counts

- keygraph_subgoal: 293
- final_goal_phase: 71

## Failure Label Counts

- SUBGOAL_SEQUENCE_DRIFT: 166
- POLICY_LOCAL_FAILURE: 127
- GOAL_INTERFACE_FAILURE: 71

## Files

- all episodes: `runs_stage31_official_gas/wide_atlas_active/global/stage31_all_episode_traces.csv`
- all path edges: `runs_stage31_official_gas/wide_atlas_active/global/stage31_all_path_edges.csv`
- summary by env: `runs_stage31_official_gas/wide_atlas_active/global/stage31_success_by_env.csv`
- failure phases: `runs_stage31_official_gas/wide_atlas_active/global/stage31_failure_phase_summary.csv`
- failure labels: `runs_stage31_official_gas/wide_atlas_active/global/stage31_failure_label_summary.csv`
- path dynamics: `runs_stage31_official_gas/wide_atlas_active/global/stage31_path_dynamics_by_env.csv`
- task sensitivity: `runs_stage31_official_gas/wide_atlas_active/global/stage31_task_sensitivity.csv`
- compute manifest: `runs_stage31_official_gas/wide_atlas_active/global/stage31_compute_manifest.csv`
