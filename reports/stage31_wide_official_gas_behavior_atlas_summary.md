# Stage31 Wide Official GAS Behavior Atlas Summary

Status: OFFICIAL_GAS_WIDE_BEHAVIOR_ATLAS.

Scope: existing official GAS artifacts only. BARS, Stage28, and Stage29 results are archived internal exploration and are not used as GAS evidence.

Run: 8 READY antmaze OGBench envs x seeds 44-46 x task_ids 1-5 x 50 episodes = 6000 official GAS episodes. The official graph, planner, policy, subgoal selection, and action outputs were unchanged. No edge probes were run in this wide pass.

## Artifact Coverage

- READY_OFFICIAL_GAS: 24 env/seed rows, covering antmaze medium/large/giant navigate/stitch/explore where policy and keygraph exist.
- Missing checkpoint queue: 75 rows. Most have local data and TDR/keygraph but no policy; kitchen has local D4RL data and is marked D4RL protocol/training pending.
- Tier3 visual artifacts are not evaluated yet because required official checkpoint pieces are missing.
- Policy-only missing state OGBench artifacts selected for backfill training: 48 rows. A background official GAS policy-training queue was launched with two GPU workers. It excludes visual and D4RL/kitchen artifacts by default.

## Wide Result

| env | episodes | success | no_path | timeout | stuck | divergence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antmaze-medium-navigate-v0 | 750 | 0.9747 | 0.0000 | 0.0253 | 0.0093 | 0.0093 |
| antmaze-medium-stitch-v0 | 750 | 0.9547 | 0.0000 | 0.0453 | 0.0200 | 0.0173 |
| antmaze-medium-explore-v0 | 750 | 0.9707 | 0.0000 | 0.0293 | 0.0160 | 0.0120 |
| antmaze-large-navigate-v0 | 750 | 0.9573 | 0.0000 | 0.0427 | 0.0213 | 0.0173 |
| antmaze-large-stitch-v0 | 750 | 0.9520 | 0.0000 | 0.0480 | 0.0200 | 0.0187 |
| antmaze-large-explore-v0 | 750 | 0.9507 | 0.0000 | 0.0493 | 0.0307 | 0.0280 |
| antmaze-giant-navigate-v0 | 750 | 0.8493 | 0.0000 | 0.1520 | 0.0307 | 0.0240 |
| antmaze-giant-stitch-v0 | 750 | 0.9053 | 0.0000 | 0.0947 | 0.0213 | 0.0173 |

## Diagnosis

Official GAS does not show an initial graph-connectivity failure on these ready artifacts: `no_path_rate=0` for every evaluated env. The failure-dense envs are giant navigate and giant stitch.

Among 364 failed episodes, the wide pass assigns:

- keygraph_subgoal phase: 293 failures
- final_goal_phase: 71 failures
- SUBGOAL_SEQUENCE_DRIFT: 166 failures
- POLICY_LOCAL_FAILURE: 127 failures
- GOAL_INTERFACE_FAILURE: 71 failures
- UNRESOLVED_FAILURE / failed episodes: 0/364

The strongest diagnostic signal is cached replanning miss after rollout drift. `cached_path_miss_count` is nearly zero for successes, but high for failures in harder envs: giant navigate 106.7, giant stitch 110.1, large navigate 240.5, and large stitch 280.7 mean misses per failed episode. This points to a failure mode where official GAS starts with a valid path, but execution drifts away from keygraph-supported source regions and repeated replanning misses until timeout.

Task sensitivity is also concentrated. Giant navigate task 1 is the sharpest pocket: success 0.607 vs task 5 success 0.973. Giant stitch is less extreme but still task-sensitive.

## Targeted Probe Addendum

I also ran a targeted nearest-execution probe on giant navigate/stitch seeds 44-46 with 50 edges per available category. This probe is execution-only: `trajectory_semantics_valid=0`, so it cannot support same/cross/dt claims.

Aggregated nearest-probe reach rates:

| env | category | valid | reach | mean_progress |
| --- | --- | ---: | ---: | ---: |
| antmaze-giant-navigate-v0 | first_failed_failure_path_edges | 61 | 0.213 | 0.029 |
| antmaze-giant-navigate-v0 | frequently_used_success_path_edges | 150 | 0.100 | 0.044 |
| antmaze-giant-navigate-v0 | high_use_edges | 148 | 0.122 | 0.039 |
| antmaze-giant-stitch-v0 | first_failed_failure_path_edges | 53 | 0.019 | 0.014 |
| antmaze-giant-stitch-v0 | frequently_used_success_path_edges | 136 | 0.015 | 0.021 |
| antmaze-giant-stitch-v0 | high_use_edges | 132 | 0.030 | 0.026 |

This does not prove that first-failed edges are uniquely worse than success-path edges. The low reach across success-path and high-use categories means the nearest probe is not yet promotion-grade as a calibrated single-edge executability test for giant envs. The safer interpretation is that the wide run identified a drift/replanning-miss failure pocket, and targeted probes now need a sanity-calibrated goal interface, horizon, and reset/set_state protocol before causal edge labels are assigned.

## Next

Run targeted high-resolution tracing on giant-navigate task 1 and giant-stitch tasks 1/3 with per-step distance-to-current-subgoal, cached path miss windows, and active path update events. Recalibrate the edge execution probe on edges that are known to be traversed successfully in official rollout before assigning causal edge labels. Do not implement a planner modification until those targeted probes distinguish policy-local failure, subgoal sequence drift, goal-interface mismatch, and keygraph abstraction loss.

When the background policy backfill queue finishes new official GAS policies, rerun artifact inventory and evaluate only newly READY env/seed rows with the same Stage31 protocol. Those results should be analyzed as a separate expansion batch, not merged silently into this antmaze-ready atlas.

Large raw files are kept local under `runs_stage31_official_gas/wide_atlas_active/global/` and should not be committed, especially `stage31_all_path_edges.csv`.
