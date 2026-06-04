# Stage31 Wide Plus Incremental Official GAS Summary

Status: OFFICIAL_GAS_WIDE_PLUS_INCREMENTAL_ATLAS.
Date: 2026-06-03.
Evidence class: official GAS instrumentation only.
Pre-Stage30 BARS/Stage28/Stage29 evidence: ARCHIVED_INTERNAL_EXPLORATION_NOT_GAS_EVIDENCE.

Official GAS graph, planner, policy, subgoal selection, and action outputs were left unchanged. The run used fallback_mode=none and EVAL_ON_CPU=0.

## Coverage

- Combined OGBench atlas: 12 envs, seeds 44-46, task_id 1-5, 50 episodes per task_id, 9000 episodes total.
- Original ready set: 8 antmaze medium/large/giant envs, 6000 episodes.
- Incremental ready set: antmaze-teleport-explore/navigate/stitch and humanoidmaze-large-navigate, 3000 episodes.
- D4RL local inventory: 25 datasets detected. Official GAS supports a kitchen-partial path, but kitchen is not yet evaluable in this environment because D4RL kitchen env construction fails under mujoco 3.2.7 / dm-control 1.0.27 with `top-level default class 'main' cannot be renamed`.

## Aggregate Result

- Combined episodes: 9000.
- Combined failures: 2304.
- Combined unresolved failures / failed episodes: 0/2304.
- Combined no_path_rate: 0.0 for all 12 evaluated envs.

The wide atlas therefore does not support a dominant `NO_OFFICIAL_GRAPH_PATH` explanation on the current official GAS artifacts. The current failure-dense evidence points to rollout/subgoal/final-goal execution failures after an official graph path exists.

## Failure-Dense Envs

| env_name | episodes | success_rate | no_path_rate | main failure signal |
| --- | ---: | ---: | ---: | --- |
| humanoidmaze-large-navigate-v0 | 750 | 0.2013 | 0.0000 | mixed subgoal drift, local policy failure, final-goal failure |
| antmaze-teleport-explore-v0 | 750 | 0.3600 | 0.0000 | keygraph subgoal drift |
| antmaze-teleport-navigate-v0 | 750 | 0.3853 | 0.0000 | final-goal interface failure plus subgoal drift |
| antmaze-teleport-stitch-v0 | 750 | 0.4667 | 0.0000 | keygraph subgoal drift plus final-goal failure |

Standard antmaze medium/large/giant remains high-success in the same protocol: medium/large envs are 0.9507-0.9747 success, giant navigate is 0.8493, and giant stitch is 0.9053.

## Failure Taxonomy

Combined failure labels:

- SUBGOAL_SEQUENCE_DRIFT: 1184.
- GOAL_INTERFACE_FAILURE: 689.
- POLICY_LOCAL_FAILURE: 431.

Combined failure phases:

- keygraph_subgoal: 1615.
- final_goal_phase: 689.

Incremental failure labels:

- antmaze-teleport-explore: 472/480 failures are SUBGOAL_SEQUENCE_DRIFT.
- antmaze-teleport-navigate: 283/461 failures are GOAL_INTERFACE_FAILURE.
- antmaze-teleport-stitch: 210/400 failures are SUBGOAL_SEQUENCE_DRIFT and 137/400 are GOAL_INTERFACE_FAILURE.
- humanoidmaze-large-navigate: failures are split across SUBGOAL_SEQUENCE_DRIFT 214/599, GOAL_INTERFACE_FAILURE 198/599, and POLICY_LOCAL_FAILURE 187/599.

## Interpretation

The failure mode broadened after adding teleport and humanoidmaze. The official GAS graph usually finds a path, but path existence is not enough in harder dynamics/task distributions. The evidence now separates two regimes:

- Standard antmaze: strong official GAS performance with failures correlated with cached-path misses and replanning drift.
- Teleport/humanoidmaze: failure-dense behavior with no graph no-path, dominated by timeout, subgoal sequence drift, final-goal interface failure, and local policy failure.

This does not justify a generic planner replacement. The next evidence step should be targeted high-resolution tracing/probes on the four failure-dense envs, especially first-failed subgoal segments and final-goal interface transitions.

## D4RL Status

The local D4RL kitchen dataset is present and symlinked for D4RL's default cache path:

- `/mnt/project/offlinerl_datasets/d4rl/kitchen_microwave_kettle_light_slider-v0.hdf5`
- `~/.d4rl/datasets/kitchen_microwave_kettle_light_slider-v0.hdf5`

However, `gym.make("kitchen-partial-v0")` currently fails before GAS training/evaluation can start because of a MuJoCo XML compatibility error in the D4RL kitchen environment. D4RL kitchen remains ENV_COMPAT_BLOCKED, not a GAS result. D4RL locomotion datasets are local data inventory only because the official GAS code path is kitchen-specific.

## Committed Small Results

Small report/result files intended for git:

- `reports/stage31_wide_plus_incremental_behavior_report.md`
- `reports/stage31_wide_plus_incremental_success_by_env.csv`
- `reports/stage31_wide_plus_incremental_failure_label_summary.csv`
- `reports/stage31_wide_plus_incremental_failure_phase_summary.csv`
- `reports/stage31_wide_plus_incremental_path_dynamics_by_env.csv`
- `reports/stage31_wide_plus_incremental_task_sensitivity.csv`
- `reports/stage31_wide_plus_incremental_compute_manifest.csv`

Large raw files are intentionally excluded from git, especially `stage31_all_path_edges.csv` and checkpoint/raw training outputs.

## Next Experiments

1. Continue state OGBench policy backfill; after each new READY batch, rerun incremental official GAS instrumentation and merge into this atlas.
2. Run targeted high-resolution segment tracing on teleport and humanoidmaze failure-dense envs.
3. Fix D4RL kitchen environment compatibility in an isolated way before launching official GAS kitchen TDR/keygraph/policy training.
4. Do not implement algorithm modifications until the official GAS-only evidence identifies a stable dominant failure mechanism with segment/probe support.
