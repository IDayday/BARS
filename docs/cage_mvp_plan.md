# CAGE-MVP Implementation Plan

## Scope

CAGE-MVP is an execution-interface layer for official GAS evaluation. It does not retrain TDR, rebuild the keygraph differently, modify policy training, or change default GAS behavior. The new logic is enabled only with `--use_cage`.

## Existing Hook Points

The evaluator entrypoint is `external_src/GAS/evaluate_gas.py`. It calls `O_utils.evaluation.evaluate_with_graph(...)` once per task.

The exact graph-to-subgoal hook is in `external_src/GAS/O_utils/evaluation.py`:

- Initial path is created at `shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs, force_closest=True)`.
- Cached replanning happens at `cached_shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs)`.
- The path is converted to the active subgoal at:
  - `distances = np.linalg.norm(np.array(shortest_path) - phi_obs, axis=1)`
  - `valid_indices = np.where(distances <= eval_subgoal_threshold)[0]`
  - `cur_node_idx = valid_indices[-1] if len(valid_indices) > 0 else 0`
  - `cur_obs_goal = shortest_path[cur_node_idx]`
- Final goal mode is triggered by `len(shortest_path) <= eval_final_goal_threshold`, then `cur_obs_goal = phi_goal`.

The exact low-level policy hook is immediately after subgoal selection:

- `skills = (cur_obs_goal - phi_obs) / (np.linalg.norm(cur_obs_goal - phi_obs) + epsilon)`
- `action = actor_fn(observations=observation, goals=skills, temperature=0.0)`

CAGE will wrap only the `cur_obs_goal` selection before the skill vector is computed.

## Available Variables

At every evaluator step:

- `observation`: raw current environment observation sent to the actor.
- `goal`: raw final task goal from environment reset.
- `phi_obs`: current TDR embedding from `get_phi_fn(observation)`.
- `phi_goal`: final goal TDR embedding from `get_phi_fn(goal)`.
- `shortest_path`: current active graph path as an array/list of TDR nodes.
- `cached_shortest_path`: optional refreshed path returned by `key_graph.get_shortest_path(...)`.
- `cur_obs_goal`: selected TDR subgoal currently sent to the low-level policy.
- `step`: current episode step.
- `task_id`, `env_name`, `seed`, and episode index `i`.
- `info`: final environment info available at episode end; flattened into existing stats.

The keygraph object exposes:

- `key_graph.nodes`: TDR node array.
- `key_graph.graph`: NetworkX directed graph.
- `key_graph.task_paths_dict` and `task_paths_dist_dict`: precomputed task shortest paths.
- `key_graph.get_shortest_path(task_id, source, force_closest=False)`.

The MVP distance function is Euclidean distance in TDR space:

```python
distance_fn = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
```

`distance_to_path` will be computed as `min(distance_fn(phi_obs, node) for node in shortest_path)` when a path exists.

## CAGE Execution Behavior

When `--use_cage` is false, `evaluate_with_graph` follows the existing branch exactly and no CAGE objects are constructed.

When enabled:

1. Construct a `CAGEController` with a `CAGEConfig` and the TDR Euclidean `distance_fn`.
2. Call `reset_episode(phi_obs, phi_goal, initial_path=shortest_path)` after the initial path is computed.
3. Each step still asks GAS for `cached_shortest_path`; if present, it updates `shortest_path`.
4. Pass `phi_obs`, `phi_goal`, `shortest_path`, current subgoal, step, and small info dict into `select_subgoal`.
5. Use CAGE's returned `selected_subgoal` as `cur_obs_goal`.
6. Compute the existing skill vector and call `actor_fn` unchanged.
7. Call `update_after_step(phi_obs, next_phi_obs, selected_subgoal, action, info)` after the environment step.
8. At episode end, append one JSONL trace record and optional step-level records.

`should_replan` is traced in this milestone. It does not force a new graph construction or alter GAS path search beyond the existing cached shortest-path query.

## Default Flag Values

Conservative defaults:

- `--use_cage=false`
- `--cage_trace_path=""`
- `--cage_min_commit_steps=8`
- `--cage_stall_window=8`
- `--cage_progress_eps=0.01`
- `--cage_drift_threshold=16.0`
- `--cage_max_subgoal_dist=24.0`
- `--cage_min_subgoal_dist=2.0`
- `--cage_recovery_commit_steps=12`
- `--cage_max_recovery_attempts=2`
- `--cage_recovery_suffix_weight=0.25`
- `--cage_final_phase_dist=8.0`
- `--cage_final_min_commit_steps=12`
- `--cage_debug=false`

If `env_name` contains `humanoid`, CAGE will internally reduce max horizon and increase commitment conservatively.

## Trace Output

Default trace path when CAGE is enabled and `--cage_trace_path` is empty:

```text
<FLAGS.save_eval_dir>/cage_trace.jsonl
```

Episode-level records are JSON lines with:

- `env_name`, `task_id`, `seed`, `episode_idx`
- `success`, `return`, `normalized_score`
- `no_path`, `timeout`
- `path_length`, `initial_path_length`, `final_active_path_length`
- `target_switch_count`, `early_switch_count`, `mean_commitment_length`
- `stall_count`, `drift_count`
- `recovery_attempt_count`, `recovery_success_count`, `recovery_failure_count`
- `global_replan_request_count`
- `final_goal_on_step`, `final_goal_switch_count`, `final_goal_stall_count`
- `segment_target_reach_rate`, `mean_segment_progress`, `mean_distance_to_path`

When `--cage_debug` is enabled, step-level JSON lines are also emitted with:

- `record_type="step"`
- `step`, `cage_state`, `selected_subgoal_idx`, `selected_subgoal_distance`
- `distance_to_path`, `progress_window_value`, `should_replan`
- `recovery_target_idx`, `final_goal_phase`

Existing CSV and WandB/TensorBoard outputs are unchanged.

## Minimal Files To Modify

New files:

- `external_src/GAS/cage/__init__.py`
- `external_src/GAS/cage/config.py`
- `external_src/GAS/cage/state_machine.py`
- `external_src/GAS/cage/monitor.py`
- `external_src/GAS/cage/subgoal_selector.py`
- `external_src/GAS/cage/recovery.py`
- `external_src/GAS/cage/tracing.py`
- `scripts/summarize_cage_traces.py`
- `tests/test_cage_state_machine.py`
- `tests/test_cage_subgoal_selector.py`
- `docs/cage_mvp_usage.md`

Modified files:

- `external_src/GAS/evaluate_gas.py`: add flags and pass optional CAGE config to evaluator.
- `external_src/GAS/O_utils/evaluation.py`: insert the optional CAGE wrapper at the existing subgoal hook.

No training, graph construction, TDR, or policy-training files are modified.
