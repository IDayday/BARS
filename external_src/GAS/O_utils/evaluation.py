import cv2
import jax
import numpy as np

from tqdm import trange
from collections import defaultdict

from D_utils.kitchen_utils import kitchen_set_obs_and_goal, kitchen_render
from cage.state_machine import CAGEController
from cage.contract_tracing import SegmentContractRecorder
from cage.state_ref import capture_state_ref, serialize_state_ref


def evaluate_with_graph(agent, key_graph, env, env_name, task_id, eval_episodes, eval_video_episodes,
                        seed, eval_on_cpu, eval_subgoal_threshold, eval_final_goal_threshold, config, recompute_paths_per_episode=False,
                        cage_config=None, cage_trace_writer=None, contract_trace_writer=None, contract_config=None):
    """
    Evaluate GAS in the environment.
    In OGBench environments, the final goal includes slight random noise in each episode.
    Empirically, we found little to no performance difference between two strategies:
    (1) recomputing the shortest path every episode
    (2) reusing a precomputed path for the same task_id
    
    By default, evaluate_with_graph() uses strategy (2), computing the shortest path once via precompute_shortest_paths_to_all_tasks().
    When using strategy (2), we recommend setting eval_final_goal_threshold >= 2 to allow agents to reach slightly perturbed final goals.
    If final goals vary significantly across episodes, or if GAS is extended to online RL, we recommend setting recompute_paths_per_episode=True to use strategy (1).
    """
    eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0]) if eval_on_cpu else agent
    get_phi_fn = eval_agent.get_phi         
    actor_fn = supply_rng(eval_agent.sample_actions, rng=jax.random.PRNGKey(seed))  
    use_cage = bool(cage_config is not None and getattr(cage_config, "use_cage", False))
    distance_fn = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    contract_config = dict(contract_config or {})
    contract_variant = contract_config.get("variant", "cage" if use_cage else "gas")
    contract_enabled = bool(contract_trace_writer is not None and contract_config.get("enabled", False)
                            and contract_variant in set(contract_config.get("capture_variants", [])))
    
    stats = defaultdict(list)
    renders = []
    for i in trange(eval_episodes + eval_video_episodes, leave=False, desc=f"Task {task_id} Episodes"):
        step = 0
        render = []
        should_render = i >= eval_episodes
        eval_seed = seed + i  
        env, observation, goal, reward, done, goal_rendered = setup_task_env(env, env_name, task_id, should_render, seed=eval_seed)
        
        epsilon=1e-10
        phi_obs = np.array(get_phi_fn(observation))
        phi_goal = np.array(get_phi_fn(goal))
        final_goal_on = False
        cur_obs_goal = None
        # Optionally, recompute the shortest path every episode (strategy (1))
        if recompute_paths_per_episode:
            key_graph.precompute_shortest_paths_to_all_tasks({task_id: goal}, {task_id: phi_goal},)
        shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs, force_closest=True)
        cage_controller = None
        if use_cage:
            cage_controller = CAGEController(cage_config, distance_fn, logger=cage_trace_writer)
            cage_controller.reset_episode(phi_obs, phi_goal, initial_path=shortest_path)
        contract_recorder = SegmentContractRecorder(
            contract_trace_writer if contract_enabled else None,
            env_name=env_name,
            seed=seed,
            episode_idx=i,
            variant=contract_variant,
            task_id=task_id,
            config=contract_config,
        )
        while not done:
            phi_obs = np.array(get_phi_fn(observation))
            trace_info = None
            should_replan = False
            if use_cage:
                gas_obs_goal, gas_node_idx, final_goal_on, shortest_path = select_gas_subgoal(
                    key_graph=key_graph,
                    task_id=task_id,
                    phi_obs=phi_obs,
                    phi_goal=phi_goal,
                    shortest_path=shortest_path,
                    final_goal_on=final_goal_on,
                    eval_final_goal_threshold=eval_final_goal_threshold,
                    eval_subgoal_threshold=eval_subgoal_threshold,
                )
                cached_shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
                if cached_shortest_path is not None:
                    shortest_path = cached_shortest_path
                planner_final_goal_on = shortest_path is not None and len(shortest_path) <= eval_final_goal_threshold
                cage_info = {
                    "planner_final_goal_on": planner_final_goal_on,
                    "original_subgoal": gas_obs_goal,
                    "original_subgoal_index": gas_node_idx,
                    "env_name": env_name,
                    "task_id": task_id,
                    "seed": seed,
                    "episode_idx": i,
                }
                state_ref = maybe_capture_cage_state_ref(
                    env=env,
                    obs=observation,
                    phi=phi_obs,
                    env_name=env_name,
                    seed=seed,
                    episode_idx=i,
                    step=step,
                    variant="cage",
                    enabled=bool(getattr(cage_config, "debug", False) or getattr(cage_config, "contract_commit", False)),
                )
                if state_ref is not None:
                    cage_info["state_ref"] = state_ref
                cur_obs_goal, cur_node_idx, cage_state, should_replan, trace_info = cage_controller.select_subgoal(
                    current_state=phi_obs,
                    final_goal=phi_goal,
                    current_path=shortest_path,
                    current_subgoal=gas_obs_goal,
                    step=step,
                    info=cage_info,
                )
                if should_replan:
                    replanned_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs, force_closest=True)
                    if replanned_path is not None:
                        shortest_path = replanned_path
                        cage_info["after_global_replan"] = True
                        cage_info["planner_final_goal_on"] = len(shortest_path) <= eval_final_goal_threshold
                        cur_obs_goal, cur_node_idx, cage_state, should_replan, trace_info = cage_controller.select_subgoal(
                            current_state=phi_obs,
                            final_goal=phi_goal,
                            current_path=shortest_path,
                            current_subgoal=gas_obs_goal,
                            step=step,
                            info=cage_info,
                        )
                cage_controller.record_step_trace(env_name, task_id, seed, i, trace_info)
            else:
                cur_obs_goal, cur_node_idx, final_goal_on, shortest_path = select_gas_subgoal(
                    key_graph=key_graph,
                    task_id=task_id,
                    phi_obs=phi_obs,
                    phi_goal=phi_goal,
                    shortest_path=shortest_path,
                    final_goal_on=final_goal_on,
                    eval_final_goal_threshold=eval_final_goal_threshold,
                    eval_subgoal_threshold=eval_subgoal_threshold,
                )

            target_source = infer_target_source(use_cage, trace_info, final_goal_on)
            contract_recorder.maybe_switch(
                env=env,
                obs=observation,
                phi=phi_obs,
                target_phi=cur_obs_goal,
                step=step,
                target_idx=cur_node_idx,
                target_source=target_source,
                path_position=cur_node_idx,
                final_phase=bool(final_goal_on or (trace_info or {}).get("final_goal_phase", False)),
                recovery_active=str((trace_info or {}).get("cage_state", "")).upper() == "RECOVERY",
                path_phi=shortest_path,
                final_goal_phi=phi_goal,
            )
            contract_recorder.record_cage_trace(trace_info, should_replan=should_replan)
            skills = (cur_obs_goal - phi_obs) / (np.linalg.norm(cur_obs_goal - phi_obs) + epsilon)  
            action = actor_fn(observations=observation, goals=skills, temperature=0.0)
            action = np.clip(np.array(action), -1, 1)
            contract_recorder.record_action(action, skill_norm=float(np.linalg.norm(skills)))
            next_observation, reward, done, info = env_step(env, env_name, action)  
            if use_cage:
                next_phi_obs = np.array(get_phi_fn(next_observation))
                cage_controller.update_after_step(phi_obs, next_phi_obs, cur_obs_goal, action=action, env_info=info)
            else:
                next_phi_obs = np.array(get_phi_fn(next_observation))

            step += 1
            reached = float(np.linalg.norm(next_phi_obs - cur_obs_goal)) <= float(eval_subgoal_threshold)
            if reached:
                contract_recorder.close(env=env, obs=next_observation, phi=next_phi_obs, step=step, release_reason="reached", reached_target=True)
            if should_render and (step % 3 == 0 or done):
                frame = get_frame(env, env_name)
                render.append(frame)
            observation = next_observation
        phi_obs = np.array(get_phi_fn(observation))
        contract_recorder.close(env=env, obs=observation, phi=phi_obs, step=step, release_reason="episode_end", reached_target=None)
        flat_info = flatten(info)
        add_to(stats, flat_info)
        if use_cage:
            cage_controller.finish_episode(env_name, task_id, seed, i, flat_info)
        if should_render:
            renders.append(np.array(render))
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, renders


def infer_target_source(use_cage, trace_info, final_goal_on):
    if final_goal_on:
        return "final_goal"
    if not use_cage:
        return "gas_path"
    trace_info = trace_info or {}
    if trace_info.get("cage_trace_only") or trace_info.get("fallback_to_gas_active"):
        return "fallback_to_gas" if trace_info.get("fallback_to_gas_active") else "gas_path"
    state = str(trace_info.get("cage_state", "")).upper()
    if state == "RECOVERY":
        return "recovery"
    return "cage_selected"


def maybe_capture_cage_state_ref(env, obs, phi, env_name, seed, episode_idx, step, variant, enabled):
    if not enabled:
        return None
    try:
        ref = capture_state_ref(
            env,
            obs=obs,
            phi=phi,
            metadata={
                "env_name": env_name,
                "seed": seed,
                "episode_idx": episode_idx,
                "step_idx": step,
                "source_variant": variant,
                "source": "cage_debug_step",
            },
        )
        return serialize_state_ref(ref)
    except Exception as exc:
        return {"reset_mode": "unsupported", "exact_reset": False, "failure_reason": str(exc)}


def select_gas_subgoal(
    key_graph,
    task_id,
    phi_obs,
    phi_goal,
    shortest_path,
    final_goal_on,
    eval_final_goal_threshold,
    eval_subgoal_threshold,
):
    """Original GAS subgoal selection logic, factored for trace-only CAGE."""
    if final_goal_on:
        return phi_goal, -1, final_goal_on, shortest_path
    cached_shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
    if cached_shortest_path is not None:
        shortest_path = cached_shortest_path
    distances = np.linalg.norm(np.array(shortest_path) - phi_obs, axis=1)
    valid_indices = np.where(distances <= eval_subgoal_threshold)[0]
    cur_node_idx = valid_indices[-1] if len(valid_indices) > 0 else 0
    if len(shortest_path) <= eval_final_goal_threshold:
        final_goal_on = True
        return phi_goal, -1, final_goal_on, shortest_path
    return shortest_path[cur_node_idx], cur_node_idx, final_goal_on, shortest_path


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped


def setup_task_env(env, env_name, task_id, should_render, seed):
    """Reset the environment for a specific task."""
    if env_name in ['kitchen-partial-v0',]:
        env, observation, goal = kitchen_set_obs_and_goal(env, env_name, task_id, seed=seed)
        goal_rendered = None
    else:
        observation, info = env.reset(seed=seed, options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        if should_render:
            goal_rendered =  info.get('goal_rendered') 
            goal_rendered = resize_frame(goal_rendered)
        else:
            goal_rendered = None
    reward = 0
    done = False
    return env, observation, goal, reward, done, goal_rendered


def env_step(env, env_name, action):
    """Step the environment once."""
    if env_name in ['kitchen-partial-v0']:
        next_observation, reward, done, info = env.step(action)
        next_observation = next_observation[:30]
    else:
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return next_observation, reward, done, info


def resize_frame(frame):
    """Resize a frame"""
    resized_frame = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_LINEAR)
    return resized_frame


def get_frame(env, env_name):
    """Render a frame from the environment."""
    if env_name in ['kitchen-partial-v0',]:
        frame = kitchen_render(env, wh=200)
    else:
        frame = np.ascontiguousarray(env.render())
        frame = resize_frame(frame)
    return frame


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)
                  
