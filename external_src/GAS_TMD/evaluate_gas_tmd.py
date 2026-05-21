import json
import os
import platform
import random
import sys
from collections import defaultdict

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_index = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
print(f"\033[38;5;208m{'=' * 14}\n Using GPU: {gpu_index}\n{'=' * 14}\033[0m")

if "mac" not in platform.platform():
    os.environ.setdefault("MUJOCO_GL", "egl")
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm, trange

from K_utils.keygraph_tmd_utils import TMDKeyGraph
from M_utils.agents import agents_dict
from M_utils.flax_utils import restore_agent as restore_gas_agent
from O_utils.datasets import Dataset
from O_utils.env_utils import make_env_and_datasets
from O_utils.log_utils import CsvLogger, get_exp_name, setup_save_directory, setup_wandb, wandb
from R_utils.checkpoint_utils import split_checkpoint_path
from R_utils.json_utils import json_safe
from R_utils.path_selection import select_reachable_path_node
from R_utils.repr_provider import TMDRepresentationProvider
from TMD_utils.tmd_agent import TMDAgent
from TMD_utils.tmd_datasets import Dataset as TMDDataset
from TMD_utils.tmd_datasets import GCDataset as TMDGCDataset
from TMD_utils.tmd_flax_utils import restore_agent as restore_tmd_agent

FLAGS = flags.FLAGS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_enum(
    "mode",
    "tmd_graph_tmd_actor",
    ["tmd_graph_gas_policy", "tmd_graph_tmd_actor", "tmd_full_gas_low"],
    "Evaluation mode.",
)
flags.DEFINE_string("run_eval_project", "TMD_GAS", "Run project.")
flags.DEFINE_string("run_group", "tmd_graph_eval", "Run group.")
flags.DEFINE_string("env_name", "antmaze-medium-stitch-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("gpu", 0, "GPU index.")
flags.DEFINE_string("save_eval_dir", "exp_eval_tmd/", "Save directory.")

flags.DEFINE_string("keygraph_tmd_path", None, "Path to keygraph_tmd.pkl.")
flags.DEFINE_string("tmd_path", None, "Pretrained TMD params path or checkpoint directory.")
flags.DEFINE_string("gas_policy_path", None, "Pretrained GAS policy params path or checkpoint directory.")
flags.DEFINE_string("gas_tdr_path", None, "Optional GAS TDR path, recorded for provenance only.")
flags.DEFINE_string("tmd_low_policy_path", None, "Pretrained TMD-conditioned GAS low policy params path.")

flags.DEFINE_integer("eval_episodes", 20, "Number of episodes per task.")
flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate.")
flags.DEFINE_integer("eval_max_steps", None, "Optional max steps per episode for smoke tests.")
flags.DEFINE_float("eval_temperature", 0.0, "Actor sampling temperature.")
flags.DEFINE_enum("eval_subgoal_threshold_mode", "tmd_distance", ["tmd_distance", "repr_l2"], "Subgoal reachability mode.")
flags.DEFINE_bool("force_closest_node", False, "Use closest graph node when no node is within the TMD edge threshold.")
flags.DEFINE_bool("direct_goal_after_progress", False, "Switch to final goal after the first reachable path progress.")
flags.DEFINE_integer("eval_final_goal_threshold", 2, "Path length threshold to switch to final goal.")

config_flags.DEFINE_config_file(
    "tmd_agent_config",
    os.path.join(SCRIPT_DIR, "TMD_utils", "tmd_agent.py"),
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "gas_agent_config",
    os.path.join(SCRIPT_DIR, "M_utils", "agents", "gas.py"),
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "tmd_low_agent_config",
    os.path.join(SCRIPT_DIR, "M_utils", "agents", "gas_tmd_low.py"),
    lock_config=False,
)


def normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def setup_task_env(env, env_name, task_id, seed):
    if env_name in ["kitchen-partial-v0"]:
        from D_utils.kitchen_utils import kitchen_set_obs_and_goal

        env, observation, goal = kitchen_set_obs_and_goal(env, env_name, task_id, seed=seed)
    else:
        observation, info = env.reset(seed=seed, options=dict(task_id=task_id, render_goal=False))
        goal = info.get("goal")
    return env, observation, goal, 0.0, False


def env_step(env, env_name, action):
    if env_name in ["kitchen-partial-v0"]:
        next_observation, reward, done, info = env.step(action)
        next_observation = next_observation[:30]
    else:
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return next_observation, reward, done, info


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def load_calibration_from_graph_path(path):
    cal_path = os.path.join(os.path.dirname(path), "tmd_calibration.json")
    if not os.path.exists(cal_path):
        return {}
    with open(cal_path, "r") as f:
        return json.load(f)


def restore_tmd(env_dataset, config):
    train_tmd_gc = TMDGCDataset(TMDDataset.create(**env_dataset), config)
    example_batch = train_tmd_gc.sample(1)
    agent = TMDAgent.create(FLAGS.seed, example_batch["observations"], example_batch["actions"], config)
    restore_dir, restore_epoch = split_checkpoint_path(FLAGS.tmd_path)
    return restore_tmd_agent(agent, restore_dir, restore_epoch)


def restore_m_agent(env_dataset, config, checkpoint_path):
    example_batch = Dataset.create(**env_dataset).sample(1)
    agent_class = agents_dict[config["agent_name"]]
    agent = agent_class.create(FLAGS.seed, example_batch["observations"], example_batch["actions"], config)
    restore_dir, restore_epoch = split_checkpoint_path(checkpoint_path)
    return restore_gas_agent(agent, restore_dir, restore_epoch)


def make_tmd_low_skill(provider, psi_obs, subgoal_obs, edge_threshold):
    psi_sub = provider.encode(subgoal_obs)
    dist = float(provider.distance_embeddings(psi_obs[None], psi_sub[None])[0, 0])
    direction = normalize(psi_sub - psi_obs)
    return np.concatenate([direction, np.asarray([np.clip(dist / edge_threshold, 0.0, 1.0)], dtype=np.float32)], axis=-1)


def evaluate_task(
    env,
    env_name,
    task_id,
    tmd_agent,
    provider,
    key_graph,
    gas_agent=None,
    tmd_low_agent=None,
    repr_cluster_threshold=None,
):
    tmd_actor_fn = supply_rng(tmd_agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed + 17 * task_id))
    gas_actor_fn = supply_rng(gas_agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed + 23 * task_id)) if gas_agent else None
    low_actor_fn = (
        supply_rng(tmd_low_agent.sample_actions, rng=jax.random.PRNGKey(FLAGS.seed + 31 * task_id))
        if tmd_low_agent
        else None
    )

    stats = defaultdict(list)
    diagnostics = defaultdict(list)
    edge_threshold = float(key_graph.edge_distance_threshold)

    for ep in trange(FLAGS.eval_episodes, desc=f"Task {task_id} episodes", leave=False):
        env, observation, goal, reward, done = setup_task_env(env, env_name, task_id, FLAGS.seed + ep)
        psi_goal = provider.encode(goal)
        psi_obs = provider.encode(observation)
        initial_goal_dist = float(provider.distance_embeddings(psi_obs[None], psi_goal[None])[0, 0])
        best_goal_dist = initial_goal_dist
        final_goal_on = False
        no_path_count = 0
        replans = 0
        subgoal_checks = 0
        subgoal_reached = 0
        final_goal_mode_steps = 0

        info = {}
        step = 0
        while not done:
            psi_obs = provider.encode(observation)
            cur_goal_obs = goal
            cur_goal_embed = psi_goal

            current_goal_dist = float(provider.distance_embeddings(psi_obs[None], psi_goal[None])[0, 0])
            best_goal_dist = min(best_goal_dist, current_goal_dist)

            if not final_goal_on:
                path = key_graph.get_shortest_path(
                    task_id=task_id,
                    source_embed=psi_obs,
                    provider=provider,
                    force_closest=FLAGS.force_closest_node,
                    edge_distance_threshold=edge_threshold,
                )
                if path is None:
                    no_path_count += 1
                    final_goal_on = True
                else:
                    replans += 1
                    path_embeds = np.asarray(path["path_embeds"])
                    path_observations = np.asarray(path["path_observations"])
                    selected_idx, dists = select_reachable_path_node(
                        provider,
                        psi_obs,
                        path_embeds,
                        edge_threshold,
                        mode=FLAGS.eval_subgoal_threshold_mode,
                        repr_cluster_threshold=repr_cluster_threshold,
                    )
                    if selected_idx is not None and len(dists):
                        subgoal_checks += 1
                        if dists[selected_idx] <= edge_threshold:
                            subgoal_reached += 1
                    if FLAGS.direct_goal_after_progress and selected_idx is not None and selected_idx > 0:
                        final_goal_on = True
                    elif len(path_embeds) <= FLAGS.eval_final_goal_threshold:
                        final_goal_on = True
                    else:
                        cur_goal_obs = path_observations[selected_idx]
                        cur_goal_embed = path_embeds[selected_idx]

            if final_goal_on:
                final_goal_mode_steps += 1
                cur_goal_obs = goal
                cur_goal_embed = psi_goal

            if FLAGS.mode == "tmd_graph_tmd_actor":
                action = tmd_actor_fn(observations=observation, goals=cur_goal_obs, temperature=FLAGS.eval_temperature)
            elif FLAGS.mode == "tmd_graph_gas_policy":
                phi_obs = np.asarray(gas_agent.get_phi(observation))
                phi_goal = np.asarray(gas_agent.get_phi(cur_goal_obs))
                skill = normalize(phi_goal - phi_obs)
                action = gas_actor_fn(observations=observation, goals=skill, temperature=FLAGS.eval_temperature)
            elif FLAGS.mode == "tmd_full_gas_low":
                skill = make_tmd_low_skill(provider, psi_obs, cur_goal_obs, edge_threshold)
                action = low_actor_fn(observations=observation, goals=skill, temperature=FLAGS.eval_temperature)
            else:
                raise ValueError(f"Unknown mode {FLAGS.mode}")

            action = np.clip(np.asarray(action), -1, 1)
            observation, reward, done, info = env_step(env, env_name, action)
            step += 1
            if FLAGS.eval_max_steps is not None and step >= FLAGS.eval_max_steps:
                done = True

        add_to(stats, flatten(info))
        final_psi_obs = provider.encode(observation)
        final_goal_dist = float(provider.distance_embeddings(final_psi_obs[None], psi_goal[None])[0, 0])
        diagnostics["eval/no_path_count"].append(no_path_count)
        diagnostics["eval/replans"].append(replans)
        diagnostics["eval/subgoal_reach_rate"].append(subgoal_reached / max(1, subgoal_checks))
        diagnostics["eval/goal_distance_initial"].append(initial_goal_dist)
        diagnostics["eval/goal_distance_best"].append(best_goal_dist)
        diagnostics["eval/goal_distance_final"].append(final_goal_dist)
        diagnostics["eval/goal_distance_improvement"].append(initial_goal_dist - final_goal_dist)
        diagnostics["eval/final_goal_mode_steps"].append(final_goal_mode_steps)

    out = {}
    for k, v in stats.items():
        out[k] = float(np.mean(v))
    for k, v in diagnostics.items():
        out[k] = float(np.mean(v))
    return out


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_eval_dir = setup_save_directory(exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_eval_dir)
    setup_wandb(FLAGS.run_eval_project, FLAGS.run_group, exp_name)

    if FLAGS.env_name in ["kitchen-partial-v0"]:
        from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.seed)

    tmd_agent = restore_tmd(train_dataset, FLAGS.tmd_agent_config)
    provider = TMDRepresentationProvider(tmd_agent, batch_size=FLAGS.tmd_agent_config["batch_size"], show_progress=False)

    key_graph = TMDKeyGraph()
    key_graph.load_keygraph(os.path.dirname(FLAGS.keygraph_tmd_path), os.path.basename(FLAGS.keygraph_tmd_path).replace(".pkl", ""))
    calibration = load_calibration_from_graph_path(FLAGS.keygraph_tmd_path)
    repr_cluster_threshold = calibration.get("repr_cluster_threshold")

    gas_agent = None
    tmd_low_agent = None
    if FLAGS.mode == "tmd_graph_gas_policy":
        gas_agent = restore_m_agent(train_dataset, FLAGS.gas_agent_config, FLAGS.gas_policy_path)
    elif FLAGS.mode == "tmd_full_gas_low":
        FLAGS.tmd_low_agent_config.edge_distance_threshold = float(key_graph.edge_distance_threshold)
        FLAGS.tmd_low_agent_config.tmd_latent_dim = int(FLAGS.tmd_agent_config["latent_dim"])
        FLAGS.tmd_low_agent_config.skill_dim = int(FLAGS.tmd_agent_config["latent_dim"]) + 1
        tmd_low_agent = restore_m_agent(train_dataset, FLAGS.tmd_low_agent_config, FLAGS.tmd_low_policy_path)

    if FLAGS.env_name in ["kitchen-partial-v0"]:
        task_infos = [{"task_name": "task1"}]
    else:
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
    task_id_list = list(range(1, num_tasks + 1))

    eval_logger = CsvLogger(os.path.join(FLAGS.save_eval_dir, "eval_tmd_gas.csv"))
    eval_metrics = {}
    overall = defaultdict(list)
    for task_id in tqdm(task_id_list, desc="Evaluating TMD-GAS tasks"):
        task_name = task_infos[task_id - 1]["task_name"]
        task_metrics = evaluate_task(
            env,
            FLAGS.env_name,
            task_id,
            tmd_agent,
            provider,
            key_graph,
            gas_agent=gas_agent,
            tmd_low_agent=tmd_low_agent,
            repr_cluster_threshold=repr_cluster_threshold,
        )
        eval_metrics.update({f"eval/{task_name}_{k}": v for k, v in task_metrics.items()})
        for k, v in task_metrics.items():
            overall[k].append(v)
    for k, v in overall.items():
        eval_metrics[f"eval/overall_{k}"] = float(np.mean(v))
    eval_metrics.update({f"graph/{k}": v for k, v in key_graph.graph_stats.items() if isinstance(v, (int, float))})

    wandb.log(eval_metrics, step=0)
    eval_logger.log(eval_metrics, step=0)
    eval_logger.close()

    with open(os.path.join(FLAGS.save_eval_dir, "eval_tmd_gas_summary.json"), "w") as f:
        json.dump(json_safe(eval_metrics), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    app.run(main)
