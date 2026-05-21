import json
import os
import platform
import random
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_index = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
print(f"\033[38;5;208m{'=' * 14}\n Using GPU: {gpu_index}\n{'=' * 14}\033[0m")

if "mac" not in platform.platform():
    os.environ.setdefault("MUJOCO_GL", "egl")
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from K_utils.keygraph_tmd_utils import TMDKeyGraph
from K_utils.keynodes_tmd_utils import TMDKeyNodes
from O_utils.env_utils import make_env_and_datasets
from O_utils.log_utils import get_exp_name, setup_save_directory
from R_utils.calibration import calibrate_tmd_scales, quantile_key, save_calibration
from R_utils.checkpoint_utils import split_checkpoint_path
from R_utils.dataset_utils import limit_dataset_to_complete_prefix
from R_utils.json_utils import json_safe
from R_utils.repr_provider import TMDRepresentationProvider
from TMD_utils.tmd_agent import TMDAgent
from TMD_utils.tmd_datasets import Dataset, GCDataset
from TMD_utils.tmd_flax_utils import restore_agent

FLAGS = flags.FLAGS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_string("run_group", "tmd_graph_construct", "Run group.")
flags.DEFINE_string("env_name", "antmaze-medium-stitch-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("gpu", 0, "GPU index.")
flags.DEFINE_string("save_graph_dir", "exp_graph_tmd/", "Save directory.")

flags.DEFINE_string("tmd_path", None, "Pretrained TMD params path or checkpoint directory.")
flags.DEFINE_string("gas_tdr_path", None, "Optional GAS TDR path, recorded for provenance only.")
flags.DEFINE_float("te_threshold", 0.99, "TE threshold.")
flags.DEFINE_integer("temporal_horizon_steps", None, "Temporal H used for calibration and TE filtering.")
flags.DEFINE_float("edge_quantile", 0.75, "TMD calibration quantile for graph edges.")
flags.DEFINE_float("target_quantile", 0.90, "TMD calibration quantile for target edges.")
flags.DEFINE_integer("max_calibration_pairs", 50000, "Maximum sampled calibration pairs.")
flags.DEFINE_integer("max_dataset_states", None, "Optional complete-trajectory dataset prefix for smoke tests.")
flags.DEFINE_integer("topk", None, "Optional top-k psi-L2 candidates per node before TMD scoring.")

config_flags.DEFINE_config_file(
    "agent_config",
    os.path.join(SCRIPT_DIR, "TMD_utils", "tmd_agent.py"),
    lock_config=False,
)


def setup_task_env_local(env, env_name, task_id, seed):
    if env_name in ["kitchen-partial-v0"]:
        from D_utils.kitchen_utils import kitchen_set_obs_and_goal

        env, observation, goal = kitchen_set_obs_and_goal(env, env_name, task_id, seed=seed)
    else:
        observation, info = env.reset(seed=seed, options=dict(task_id=task_id, render_goal=False))
        goal = info.get("goal")
    return env, observation, goal


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent_config
    horizon = FLAGS.temporal_horizon_steps or int(config.get("way_steps", 8))

    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_graph_dir = setup_save_directory(exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_graph_dir)

    if FLAGS.env_name in ["kitchen-partial-v0"]:
        from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.seed)

    train_tmd_dataset = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = train_tmd_dataset.sample(1)
    tmd_agent = TMDAgent.create(FLAGS.seed, example_batch["observations"], example_batch["actions"], config)
    restore_dir, restore_epoch = split_checkpoint_path(FLAGS.tmd_path)
    tmd_agent = restore_agent(tmd_agent, restore_dir, restore_epoch)

    provider = TMDRepresentationProvider(tmd_agent, batch_size=config["batch_size"])
    dataset = limit_dataset_to_complete_prefix(train_tmd_dataset.dataset, FLAGS.max_dataset_states)
    all_psi = provider.encode(dataset["observations"])

    quantiles = sorted(set([0.5, 0.75, 0.9, float(FLAGS.edge_quantile), float(FLAGS.target_quantile)]))
    scales = calibrate_tmd_scales(
        dataset,
        provider,
        dataset["terminals"],
        temporal_horizon_steps=horizon,
        sample_size=FLAGS.max_calibration_pairs,
        quantiles=quantiles,
        seed=FLAGS.seed,
    )
    calibration_path = save_calibration(scales, FLAGS.save_graph_dir)

    edge_key = quantile_key("tmd_dist", FLAGS.edge_quantile)
    target_key = quantile_key("tmd_dist", FLAGS.target_quantile)
    edge_threshold = scales[edge_key]
    target_threshold = scales[target_key]

    key_nodes = TMDKeyNodes().construct_nodes(
        all_psi,
        dataset["observations"],
        dataset["terminals"],
        temporal_horizon_steps=horizon,
        repr_cluster_threshold=scales["repr_cluster_threshold"],
        te_threshold=FLAGS.te_threshold,
    )
    key_nodes.save_keynodes(FLAGS.save_graph_dir, "keynodes_tmd")

    key_graph = TMDKeyGraph().construct_graph(
        key_nodes,
        provider,
        edge_distance_threshold=edge_threshold,
        batch_size=config["batch_size"],
        topk=FLAGS.topk,
    )

    if FLAGS.env_name in ["kitchen-partial-v0"]:
        task_infos = [{"task_name": "task1"}]
    else:
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    task_id_list = list(range(1, len(task_infos) + 1))

    task_goal_dict = {}
    task_node_dict = {}
    task_obs_dict = {}
    for task_id in task_id_list:
        env, observation, goal = setup_task_env_local(env, FLAGS.env_name, task_id, FLAGS.seed)
        task_goal_dict[task_id] = goal
        task_obs_dict[task_id] = goal
        task_node_dict[task_id] = provider.encode(goal)

    key_graph.precompute_shortest_paths_to_all_tasks(
        task_goal_dict,
        task_node_dict,
        task_obs_dict,
        provider,
        target_distance_threshold=target_threshold,
    )
    key_graph.save_keygraph(FLAGS.save_graph_dir, "keygraph_tmd")
    key_graph.save_graph_stats(FLAGS.save_graph_dir, "graph_stats.csv")
    with open(os.path.join(FLAGS.save_graph_dir, "graph_stats.json"), "w") as f:
        json.dump(json_safe(key_graph.graph_stats), f, indent=2, sort_keys=True)

    provenance = {
        "tmd_path": FLAGS.tmd_path,
        "gas_tdr_path": FLAGS.gas_tdr_path,
        "calibration_path": calibration_path,
        "edge_quantile": FLAGS.edge_quantile,
        "target_quantile": FLAGS.target_quantile,
        "edge_distance_threshold": edge_threshold,
        "target_distance_threshold": target_threshold,
    }
    with open(os.path.join(FLAGS.save_graph_dir, "tmd_graph_provenance.json"), "w") as f:
        json.dump(json_safe(provenance), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    app.run(main)
