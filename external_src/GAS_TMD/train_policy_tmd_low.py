import json
import os
import platform
import random
import sys
import time

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
from tqdm import tqdm

from M_utils.agents import agents_dict
from M_utils.flax_utils import save_agent
from O_utils.datasets import Dataset
from O_utils.env_utils import make_env_and_datasets
from O_utils.log_utils import CsvLogger, get_exp_name, setup_save_directory, setup_wandb, wandb
from O_utils.tmd_gas_datasets import TMDGASDataset
from R_utils.calibration import calibrate_tmd_scales, save_calibration
from R_utils.checkpoint_utils import split_checkpoint_path
from R_utils.dataset_utils import limit_dataset_to_complete_prefix
from R_utils.repr_provider import TMDRepresentationProvider
from TMD_utils.tmd_agent import TMDAgent
from TMD_utils.tmd_datasets import Dataset as TMDDataset
from TMD_utils.tmd_datasets import GCDataset as TMDGCDataset
from TMD_utils.tmd_flax_utils import restore_agent as restore_tmd_agent

FLAGS = flags.FLAGS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_string("run_policy_project", "TMD_GAS", "Run project.")
flags.DEFINE_string("run_group", "tmd_full_gas_low_train", "Run group.")
flags.DEFINE_string("env_name", "antmaze-medium-stitch-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("gpu", 0, "GPU index.")
flags.DEFINE_string("save_policy_dir", "exp_policy_tmd_low/", "Save directory.")

flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval.")

flags.DEFINE_string("tmd_path", None, "Pretrained TMD params path or checkpoint directory.")
flags.DEFINE_string("tmd_calibration_path", None, "Calibration JSON from construct_graph_tmd.py.")
flags.DEFINE_float("edge_distance_threshold", None, "Override edge threshold for TMD skill distance normalization.")
flags.DEFINE_integer("max_calibration_pairs", 50000, "Calibration pairs if no calibration JSON is provided.")
flags.DEFINE_integer("max_dataset_states", None, "Optional complete-trajectory dataset prefix for smoke tests.")

config_flags.DEFINE_config_file(
    "agent_config",
    os.path.join(SCRIPT_DIR, "M_utils", "agents", "gas_tmd_low.py"),
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "tmd_agent_config",
    os.path.join(SCRIPT_DIR, "TMD_utils", "tmd_agent.py"),
    lock_config=False,
)


def load_or_calibrate_edge_threshold(dataset, provider, config, save_dir):
    if FLAGS.edge_distance_threshold is not None:
        return float(FLAGS.edge_distance_threshold)
    if FLAGS.tmd_calibration_path is not None:
        with open(FLAGS.tmd_calibration_path, "r") as f:
            scales = json.load(f)
        return float(scales["edge_distance_threshold"])
    scales = calibrate_tmd_scales(
        dataset,
        provider,
        dataset["terminals"],
        temporal_horizon_steps=int(config.get("way_steps", 8)),
        sample_size=FLAGS.max_calibration_pairs,
        seed=FLAGS.seed,
    )
    save_calibration(scales, save_dir)
    return float(scales["edge_distance_threshold"])


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent_config
    tmd_config = FLAGS.tmd_agent_config

    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_policy_dir = setup_save_directory(exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_policy_dir)
    setup_wandb(FLAGS.run_policy_project, FLAGS.run_group, exp_name)

    if FLAGS.env_name in ["kitchen-partial-v0"]:
        from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.seed)

    train_tmd_base = limit_dataset_to_complete_prefix(TMDDataset.create(**train_dataset), FLAGS.max_dataset_states)
    train_tmd_gc = TMDGCDataset(train_tmd_base, tmd_config)
    example_tmd_batch = train_tmd_gc.sample(1)
    tmd_agent = TMDAgent.create(FLAGS.seed, example_tmd_batch["observations"], example_tmd_batch["actions"], tmd_config)
    tmd_restore_dir, tmd_restore_epoch = split_checkpoint_path(FLAGS.tmd_path)
    tmd_agent = restore_tmd_agent(tmd_agent, tmd_restore_dir, tmd_restore_epoch)
    provider = TMDRepresentationProvider(tmd_agent, batch_size=tmd_config["batch_size"])

    train_base = limit_dataset_to_complete_prefix(Dataset.create(**train_dataset), FLAGS.max_dataset_states)
    train_tmd_gas_dataset = TMDGASDataset(train_base, config)
    edge_threshold = load_or_calibrate_edge_threshold(train_tmd_gas_dataset.dataset, provider, config, FLAGS.save_policy_dir)
    config.edge_distance_threshold = edge_threshold
    config.tmd_latent_dim = int(tmd_config["latent_dim"])
    config.skill_dim = int(tmd_config["latent_dim"]) + 1

    train_tmd_gas_dataset.process_features(provider, edge_threshold)
    val_tmd_gas_dataset = None
    if val_dataset is not None:
        val_tmd_gas_dataset = TMDGASDataset(
            limit_dataset_to_complete_prefix(Dataset.create(**val_dataset), FLAGS.max_dataset_states),
            config,
        )
        val_tmd_gas_dataset.process_features(provider, edge_threshold)

    example_batch = train_tmd_gas_dataset.dataset.sample(1)
    agent_class = agents_dict[config["agent_name"]]
    agent = agent_class.create(FLAGS.seed, example_batch["observations"], example_batch["actions"], config)

    train_logger = CsvLogger(os.path.join(FLAGS.save_policy_dir, "train.csv"))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm(range(1, FLAGS.train_steps + 1), desc="Training TMD-conditioned GAS low policy", smoothing=0.1, dynamic_ncols=True):
        batch = train_tmd_gas_dataset.sample(config["batch_size"])
        agent, update_info = agent.critic_actor_update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_tmd_gas_dataset is not None:
                val_batch = val_tmd_gas_dataset.sample(config["batch_size"])
                _, val_info = agent.total_critic_actor_loss(val_batch, grad_params=None)
                train_metrics.update({f"validation/{k}": v for k, v in val_info.items()})
            train_metrics["training/edge_distance_threshold"] = edge_threshold
            train_metrics["time/epoch_time"] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_policy_dir, i)
    train_logger.close()


if __name__ == "__main__":
    app.run(main)
