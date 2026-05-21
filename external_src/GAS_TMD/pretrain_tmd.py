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

from O_utils.env_utils import make_env_and_datasets
from O_utils.log_utils import CsvLogger, get_exp_name, setup_save_directory, setup_wandb, wandb
from TMD_utils.tmd_agent import TMDAgent
from TMD_utils.tmd_datasets import Dataset, GCDataset
from TMD_utils.tmd_flax_utils import save_agent

FLAGS = flags.FLAGS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_string("run_tmd_project", "TMD_GAS", "Run project.")
flags.DEFINE_string("run_group", "tmd_actor_train", "Run group.")
flags.DEFINE_string("env_name", "antmaze-medium-stitch-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("gpu", 0, "GPU index.")
flags.DEFINE_string("save_tmd_dir", "exp_tmd/", "Save directory.")

flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval.")

config_flags.DEFINE_config_file(
    "agent_config",
    os.path.join(SCRIPT_DIR, "TMD_utils", "tmd_agent.py"),
    lock_config=False,
)


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent_config

    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_tmd_dir = setup_save_directory(exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_tmd_dir)
    setup_wandb(FLAGS.run_tmd_project, FLAGS.run_group, exp_name)

    if FLAGS.env_name in ["kitchen-partial-v0"]:
        from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.seed)

    train_tmd_dataset = GCDataset(Dataset.create(**train_dataset), config)
    val_tmd_dataset = GCDataset(Dataset.create(**val_dataset), config) if val_dataset is not None else None

    example_batch = train_tmd_dataset.sample(1)
    agent = TMDAgent.create(FLAGS.seed, example_batch["observations"], example_batch["actions"], config)

    train_logger = CsvLogger(os.path.join(FLAGS.save_tmd_dir, "train.csv"))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm(range(1, FLAGS.train_steps + 1), desc="Training TMD", smoothing=0.1, dynamic_ncols=True):
        batch = train_tmd_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_tmd_dataset is not None:
                val_batch = val_tmd_dataset.sample(config["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f"validation/{k}": v for k, v in val_info.items()})
            train_metrics["time/epoch_time"] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_tmd_dir, i)
    train_logger.close()


if __name__ == "__main__":
    app.run(main)
