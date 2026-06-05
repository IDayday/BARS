import os
import sys
import platform

# Disable preallocation of GPU memory.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 

# Set the GPU index.
gpu_index = sys.argv[sys.argv.index('--gpu') + 1] if '--gpu' in sys.argv else "0" 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
print(f"\033[38;5;208m{'=' * 14}\n Using GPU: {gpu_index}\n{'=' * 14}\033[0m")

# Set up EGL for rendering.
if 'mac' in platform.platform():
    pass
else:
    os.environ.setdefault('MUJOCO_GL', 'egl')
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']
       
import random
import numpy as np

from tqdm import tqdm
from absl import app, flags
from collections import defaultdict
from dataclasses import replace
from ml_collections import config_flags

from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

from O_utils.datasets import Dataset, GCDataset
from O_utils.evaluation import evaluate_with_graph
from O_utils.env_utils import make_env_and_datasets
from O_utils.log_utils import get_exp_name, setup_save_directory, setup_wandb, get_wandb_video, CsvLogger, wandb

from K_utils.keygraph_utils import KeyGraph

from M_utils.agents import agents_dict
from M_utils.flax_utils import restore_agent
from cage.config import CAGEConfig
from cage.tracing import CAGETraceWriter

# Flags for task Planning and execution. (GAS Stage 4).
FLAGS = flags.FLAGS

flags.DEFINE_string('run_eval_project', 'Debug', 'Run Evaluation Project.') 
flags.DEFINE_string('run_group', 'Debug', 'Run group.') 
flags.DEFINE_string('env_name', 'antmaze-giant-stitch-v0', 'Environment name.') 
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', 0, 'GPU index')
flags.DEFINE_string('save_eval_dir', 'exp_policy/', 'Save directory.')

flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.') 
flags.DEFINE_integer('eval_episodes', 49, 'Number of episodes for each task.') 
flags.DEFINE_integer('eval_video_episodes', 1, 'Number of video episodes for each task.') 
flags.DEFINE_integer('eval_final_goal_threshold', 2, 'Threshold to switch to final goal') 

flags.DEFINE_string('keygraph_path', None, 'Path to the constructed TD-aware graph') 
flags.DEFINE_string('policy_path', None, 'Pretrained low-level policy path.') 

flags.DEFINE_bool('use_cage', False, 'Enable Control-Aligned Graph Execution wrapper.')
flags.DEFINE_string('cage_trace_path', '', 'JSONL trace path for CAGE. Defaults to save_eval_dir/cage_trace.jsonl.')
flags.DEFINE_integer('cage_min_commit_steps', 8, 'Minimum steps to commit to a CAGE subgoal.')
flags.DEFINE_integer('cage_stall_window', 8, 'Rolling window used for CAGE stall detection.')
flags.DEFINE_float('cage_progress_eps', 0.01, 'Minimum progress over the stall window.')
flags.DEFINE_float('cage_drift_threshold', 16.0, 'Distance-to-path threshold for CAGE path drift.')
flags.DEFINE_float('cage_max_subgoal_dist', 24.0, 'Maximum CAGE subgoal distance in TDR space.')
flags.DEFINE_float('cage_min_subgoal_dist', 2.0, 'Minimum CAGE subgoal distance unless near the final goal.')
flags.DEFINE_integer('cage_recovery_commit_steps', 12, 'Commitment length for CAGE local recovery targets.')
flags.DEFINE_integer('cage_max_recovery_attempts', 2, 'Maximum local recovery attempts before requesting global replanning.')
flags.DEFINE_float('cage_recovery_suffix_weight', 0.25, 'Penalty for CAGE recovery targets that move backward on the path.')
flags.DEFINE_float('cage_final_phase_dist', 8.0, 'Distance threshold for CAGE final-goal phase.')
flags.DEFINE_integer('cage_final_min_commit_steps', 12, 'Minimum commitment in CAGE final-goal phase.')
flags.DEFINE_bool('cage_debug', False, 'Emit CAGE step-level JSONL trace records.')

config_flags.DEFINE_config_file('agent_config', 'M_utils/agents/gas.py', lock_config=False) 


def main(_):
    # Set random seeds and load agent configuration.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent_config
    
    # Set up logger
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_eval_dir = setup_save_directory(exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_eval_dir)
    setup_wandb(FLAGS.run_eval_project, FLAGS.run_group, exp_name)
    
    # Set up environment and dataset.
    if FLAGS.env_name in ['kitchen-partial-v0',]:
        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.seed)
    train_gc_dataset = GCDataset(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_gc_dataset = GCDataset(Dataset.create(**val_dataset), config)

    # Initialize agent.
    example_batch = train_gc_dataset.sample(1)
    agent_class = agents_dict[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config,)

    # Restore low-level policy.
    policy_restore_path = os.path.dirname(FLAGS.policy_path)
    policy_restore_epoch = os.path.basename(FLAGS.policy_path).split('_')[-1].split('.')[0]
    agent = restore_agent(agent, policy_restore_path, policy_restore_epoch)
                
    # Restore graph.
    key_graph = KeyGraph() 
    keygraph_load_path = os.path.dirname(FLAGS.keygraph_path)
    keygraph_load_filename = os.path.basename(FLAGS.keygraph_path).split('_')[-1].split('.')[0]
    key_graph.load_keygraph(keygraph_load_path, keygraph_load_filename)

    # Set up evaluation tasks.
    if FLAGS.env_name in ['kitchen-partial-v0',]: 
        task_infos = [{'task_name': 'task1',}]
    else:   
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    task_id_list = list(range(1, len(task_infos) + 1))
    
    # Evaluate GAS.   
    eval_logger = CsvLogger(os.path.join(policy_restore_path, 'eval.csv'))
    metric_names = ["episode.success", "episode.return",  "episode.normalized_return", "episode.length", "episode.duration"]
    renders = []
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    cage_config = None
    cage_trace_writer = None
    if FLAGS.use_cage:
        cage_trace_path = FLAGS.cage_trace_path or os.path.join(FLAGS.save_eval_dir, 'cage_trace.jsonl')
        cage_config = CAGEConfig.from_flags(FLAGS)
        cage_config = replace(cage_config, trace_path=cage_trace_path)
        cage_trace_writer = CAGETraceWriter(cage_trace_path, debug=FLAGS.cage_debug)
    for task_id in tqdm(task_id_list, desc="Evaluating Tasks"):
        task_name = task_infos[task_id - 1]['task_name']
        eval_info, cur_renders = evaluate_with_graph(agent, key_graph, env, FLAGS.env_name, task_id, FLAGS.eval_episodes, FLAGS.eval_video_episodes, 
                                                     FLAGS.seed, FLAGS.eval_on_cpu, config['way_steps'], FLAGS.eval_final_goal_threshold, config,
                                                     cage_config=cage_config, cage_trace_writer=cage_trace_writer,)
        renders.extend(cur_renders)
        eval_metrics.update({f'eval/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names})
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v) 
    for k, v in overall_metrics.items():
        eval_metrics[f'eval/overall_{k}'] = np.mean(v)
    if FLAGS.eval_video_episodes > 0:
        video = get_wandb_video(renders=renders, n_cols=len(task_id_list))
        eval_metrics['video'] = video
    wandb.log(eval_metrics, step=0) 
    eval_logger.log(eval_metrics, step=0)
    eval_logger.close()
    if cage_trace_writer is not None:
        cage_trace_writer.close()
        
if __name__ == '__main__':
    app.run(main)
