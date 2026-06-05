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
from cage.contract_tracing import ContractTraceWriter

# Flags for task Planning and execution. (GAS Stage 4).
FLAGS = flags.FLAGS

flags.DEFINE_string('run_eval_project', 'Debug', 'Run Evaluation Project.') 
flags.DEFINE_string('run_group', 'Debug', 'Run group.') 
flags.DEFINE_string('env_name', 'antmaze-giant-stitch-v0', 'Environment name.') 
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', 0, 'GPU index')
flags.DEFINE_string('save_eval_dir', 'exp_policy/', 'Save directory.')
flags.DEFINE_string('eval_result_path', '', 'Optional CSV path for evaluation metrics. Defaults to policy checkpoint directory eval.csv.')

flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.') 
flags.DEFINE_integer('eval_episodes', 49, 'Number of episodes for each task.') 
flags.DEFINE_integer('eval_video_episodes', 1, 'Number of video episodes for each task.') 
flags.DEFINE_integer('eval_final_goal_threshold', 2, 'Threshold to switch to final goal') 
flags.DEFINE_integer('eval_max_tasks', 0, 'Optional maximum number of task IDs to evaluate. Zero keeps all tasks.')

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
flags.DEFINE_bool('cage_disable_commitment', False, 'Disable CAGE subgoal commitment for ablations.')
flags.DEFINE_bool('cage_disable_drift_monitor', False, 'Disable CAGE drift-triggered control for ablations.')
flags.DEFINE_bool('cage_disable_recovery', False, 'Disable CAGE local recovery for ablations.')
flags.DEFINE_bool('cage_disable_adaptive_horizon', False, 'Disable CAGE adaptive subgoal horizon for ablations.')
flags.DEFINE_bool('cage_disable_final_phase_controller', False, 'Disable CAGE final-goal phase controller for ablations.')
flags.DEFINE_bool('cage_use_reachability', False, 'Reserved flag for future learned CAGE reachability support.')
flags.DEFINE_string('cage_reachability_path', '', 'Reserved path for a future learned CAGE reachability model.')
flags.DEFINE_bool('cage_risk_aware_path', False, 'Reserved flag for future CAGE risk-aware path execution.')
flags.DEFINE_bool('cage_trace_only', False, 'Emit CAGE traces while passing through the original GAS subgoal.')
flags.DEFINE_bool('cage_enable_churn_guard', False, 'Enable CAGE replan-churn guardrails.')
flags.DEFINE_integer('cage_replan_cooldown_steps', 10, 'Minimum steps between CAGE-triggered global replans in safe mode.')
flags.DEFINE_integer('cage_max_global_replans_per_episode', 50, 'Maximum allowed CAGE-triggered global replans per episode in safe mode.')
flags.DEFINE_integer('cage_max_replans_per_100_steps', 10, 'Maximum CAGE replan requests in a 100-step window in safe mode.')
flags.DEFINE_integer('cage_max_consecutive_replan_requests', 5, 'Maximum consecutive CAGE replan requests before fallback.')
flags.DEFINE_bool('cage_fallback_to_gas_on_churn', True, 'Fallback to original GAS target selection after churn guard triggers.')
flags.DEFINE_integer('cage_fallback_to_gas_steps', 50, 'Number of steps to use GAS target selection after churn fallback.')
flags.DEFINE_integer('cage_recovery_lockout_steps_after_failure', 25, 'Steps to suppress recovery after a failed recovery attempt.')
flags.DEFINE_integer('cage_min_steps_between_recovery_attempts', 20, 'Minimum steps between CAGE recovery attempts in safe mode.')
flags.DEFINE_float('cage_min_progress_for_recovery_success', 1e-4, 'Minimum recovery progress before considering recovery nonfailed.')
flags.DEFINE_bool('cage_disable_recovery_after_churn', False, 'Disable further CAGE recovery attempts after churn guard triggers.')
flags.DEFINE_bool('cage_log_churn_events', False, 'Include churn guard event fields in CAGE traces.')

flags.DEFINE_string('contract_trace_path', '', 'Optional JSONL path for closed-loop segment contract traces.')
flags.DEFINE_bool('contract_trace_debug', False, 'Enable verbose contract trace behavior.')
flags.DEFINE_bool('store_contract_state_refs', False, 'Store exact StateRefs in contract traces when available.')
flags.DEFINE_string('contract_state_ref_mode', 'metadata_only', 'StateRef mode: exact_only|best_effort|metadata_only.')
flags.DEFINE_string('contract_capture_variants', 'gas,cage_trace_only,cage_fixed_commit,cage_safe_full', 'Comma-separated variants to contract-capture.')
flags.DEFINE_bool('contract_capture_segment_start', True, 'Capture segment starts in contract trace.')
flags.DEFINE_bool('contract_capture_segment_end', True, 'Capture segment ends in contract trace.')
flags.DEFINE_bool('contract_capture_qpos_qvel', True, 'Capture qpos/qvel in StateRefs when available.')
flags.DEFINE_bool('contract_capture_phi', True, 'Capture phi in segment StateRefs.')
flags.DEFINE_bool('contract_capture_action_stats', True, 'Capture action/skill norm stats in segment records.')

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
    if FLAGS.eval_max_tasks > 0:
        task_id_list = task_id_list[:FLAGS.eval_max_tasks]
    
    # Evaluate GAS.   
    eval_logger = CsvLogger(FLAGS.eval_result_path or os.path.join(policy_restore_path, 'eval.csv'))
    metric_names = ["episode.success", "episode.return",  "episode.normalized_return", "episode.length", "episode.duration"]
    renders = []
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    eval_video_episodes = 0 if FLAGS.env_name in ['kitchen-partial-v0'] else FLAGS.eval_video_episodes
    cage_config = None
    cage_trace_writer = None
    contract_trace_writer = None
    contract_config = None
    if FLAGS.use_cage:
        cage_trace_path = FLAGS.cage_trace_path or os.path.join(FLAGS.save_eval_dir, 'cage_trace.jsonl')
        cage_config = CAGEConfig.from_flags(FLAGS)
        cage_config = replace(cage_config, trace_path=cage_trace_path)
        cage_trace_writer = CAGETraceWriter(cage_trace_path, debug=FLAGS.cage_debug)
    if FLAGS.contract_trace_path:
        contract_trace_writer = ContractTraceWriter(FLAGS.contract_trace_path, debug=FLAGS.contract_trace_debug)
        contract_config = {
            "enabled": True,
            "trace_path": FLAGS.contract_trace_path,
            "debug": FLAGS.contract_trace_debug,
            "store_state_refs": FLAGS.store_contract_state_refs,
            "state_ref_mode": FLAGS.contract_state_ref_mode,
            "capture_variants": [x.strip() for x in FLAGS.contract_capture_variants.split(",") if x.strip()],
            "capture_segment_start": FLAGS.contract_capture_segment_start,
            "capture_segment_end": FLAGS.contract_capture_segment_end,
            "capture_qpos_qvel": FLAGS.contract_capture_qpos_qvel,
            "capture_phi": FLAGS.contract_capture_phi,
            "capture_action_stats": FLAGS.contract_capture_action_stats,
            "hit_threshold": config["way_steps"],
            "variant": infer_contract_variant(FLAGS),
        }
    for task_id in tqdm(task_id_list, desc="Evaluating Tasks"):
        task_name = task_infos[task_id - 1]['task_name']
        eval_info, cur_renders = evaluate_with_graph(agent, key_graph, env, FLAGS.env_name, task_id, FLAGS.eval_episodes, eval_video_episodes, 
                                                     FLAGS.seed, FLAGS.eval_on_cpu, config['way_steps'], FLAGS.eval_final_goal_threshold, config,
                                                     cage_config=cage_config, cage_trace_writer=cage_trace_writer,
                                                     contract_trace_writer=contract_trace_writer, contract_config=contract_config,)
        renders.extend(cur_renders)
        eval_metrics.update({f'eval/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names})
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v) 
    for k, v in overall_metrics.items():
        eval_metrics[f'eval/overall_{k}'] = np.mean(v)
    if eval_video_episodes > 0:
        video = get_wandb_video(renders=renders, n_cols=len(task_id_list))
        eval_metrics['video'] = video
    wandb.log(eval_metrics, step=0) 
    eval_logger.log(eval_metrics, step=0)
    eval_logger.close()
    if cage_trace_writer is not None:
        cage_trace_writer.close()
    if contract_trace_writer is not None:
        contract_trace_writer.close()
        
def infer_contract_variant(flags_obj):
    if not flags_obj.use_cage:
        return 'gas'
    if flags_obj.cage_trace_only:
        return 'cage_trace_only'
    if flags_obj.cage_enable_churn_guard:
        return 'cage_safe_full'
    if (flags_obj.cage_disable_drift_monitor and flags_obj.cage_disable_recovery
            and flags_obj.cage_disable_adaptive_horizon and flags_obj.cage_disable_final_phase_controller):
        return 'cage_fixed_commit'
    return 'cage_full'

if __name__ == '__main__':
    app.run(main)
