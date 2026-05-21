# Round 003 Medium Full-Budget Training Plan

Medium official artifacts remain unavailable in the official checkpoint listing. These commands are a plan, not a reduced substitute.

Evidence class before completion: E1_BASELINE_REGISTRY. Any reduced 10k/100k run must be labeled E0_SMOKE_ONLY.

## antmaze-medium-stitch-v0

- Required train steps: 1000000
- Max episode steps: 200
- Evaluation: official 49 eval + 1 video episode per task, equivalent to 50 rollouts/task across task IDs 1-5.

```bash
python pretrain_tdr.py --run_tdr_project Round003_medium_tdr --run_group round003_antmaze-medium-stitch --save_tdr_dir $OUT_ROOT/antmaze-medium-stitch-v0/seed0/tdr --train_steps 1000000 --log_interval 5000 --save_interval 100000 --env_name antmaze-medium-stitch-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group round003_antmaze-medium-stitch --save_graph_dir $OUT_ROOT/antmaze-medium-stitch-v0/seed0/graph --te_threshold 0.99 --tdr_path $OUT_ROOT/antmaze-medium-stitch-v0/seed0/tdr/round003_antmaze-medium-stitch/params_1000000.pkl --env_name antmaze-medium-stitch-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project Round003_medium_policy --run_group round003_antmaze-medium-stitch --save_policy_dir $OUT_ROOT/antmaze-medium-stitch-v0/seed0/policy --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path $OUT_ROOT/antmaze-medium-stitch-v0/seed0/tdr/round003_antmaze-medium-stitch/params_1000000.pkl --env_name antmaze-medium-stitch-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project Round003_medium_eval --run_group round003_antmaze-medium-stitch --save_eval_dir $OUT_ROOT/antmaze-medium-stitch-v0/seed0/eval --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path $OUT_ROOT/antmaze-medium-stitch-v0/seed0/graph/round003_antmaze-medium-stitch/keygraph.pkl --policy_path $OUT_ROOT/antmaze-medium-stitch-v0/seed0/policy/round003_antmaze-medium-stitch/params_1000000.pkl --env_name antmaze-medium-stitch-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
```

## antmaze-medium-navigate-v0

- Required train steps: 1000000
- Max episode steps: 1000
- Evaluation: official 49 eval + 1 video episode per task, equivalent to 50 rollouts/task across task IDs 1-5.

```bash
python pretrain_tdr.py --run_tdr_project Round003_medium_tdr --run_group round003_antmaze-medium-navigate --save_tdr_dir $OUT_ROOT/antmaze-medium-navigate-v0/seed0/tdr --train_steps 1000000 --log_interval 5000 --save_interval 100000 --env_name antmaze-medium-navigate-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group round003_antmaze-medium-navigate --save_graph_dir $OUT_ROOT/antmaze-medium-navigate-v0/seed0/graph --te_threshold 0.99 --tdr_path $OUT_ROOT/antmaze-medium-navigate-v0/seed0/tdr/round003_antmaze-medium-navigate/params_1000000.pkl --env_name antmaze-medium-navigate-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project Round003_medium_policy --run_group round003_antmaze-medium-navigate --save_policy_dir $OUT_ROOT/antmaze-medium-navigate-v0/seed0/policy --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path $OUT_ROOT/antmaze-medium-navigate-v0/seed0/tdr/round003_antmaze-medium-navigate/params_1000000.pkl --env_name antmaze-medium-navigate-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project Round003_medium_eval --run_group round003_antmaze-medium-navigate --save_eval_dir $OUT_ROOT/antmaze-medium-navigate-v0/seed0/eval --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path $OUT_ROOT/antmaze-medium-navigate-v0/seed0/graph/round003_antmaze-medium-navigate/keygraph.pkl --policy_path $OUT_ROOT/antmaze-medium-navigate-v0/seed0/policy/round003_antmaze-medium-navigate/params_1000000.pkl --env_name antmaze-medium-navigate-v0 --seed 0 --gpu $GPU --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
```
