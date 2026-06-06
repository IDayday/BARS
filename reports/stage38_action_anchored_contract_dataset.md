# Stage38 Action-Anchored Contract Dataset

- status: `ACTION_ANCHORED_DATASET_READY`
- jsonl: `/tmp/pytest-of-root/pytest-129/test_action_anchored_dataset_f0/out/action_contracts.jsonl`
- npz: `/tmp/pytest-of-root/pytest-129/test_action_anchored_dataset_f0/out/action_contracts.npz`
- total_examples: 64
- action_supervision_rate: 1.0
- positive_with_action_count: 28
- final_goal_with_action_count: 3

本数据集从 raw/offline trajectory 的 observation/action/next_observation 重建样本；不使用 Stage37 的 phi-only contract examples 反向匹配 action 作为主路线。

## Per-Env

### antmaze-giant-navigate-v0
- status: `ok`
- dataset_paths: `['/mnt/project/BARS/artifacts/stage27_gas/datasets/antmaze-giant-navigate-v0/gas_seed44/dataset.npz']`
- phi_sources: `['tdr_emb']`
- source_rows: 64
- sample_count: 64
- trajectory_count: 22
- action_dim: 8
- phi_dim: 32
- positive_count: 28
- negative_count: 36
- final_goal_count: 3
