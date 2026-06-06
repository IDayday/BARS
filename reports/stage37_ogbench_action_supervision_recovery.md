# Stage37 OGBench Action Supervision Recovery

| metric | value |
|---|---:|
| status | BLOCKED_NO_ACTION_MATCH |
| total_examples | 7785 |
| exact_action_count | 0 |
| loose_action_count | 0 |
| action_supervision_rate | 0.0000 |
| positive_with_action_count | 0 |
| final_goal_with_action_count | 0 |
| recovery_with_action_count | 0 |

## Per-Env

{
  "antmaze-giant-navigate-v0": {
    "action_available": 0,
    "action_supervision_rate": 0.0,
    "dataset_files": 3,
    "examples": 6324
  },
  "antmaze-giant-stitch-v0": {
    "action_available": 0,
    "action_supervision_rate": null,
    "dataset_files": 3,
    "examples": 0
  },
  "humanoidmaze-large-navigate-v0": {
    "action_available": 0,
    "action_supervision_rate": 0.0,
    "dataset_files": 0,
    "examples": 1461
  }
}

## Missing Reason Counts

{
  "no_dataset_npz_with_tdr_emb_actions": 1461,
  "no_phi_match_in_dataset": 6324
}

如果 action_supervision_rate 为 0，则 BC policy alignment 继续 BLOCKED；不能从 phi-only 样本虚构 action。
