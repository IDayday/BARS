# Stage36 Action-Supervised Contract Mining

| metric | value |
|---|---:|
| total_candidates | 2247 |
| action_available_count | 0 |
| action_supervision_rate | 0.0000 |
| positive_with_action_count | 0 |
| final_goal_with_action_count | 0 |
| recovery_with_action_count | 0 |

## Missing Reason Counts

{
  "segment_capture_has_no_action_fields": 2247
}

## Interpretation

如果 action_supervision_rate 仍接近 0，则不能做 BC，只能做 ranking/contrastive/conservative filtering，或继续恢复带 action 的 segment trace / OGBench trajectory supervision。
