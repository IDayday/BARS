# Stage37 ECG Policy Alignment Dataset v3

| metric | value |
|---|---:|
| total_examples | 44635 |
| action_supervision_rate | 0.0000 |
| positive_with_action_count | 0 |
| final_goal_with_action_count | 0 |
| recovery_with_action_count | 0 |
| bc_trainable_count | 0 |
| bc_split_generated | False |

## Trust Counts

{
  "candidate_knn": 32641,
  "final_candidate": 2995,
  "observed": 8999
}

## Training Stage Counts

{
  "contract_ranking_contrastive": 5042,
  "final_goal_contract_ranking": 2995,
  "negative_contract_conservative_filtering": 3449,
  "positive_contract_ranking": 508,
  "trusted_bridge_conservative_filtering": 32641
}

KNN bridge candidate 只能进入 conservative filtering 或 ranking，不能直接 BC。BC split 只有在 positive_with_action_count > 0 时生成。
