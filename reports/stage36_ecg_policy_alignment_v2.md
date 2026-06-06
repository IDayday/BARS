# Stage36 ECG Policy Alignment Dataset v2

| metric | value |
|---|---:|
| total_examples | 80390 |
| positive_contract_rate | 0.2836 |
| negative_contract_rate | 0.2256 |
| action_supervision_rate | 0.0000 |
| bc_trainable_count | 0 |
| final_phase_rate | 0.0373 |
| recovery_rate | 0.0006 |

## Objective Counts

{
  "boundary_compatibility_ranking": 28848,
  "contract_ranking_or_curriculum": 38191,
  "final_goal_contract_ranking": 2995,
  "ranking_contrastive_conservative_filtering": 10322,
  "recovery_contract_filtering": 34
}

## Interpretation

该数据集可用于 ranking / contrastive / conservative filtering。只有 `bc_trainable_count > 0` 时才可以规划 BC policy alignment。
