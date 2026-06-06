# Stage35 Graph-Contract Policy Dataset

| metric | value |
|---|---:|
| total_examples | 2956 |
| positive_contract_rate | 0.2710 |
| negative_contract_rate | 0.4259 |
| action_supervision_rate | 0.0000 |
| final_phase_rate | 0.0000 |
| recovery_rate | 0.0041 |

## Objective Counts

{
  "boundary_compatibility_ranking": 1414,
  "contract_ranking_or_curriculum": 648,
  "ranking_contrastive_conservative_filtering": 894
}

## Feasibility

Policy finetuning is feasible only if positive contract examples also have action supervision. If action supervision rate is zero, this dataset should be used for ranking, contrastive objectives, conservative filtering, or future data collection, not naive BC.
