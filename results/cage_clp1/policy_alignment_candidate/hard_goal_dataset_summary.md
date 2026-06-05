# CAGE-CLP1 Graph-Induced Hard Goal Dataset

- `num_examples`: 2501
- `hard_positive`: 0
- `hard_unlabeled`: 1336
- `hard_negative`: 1165
- `available_action_supervision_rate`: 0.0
- `d_phi_mean`: 49.291824763129114
- `q_train_support_mean`: None

## Breakdown

| group | value | count |
| --- | --- | ---: |
| env_name | antmaze-giant-navigate-v0 | 1121 |
| env_name | humanoidmaze-large-navigate-v0 | 1380 |
| target_mode | farther_path_target | 711 |
| target_mode | final_goal | 768 |
| target_mode | nearest_path_target | 526 |
| target_mode | original_target | 496 |
| category | hard_negative | 1165 |
| category | hard_unlabeled | 1336 |
| d_phi_bin | 16-32 | 964 |
| d_phi_bin | 4-8 | 172 |
| d_phi_bin | 8-16 | 210 |
| d_phi_bin | >=32 | 1155 |
| q_train_support_bin | unknown | 2501 |

## Feasibility

Only `hard_positive` rows should be considered for supervised policy finetuning. `hard_negative` rows are suitable for contract/ranking losses, not naive behavior cloning.
