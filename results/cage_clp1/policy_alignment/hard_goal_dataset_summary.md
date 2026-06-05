# CAGE-CLP1 Graph-Induced Hard Goal Dataset

- `num_examples`: 956
- `hard_positive`: 0
- `hard_unlabeled`: 425
- `hard_negative`: 531
- `available_action_supervision_rate`: 0.0
- `d_phi_mean`: 21.666575850552594
- `q_train_support_mean`: None

## Breakdown

| group | value | count |
| --- | --- | ---: |
| env_name | antmaze-giant-navigate-v0 | 326 |
| env_name | humanoidmaze-large-navigate-v0 | 630 |
| target_mode | original_target | 953 |
| target_mode | recovery_candidate | 3 |
| category | hard_negative | 531 |
| category | hard_unlabeled | 425 |
| d_phi_bin | 16-32 | 719 |
| d_phi_bin | 4-8 | 181 |
| d_phi_bin | 8-16 | 44 |
| d_phi_bin | <4 | 12 |
| q_train_support_bin | unknown | 956 |

## Feasibility

Only `hard_positive` rows should be considered for supervised policy finetuning. `hard_negative` rows are suitable for contract/ranking losses, not naive behavior cloning.
