# CAGE Contract Dataset Splits

- status: ok
- input_path: `/mnt/project/BARS/results/cage_clp1/datasets/closed_loop_contracts.jsonl`
- total_examples: 9216
- group_keys: env_name, seed, task_id, variant_source

| split | examples | hit | contract_positive | negative_progress | final_goal | recovery | policy_weak |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6912 | 0.083 | 0.088 | 0.051 | 0.167 | 0.171 | 0.302 |
| val | 1152 | 0.029 | 0.036 | 0.078 | 0.167 | 0.167 | 0.469 |
| test | 1152 | 0.029 | 0.036 | 0.078 | 0.167 | 0.167 | 0.469 |
