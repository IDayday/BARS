# CAGE Contract Model Evaluation

- status: ok
- dataset_path: `results/cage_v02_contract/splits/test.jsonl`
- model_path: `results/cage_v02_contract/models/contract_model.json`

| label | AUROC | AUPRC | Brier | d_phi baseline AUROC |
|---|---:|---:|---:|---:|
| hit | 1.000 | 1.000 | 0.020 | 0.997 |
| contract_positive | 0.954 | 0.857 | 0.079 | 0.873 |
| negative_progress | 0.719 | 0.688 | 0.222 | 0.339 |
