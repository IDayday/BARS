# CAGE-CLP1 Contract Dataset

`scripts/build_closed_loop_contract_dataset.py` converts branchable probe JSONL files into offline contract examples.

Fields include:
- `phi_start`, `phi_target`, `phi_delta`, `abs_phi_delta`
- `d_phi_start`
- `env_name`, `variant_source`, `target_mode`, `path_position`
- `final_phase`, `recovery_candidate`, optional `q_train_support`
- probe outcomes: `hit`, `normalized_progress`, `negative_progress`, action norms
- labels:
  - `label_contract_positive`: hit or sufficient normalized progress
  - `label_contract_negative`: negative or near-zero progress
  - `label_recovery_bad`: recovery candidate with a negative contract
  - `label_policy_weak`: all valid candidates for the same segment/horizon are contract-negative

This dataset is for contract-model and policy-alignment analysis only. It is not integrated into GAS/CAGE evaluation in CLP1.
