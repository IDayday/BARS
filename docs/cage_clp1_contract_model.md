# CAGE-CLP1 Contract Model

CLP1 includes a minimal offline linear logistic contract predictor:
- `external_src/GAS/cage/contract_model.py`
- `train_cage_contract.py`
- `evaluate_cage_contract.py`

Inputs:
- `phi_start`
- `phi_target`
- `phi_target - phi_start`
- `abs(phi_target - phi_start)`
- `d_phi_start`

Targets:
- `hit`
- `contract_positive`
- `negative_progress`

This model is intentionally simple. It is a gate for deciding whether closed-loop contract labels are learnable above a distance-only heuristic. It is not used by `evaluate_gas.py` in CLP1.
