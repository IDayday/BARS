# Stage38 Action-Anchored ECG Contract Model

- status: `CONTRACT_MODEL_READY`
- examples: 80
- model: `/tmp/pytest-of-root/pytest-129/test_action_anchored_contract_0/model/model.pt`
- metrics: `/tmp/pytest-of-root/pytest-129/test_action_anchored_contract_0/model/metrics.json`

## Gate

- positive_contract_beats_dphi: `True`
- negative_contract_beats_dphi: `True`

## Test Metrics

- contract_positive: status=trained auroc_test=1.0 dphi_auroc_test=0.5625 brier_test=0.0455528125166893
- negative_progress: status=trained auroc_test=1.0 dphi_auroc_test=0.4375 brier_test=0.0455528162419796
- final_goal: status=single_class_train auroc_test=None dphi_auroc_test=None brier_test=None