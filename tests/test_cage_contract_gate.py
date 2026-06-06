from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.closed_loop_contracts import evaluate_contract_gate
from cage.contract_model import ContractPrediction, ContractScorer


def test_contract_gate_passes_conservative_good_prediction():
    prediction = ContractPrediction(
        predicted_hit=0.8,
        predicted_contract_positive=0.75,
        predicted_negative_progress=0.1,
        uncertainty=0.1,
        lower_confidence_bound=0.7,
        model_loaded=True,
    )
    gate = evaluate_contract_gate(
        prediction,
        lcb_threshold=0.35,
        negative_progress_threshold=0.45,
        target_mode="cage_selected",
    )
    assert gate.passed
    assert gate.reason == "contract_pass"
    assert gate.trace_fields()["contract_gate_pass"] is True


def test_contract_gate_rejects_negative_progress_risk_before_lcb():
    prediction = ContractPrediction(
        predicted_hit=0.8,
        predicted_contract_positive=0.75,
        predicted_negative_progress=0.9,
        uncertainty=0.1,
        lower_confidence_bound=0.7,
        model_loaded=True,
    )
    gate = evaluate_contract_gate(
        prediction,
        lcb_threshold=0.35,
        negative_progress_threshold=0.45,
        target_mode="cage_selected",
    )
    assert not gate.passed
    assert gate.reason == "negative_progress_risk"


def test_contract_scorer_fallback_marks_model_unloaded_and_penalizes_far_targets():
    scorer = ContractScorer.from_path(None, uncertainty_penalty=0.25)
    near = scorer.predict({"phi_s": np.asarray([0.0]), "phi_g": np.asarray([1.0]), "d_phi": 1.0})
    far = scorer.predict({"phi_s": np.asarray([0.0]), "phi_g": np.asarray([20.0]), "d_phi": 20.0})
    assert near.model_loaded is False
    assert far.model_loaded is False
    assert near.lower_confidence_bound > far.lower_confidence_bound
