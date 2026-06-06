from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recover_ogbench_action_supervision import match_action, output_row


def test_match_action_recovers_exact_dataset_action():
    datasets = [{"path": "toy.npz", "tdr_emb": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), "actions": np.asarray([[0.1], [0.9]], dtype=np.float32)}]

    match = match_action([1.0, 1.0], datasets, exact_threshold=1e-6, loose_threshold=1e-2)

    assert match is not None
    action, distance, source, quality = match
    assert action == [0.8999999761581421]
    assert distance == 0.0
    assert source == "toy.npz"
    assert quality == "exact_dataset_action"


def test_output_row_does_not_mark_missing_action_as_bc():
    row = {"env_name": "toy", "phi_s": [0.0], "phi_g": [1.0], "label_contract_positive": True}

    out = output_row(row, action=None, source=None, distance=None, quality="no_action_match", reason="no_phi_match_in_dataset")

    assert out["action_available"] is False
    assert out["trainable_for_bc"] is False
    assert out["missing_reason"] == "no_phi_match_in_dataset"
