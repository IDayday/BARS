import sys
from pathlib import Path

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.closed_loop_contracts import ContractThresholds, contract_dataset_row, label_closed_loop_contract


def test_contract_labels_mark_hit_as_good_contract():
    record = {
        "hit": True,
        "normalized_progress": 0.05,
        "action_norm_max": 0.5,
        "d_phi_start": 10.0,
        "final_d_phi": 9.5,
    }
    labels = label_closed_loop_contract(record, ContractThresholds(contractive_progress=0.2, action_norm_high=1.0))
    assert labels["label_hit"]
    assert labels["label_good_contract"]
    assert not labels["label_unstable"]


def test_contract_labels_negative_progress_unstable():
    record = {
        "hit": False,
        "normalized_progress": -0.1,
        "action_norm_max": 1.2,
        "d_phi_start": 10.0,
        "final_d_phi": 11.0,
    }
    labels = label_closed_loop_contract(record, ContractThresholds(contractive_progress=0.2, action_norm_high=1.0))
    assert labels["label_negative_progress"]
    assert labels["label_unstable"]
    assert not labels["label_good_contract"]


def test_contract_dataset_row_preserves_features_and_labels():
    row = contract_dataset_row(
        {
            "phi_s": [0.0, 1.0],
            "phi_g": [1.0, 1.0],
            "d_phi_start": 1.0,
            "pair_source": "path_edge",
            "path_position": 3,
            "final_phase": True,
            "q_train_support": 0.7,
            "hit": False,
            "normalized_progress": 0.3,
        },
        ContractThresholds(contractive_progress=0.2),
    )
    assert row["phi_s"] == [0.0, 1.0]
    assert row["final_phase"]
    assert row["label_contractive"]
    assert row["label_good_contract"]
