from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mine_action_supervised_contract_examples import mine_examples


def test_action_supervised_mining_marks_first_action_only():
    rows = [
        {
            "env_name": "toy",
            "seed": 0,
            "source_segment_id": "toy__seed0__task0__ep0__seg0",
            "phi_s": [0.0],
            "phi_g": [1.0],
            "target_mode": "original_target",
            "label_contract_positive": True,
        }
    ]
    segments = {"toy__seed0__task0__ep0__seg0": {"first_action": [0.1, -0.1], "start_state_ref": {"reset_mode": "exact_mujoco_state"}}}

    examples = mine_examples(rows, segments)

    assert len(examples) == 1
    assert examples[0]["action_available"] is True
    assert examples[0]["supervision_quality"] == "first_action_only"
    assert examples[0]["first_action"] == [0.1, -0.1]


def test_action_supervised_mining_reports_missing_action_fields():
    rows = [{"source_segment_id": "seg0", "phi_s": [0.0], "phi_g": [1.0], "label_contract_positive": True}]
    examples = mine_examples(rows, {"seg0": {"start_state_ref": {}}})

    assert examples[0]["action_available"] is False
    assert examples[0]["failure_reason_if_missing"] == "segment_capture_has_no_action_fields"
