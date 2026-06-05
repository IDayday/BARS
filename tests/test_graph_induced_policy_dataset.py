import sys
from pathlib import Path

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.graph_induced_policy_dataset import classify_hard_goal, hard_goal_example, has_action_supervision


def test_hard_negative_without_supervision():
    row = {
        "env_name": "humanoidmaze-large-navigate-v0",
        "target_mode": "original_target",
        "label_contract_negative": True,
        "normalized_progress": -0.2,
    }
    assert classify_hard_goal(row) == "hard_negative"
    example = hard_goal_example(row)
    assert example is not None
    assert example.category == "hard_negative"


def test_qtrain_control_is_hard_positive_when_hard():
    row = {
        "target_mode": "qtrain_matched",
        "hit": False,
        "normalized_progress": 0.0,
        "label_contract_negative": False,
    }
    assert has_action_supervision(row)
    assert classify_hard_goal(row) == "hard_positive"


def test_non_hard_goal_is_skipped():
    row = {"target_mode": "original_target", "hit": True, "normalized_progress": 0.8}
    assert classify_hard_goal(row) is None
    assert hard_goal_example(row) is None
