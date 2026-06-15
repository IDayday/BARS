import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4o_regret_guard import (  # noqa: E402
    RegretGuardConfig,
    annotate_regret_guard_candidates,
    select_regret_guard_candidate,
)


def _comparison_table():
    return pd.DataFrame(
        {
            "dataset": ["scene-play-v0"] * 4,
            "phase2_run": ["core_plus_bottleneck_budget192_H10"] * 4,
            "method": [
                "augmented_loss_support_bottleneck_s03",
                "planner_relevant_repair_s01",
                "planner_relevant_repair_s02",
                "planner_relevant_repair_s04",
            ],
            "final_val_action_mse_ratio_vs_baseline": [1.0, 1.0067, 1.0058, 1.0088],
            "direct_repair_edge_mse_ratio_vs_baseline": [1.0, 1.0006, 0.9883, 0.9935],
            "planner_used_repair_edge_mse_ratio_vs_baseline": [1.0, 0.9949, 0.9605, 0.9625],
            "direct_repair_policy_support_score_ratio_vs_baseline": [1.0, 0.9989, 1.0020, 1.0002],
        }
    )


def _h25_like_table():
    return pd.DataFrame(
        {
            "dataset": ["scene-play-v0"] * 3,
            "phase2_run": ["core_plus_bottleneck_budget192_H25"] * 3,
            "method": [
                "augmented_loss_support_bottleneck_s03",
                "planner_relevant_repair_s02",
                "planner_relevant_repair_s04",
            ],
            "final_val_action_mse_ratio_vs_baseline": [1.0, 0.9946, 0.9924],
            "direct_repair_edge_mse_ratio_vs_baseline": [1.0, 0.9928, 0.9836],
            "planner_used_repair_edge_mse_ratio_vs_baseline": [1.0, 1.0090, 0.9914],
            "direct_repair_policy_support_score_ratio_vs_baseline": [1.0, 1.0016, 1.0047],
        }
    )


def test_regret_guard_selects_scene_h10_s02_over_aggressive_s04():
    selection = select_regret_guard_candidate(_comparison_table())
    assert selection["selected_method"] == "planner_relevant_repair_s02"
    assert selection["selection_status"] == "guard_pass"
    assert selection["num_guard_pass_candidates"] == 2


def test_regret_guard_falls_back_to_baseline_when_no_candidate_passes():
    table = _comparison_table()
    config = RegretGuardConfig(max_final_val_ratio=1.001)
    selection = select_regret_guard_candidate(table, config)
    assert selection["selected_method"] == "augmented_loss_support_bottleneck_s03"
    assert selection["selected_is_baseline"] is True
    assert selection["selection_status"] == "fallback_baseline_no_guard_pass"


def test_relaxed_guard_selects_all_metric_improver_below_strict_planner_gain():
    selection = select_regret_guard_candidate(_h25_like_table())
    assert selection["selected_method"] == "planner_relevant_repair_s04"
    assert selection["selection_status"] == "relaxed_guard_pass"
    assert selection["num_guard_pass_candidates"] == 0
    assert selection["num_relaxed_guard_pass_candidates"] == 1


def test_disabling_relaxed_guard_keeps_strict_baseline_fallback():
    config = RegretGuardConfig(allow_relaxed_improvement_fallback=False)
    selection = select_regret_guard_candidate(_h25_like_table(), config)
    assert selection["selected_method"] == "augmented_loss_support_bottleneck_s03"
    assert selection["selection_status"] == "fallback_baseline_no_guard_pass"


def test_regret_guard_records_violation_reasons():
    annotated = annotate_regret_guard_candidates(_comparison_table())
    s01 = annotated[annotated["method"] == "planner_relevant_repair_s01"].iloc[0]
    assert bool(s01["guard_pass"]) is False
    assert "direct_repair_mse" in s01["guard_violation_reasons"]
    assert "policy_support_score" in s01["guard_violation_reasons"]
    s02 = annotated[annotated["method"] == "planner_relevant_repair_s02"].iloc[0]
    assert bool(s02["guard_pass"]) is True
    assert s02["guard_violation_reasons"] == ""
