import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4k_loss_weighted_repair_validation import (
    aggregate_phase4k_rows,
    compare_phase4k_to_baseline,
    flatten_direct_repair_summary,
    recommend_phase4k_methods,
)


def _summary(policy_mse, certified_rate, threshold_uncertified, penalty_uncertified):
    return {
        "diagnostics": {
            "mean_direct_edge_action_mse": policy_mse,
            "median_direct_edge_action_mse": policy_mse * 0.9,
            "mean_direct_policy_support_score": 1.0 / (1.0 + policy_mse),
            "direct_certified_rate": certified_rate,
            "transfer_certified_rate": 0.7,
        },
        "method_metrics": [
            {
                "method": "support_shortest_path",
                "path_coverage": 0.5,
                "mean_uncertified_edge_fraction": 0.4,
            },
            {
                "method": "calibrated_compat_threshold",
                "path_coverage": 0.48,
                "mean_min_edge_proxy_score": 0.25,
                "mean_uncertified_edge_fraction": threshold_uncertified,
                "mean_pair_incompatible_fraction": 0.0,
                "mean_repair_edge_fraction": 0.3,
                "mean_repair_certified_fraction": 0.9,
                "mean_base_path_cost": 10.0,
            },
            {
                "method": "calibrated_compat_penalized",
                "path_coverage": 0.5,
                "mean_uncertified_edge_fraction": penalty_uncertified,
            },
        ],
    }


def _row(method, seed, final_mse, direct_mse, certified_rate, threshold_uncertified):
    return flatten_direct_repair_summary(
        summary=_summary(direct_mse, certified_rate, threshold_uncertified, penalty_uncertified=0.2),
        model_record={
            "dataset": "dummy-v0",
            "phase2_run": "core",
            "method": method,
            "seed": seed,
            "sampling_mode": "uniform_transition",
            "loss_weight_mode": "none" if method == "uniform_transition_none" else "support_bottleneck",
            "loss_weight_strength": 0.0 if method == "uniform_transition_none" else 0.3,
        },
        training_summary={"final_val_action_mse": final_mse, "best_val_action_mse": final_mse},
        planner_method="calibrated_compat_threshold",
    )


def test_phase4k_flattens_selected_planner_method():
    row = _row("uniform_transition_none", 0, final_mse=1.0, direct_mse=0.5, certified_rate=0.8, threshold_uncertified=0.1)
    assert row["planner_method"] == "calibrated_compat_threshold"
    assert row["path_coverage"] == 0.48
    assert row["mean_uncertified_edge_fraction"] == 0.1


def test_phase4k_baseline_comparison_uses_direct_repair_mse():
    rows = pd.DataFrame(
        [
            _row("uniform_transition_none", 0, 1.0, 0.50, 0.80, 0.10),
            _row("uniform_transition_none", 1, 1.0, 0.50, 0.80, 0.10),
            _row("loss_support_bottleneck_s03", 0, 1.02, 0.40, 0.90, 0.05),
            _row("loss_support_bottleneck_s03", 1, 1.04, 0.42, 0.92, 0.05),
        ]
    )
    comparison = compare_phase4k_to_baseline(aggregate_phase4k_rows(rows))
    weighted = comparison.set_index("method").loc["loss_support_bottleneck_s03"]
    assert weighted["final_val_action_mse_ratio_vs_baseline"] < 1.05
    assert weighted["mean_direct_edge_action_mse_ratio_vs_baseline"] < 1.0
    assert weighted["direct_certified_rate_delta_vs_baseline"] > 0


def test_phase4k_recommendation_rejects_direct_mse_regression_when_possible():
    rows = pd.DataFrame(
        [
            _row("uniform_transition_none", 0, 1.0, 0.50, 0.80, 0.10),
            _row("loss_bad_direct", 0, 1.01, 0.70, 0.90, 0.02),
            _row("loss_good_direct", 0, 1.03, 0.45, 0.86, 0.08),
        ]
    )
    comparison = compare_phase4k_to_baseline(aggregate_phase4k_rows(rows))
    rec = recommend_phase4k_methods(comparison, max_overall_regret=0.05, require_direct_mse_improvement=True)
    assert rec.iloc[0]["method"] == "loss_good_direct"
