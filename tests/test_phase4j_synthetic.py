import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4j_loss_weighting import (
    add_loss_weight_tradeoff_metrics,
    aggregate_loss_weighting_rows,
    compare_loss_weighting_to_baseline,
    recommend_loss_weighting_methods,
)


def _rows():
    base = {
        "dataset": "dummy-v0",
        "phase2_run": "core",
        "sampling_mode": "uniform_transition",
        "loss_weight_min": 1.0,
        "loss_weight_max": 1.0,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "method": "uniform_transition_none",
                "loss_weight_mode": "none",
                "loss_weight_strength": 0.0,
                "seed": 0,
                "final_val_action_mse": 1.00,
                "best_val_action_mse": 0.95,
                "bottleneck_edge_val_mse": 1.20,
                "low_support_edge_val_mse": 1.30,
                "long_horizon_edge_val_mse": 1.10,
            },
            {
                **base,
                "method": "uniform_transition_none",
                "loss_weight_mode": "none",
                "loss_weight_strength": 0.0,
                "seed": 1,
                "final_val_action_mse": 1.00,
                "best_val_action_mse": 0.95,
                "bottleneck_edge_val_mse": 1.20,
                "low_support_edge_val_mse": 1.30,
                "long_horizon_edge_val_mse": 1.10,
            },
            {
                **base,
                "method": "loss_support_s03",
                "loss_weight_mode": "support",
                "loss_weight_strength": 0.3,
                "seed": 0,
                "final_val_action_mse": 1.03,
                "best_val_action_mse": 0.96,
                "bottleneck_edge_val_mse": 1.00,
                "low_support_edge_val_mse": 0.90,
                "long_horizon_edge_val_mse": 1.05,
            },
            {
                **base,
                "method": "loss_support_s03",
                "loss_weight_mode": "support",
                "loss_weight_strength": 0.3,
                "seed": 1,
                "final_val_action_mse": 1.04,
                "best_val_action_mse": 0.96,
                "bottleneck_edge_val_mse": 0.98,
                "low_support_edge_val_mse": 0.92,
                "long_horizon_edge_val_mse": 1.04,
            },
            {
                **base,
                "method": "loss_bad",
                "loss_weight_mode": "support_bottleneck",
                "loss_weight_strength": 0.3,
                "seed": 0,
                "final_val_action_mse": 1.20,
                "best_val_action_mse": 1.10,
                "bottleneck_edge_val_mse": 0.80,
                "low_support_edge_val_mse": 0.80,
                "long_horizon_edge_val_mse": 0.80,
            },
        ]
    )


def test_phase4j_tradeoff_metrics_average_rare_edge_groups():
    rows = add_loss_weight_tradeoff_metrics(_rows())
    first = rows.iloc[0]
    assert abs(first["rare_edge_mean_mse"] - 1.2) < 1e-12
    assert first["overall_to_rare_gap"] > 0


def test_phase4j_comparison_uses_named_baseline():
    aggregate = aggregate_loss_weighting_rows(_rows())
    comparison = compare_loss_weighting_to_baseline(aggregate, baseline_method="uniform_transition_none")
    support = comparison.set_index("method").loc["loss_support_s03"]
    assert support["final_val_action_mse_ratio_vs_baseline"] < 1.05
    assert support["rare_edge_mean_mse_ratio_vs_baseline"] < 1.0


def test_phase4j_recommendation_rejects_large_overall_regret():
    aggregate = aggregate_loss_weighting_rows(_rows())
    comparison = compare_loss_weighting_to_baseline(aggregate, baseline_method="uniform_transition_none")
    rec = recommend_loss_weighting_methods(comparison, max_overall_regret=0.05)
    assert rec.iloc[0]["method"] == "loss_support_s03"
