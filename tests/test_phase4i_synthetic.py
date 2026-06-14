import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4i_sampling_study import (
    add_tradeoff_metrics,
    aggregate_sampling_tradeoffs,
    compare_to_baseline,
    recommend_sampling_modes,
)


def _rows():
    return pd.DataFrame(
        [
            {
                "dataset": "dummy-v0",
                "phase2_run": "core",
                "sampling_mode": "uniform_transition",
                "seed": 0,
                "final_val_action_mse": 1.0,
                "best_val_action_mse": 0.9,
                "bottleneck_edge_val_mse": 1.2,
                "low_support_edge_val_mse": 1.4,
                "long_horizon_edge_val_mse": 1.1,
            },
            {
                "dataset": "dummy-v0",
                "phase2_run": "core",
                "sampling_mode": "uniform_transition",
                "seed": 1,
                "final_val_action_mse": 1.1,
                "best_val_action_mse": 1.0,
                "bottleneck_edge_val_mse": 1.3,
                "low_support_edge_val_mse": 1.5,
                "long_horizon_edge_val_mse": 1.2,
            },
            {
                "dataset": "dummy-v0",
                "phase2_run": "core",
                "sampling_mode": "support_balanced",
                "seed": 0,
                "final_val_action_mse": 1.03,
                "best_val_action_mse": 0.95,
                "bottleneck_edge_val_mse": 0.9,
                "low_support_edge_val_mse": 0.8,
                "long_horizon_edge_val_mse": 1.0,
            },
            {
                "dataset": "dummy-v0",
                "phase2_run": "core",
                "sampling_mode": "support_balanced",
                "seed": 1,
                "final_val_action_mse": 1.04,
                "best_val_action_mse": 0.96,
                "bottleneck_edge_val_mse": 0.95,
                "low_support_edge_val_mse": 0.85,
                "long_horizon_edge_val_mse": 1.0,
            },
            {
                "dataset": "dummy-v0",
                "phase2_run": "core",
                "sampling_mode": "bad_rare",
                "seed": 0,
                "final_val_action_mse": 0.5,
                "best_val_action_mse": 0.5,
                "bottleneck_edge_val_mse": 2.0,
                "low_support_edge_val_mse": 2.0,
                "long_horizon_edge_val_mse": 2.0,
            },
        ]
    )


def test_phase4i_rare_edge_mean_mse_is_group_average():
    rows = add_tradeoff_metrics(_rows())
    first = rows.iloc[0]
    assert abs(first["rare_edge_mean_mse"] - ((1.2 + 1.4 + 1.1) / 3.0)) < 1e-12
    assert first["overall_to_rare_gap"] > 0


def test_phase4i_comparison_uses_uniform_transition_baseline():
    summary = aggregate_sampling_tradeoffs(_rows())
    comparison = compare_to_baseline(summary, baseline_sampling_mode="uniform_transition")
    support = comparison.set_index("sampling_mode").loc["support_balanced"]
    assert support["final_val_action_mse_ratio_vs_baseline"] < 1.05
    assert support["rare_edge_mean_mse_ratio_vs_baseline"] < 1.0


def test_phase4i_recommendation_prefers_rare_gain_with_small_overall_regret():
    summary = aggregate_sampling_tradeoffs(_rows())
    comparison = compare_to_baseline(summary, baseline_sampling_mode="uniform_transition")
    rec = recommend_sampling_modes(comparison, max_overall_regret=0.05)
    assert rec.iloc[0]["sampling_mode"] == "support_balanced"
