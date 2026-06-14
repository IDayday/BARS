import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4h_scene_validation import (
    build_phase4h_summary,
    diagnostics_delta_frame,
    method_delta_frame,
    write_phase4h_outputs,
)


def _summary(action_mse: float, certified_rate: float, coverage: float, uncertified: float) -> dict:
    return {
        "diagnostics": {
            "mean_direct_edge_action_mse": action_mse,
            "direct_certified_rate": certified_rate,
            "mean_direct_policy_support_score": 0.5,
            "note": "offline only",
        },
        "method_metrics": [
            {
                "method": "calibrated_compat_threshold",
                "path_coverage": coverage,
                "mean_uncertified_edge_fraction": uncertified,
                "mean_pair_incompatible_fraction": 0.0,
                "mean_min_edge_proxy_score": 0.2,
            }
        ],
    }


def test_phase4h_diagnostic_delta_tracks_candidate_minus_baseline():
    baseline = _summary(action_mse=0.08, certified_rate=0.4, coverage=0.3, uncertified=0.2)
    candidate = _summary(action_mse=0.05, certified_rate=0.7, coverage=0.3, uncertified=0.1)
    delta = diagnostics_delta_frame(baseline, candidate, "base", "candidate")
    by_metric = delta.set_index("metric")
    assert abs(by_metric.loc["mean_direct_edge_action_mse", "delta_candidate_minus_baseline"] + 0.03) < 1e-12
    assert abs(by_metric.loc["direct_certified_rate", "delta_candidate_minus_baseline"] - 0.3) < 1e-12


def test_phase4h_method_delta_keeps_planner_metrics_by_method():
    baseline = _summary(action_mse=0.08, certified_rate=0.4, coverage=0.3, uncertified=0.2)
    candidate = _summary(action_mse=0.05, certified_rate=0.7, coverage=0.45, uncertified=0.1)
    delta = method_delta_frame(baseline, candidate, "base", "candidate")
    rows = delta.set_index(["method", "metric"])
    assert (
        abs(rows.loc[("calibrated_compat_threshold", "path_coverage"), "delta_candidate_minus_baseline"] - 0.15)
        < 1e-12
    )
    assert abs(
        rows.loc[
        ("calibrated_compat_threshold", "mean_uncertified_edge_fraction"),
        "delta_candidate_minus_baseline",
        ]
        + 0.1
    ) < 1e-12


def test_phase4h_summary_outputs_offline_proxy_warning(tmp_path):
    baseline = _summary(action_mse=0.08, certified_rate=0.4, coverage=0.3, uncertified=0.2)
    candidate = _summary(action_mse=0.05, certified_rate=0.7, coverage=0.45, uncertified=0.1)
    diag = diagnostics_delta_frame(baseline, candidate, "base", "candidate")
    methods = method_delta_frame(baseline, candidate, "base", "candidate")
    payload = build_phase4h_summary(
        config={"candidate_label": "candidate"},
        training_summary={"final_val_action_mse": 0.05},
        baseline_summary=baseline,
        candidate_summary=candidate,
        diagnostic_deltas=diag,
        method_deltas=methods,
    )
    write_phase4h_outputs(tmp_path, payload, diag, methods)
    summary = json.loads((tmp_path / "phase4h_scene_gcbc_repair_validation_summary.json").read_text())
    md = (tmp_path / "phase4h_scene_gcbc_repair_validation_summary.md").read_text()
    assert summary["note"] == "This is reset-free offline supervised evidence. It is not rollout success."
    assert "not rollout success" in md
    assert pd.read_csv(tmp_path / "phase4h_method_delta.csv").shape[0] > 0
