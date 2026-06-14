import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.compatibility_aware_planning import CompatibilityPlannerConfig
from phase3e.compatibility_graph_repair import (
    GraphRepairConfig,
    build_augmented_graph_inputs,
    run_repair_evaluation,
    select_repair_edges,
)


def _base_edges():
    return pd.DataFrame(
        [
            {"edge_id": 0, "src": 1, "dst": 2, "num_segments": 10, "num_episodes": 2, "num_unique_starts": 5, "median_h": 1.0, "cost": 1.0},
            {"edge_id": 1, "src": 2, "dst": 4, "num_segments": 10, "num_episodes": 2, "num_unique_starts": 5, "median_h": 1.0, "cost": 1.0},
        ]
    )


def _bank_edges():
    return pd.DataFrame(
        [
            {"edge_id": 10, "src": 1, "dst": 2, "num_segments": 100, "num_episodes": 4, "num_unique_starts": 10, "median_h": 1.0, "cost": 1.0},
            {"edge_id": 11, "src": 2, "dst": 3, "num_segments": 50, "num_episodes": 3, "num_unique_starts": 8, "median_h": 1.0, "cost": 1.0},
            {"edge_id": 12, "src": 3, "dst": 4, "num_segments": 40, "num_episodes": 3, "num_unique_starts": 8, "median_h": 1.0, "cost": 1.0},
            {"edge_id": 13, "src": 5, "dst": 6, "num_segments": 60, "num_episodes": 3, "num_unique_starts": 8, "median_h": 1.0, "cost": 1.0},
        ]
    )


def _pair_df():
    return pd.DataFrame(
        [
            {
                "edge_id_first": 0,
                "edge_id_second": 1,
                "first_src": 1,
                "junction": 2,
                "second_dst": 4,
                "termination_bridge_coverage": 0.0,
                "strict_compatible": False,
            }
        ]
    )


def _segments():
    base = {
        "edge_id": np.array([0, 1], dtype=np.int64),
        "ep_id": np.array([0, 0], dtype=np.int64),
        "global_i": np.array([0, 20], dtype=np.int64),
        "global_j": np.array([2, 22], dtype=np.int64),
        "h": np.array([2, 2], dtype=np.int64),
        "t": np.array([0, 20], dtype=np.int64),
    }
    bank = {
        "edge_id": np.array([10, 11, 12], dtype=np.int64),
        "ep_id": np.array([0, 0, 0], dtype=np.int64),
        "global_i": np.array([0, 3, 6], dtype=np.int64),
        "global_j": np.array([2, 5, 8], dtype=np.int64),
        "h": np.array([2, 2, 2], dtype=np.int64),
        "t": np.array([0, 3, 6], dtype=np.int64),
    }
    return base, bank


def test_select_repair_edges_targets_bad_junction_and_skips_duplicates():
    selected = select_repair_edges(
        _base_edges(),
        _bank_edges(),
        _pair_df(),
        GraphRepairConfig(max_repair_edges=2, min_pair_coverage=0.05),
    )
    selected_pairs = {(int(row.src), int(row.dst)) for row in selected.itertuples(index=False)}
    assert (1, 2) not in selected_pairs
    assert (2, 3) in selected_pairs
    assert selected.iloc[0]["repair_reason"] == "low_compatibility_junction"


def test_build_augmented_graph_reassigns_repair_edge_ids_and_segments():
    base_segments, bank_segments = _segments()
    selected = _bank_edges()[_bank_edges()["edge_id"].isin([11, 12])].copy()
    augmented_edges, augmented_segments, repair_map = build_augmented_graph_inputs(
        _base_edges(),
        base_segments,
        selected,
        bank_segments,
    )
    assert augmented_edges["edge_id"].is_unique
    assert set(repair_map["bank_edge_id"]) == {11, 12}
    assert set(augmented_segments["edge_id"]) == {0, 1, 2, 3}


def test_repair_edges_can_restore_threshold_path_coverage():
    base_segments, bank_segments = _segments()
    outputs = run_repair_evaluation(
        base_edges=_base_edges(),
        base_segments=base_segments,
        bank_edges=_bank_edges(),
        bank_segments=bank_segments,
        path_queries=pd.DataFrame([{"query_id": 0, "start_cluster": 1, "goal_cluster": 4}]),
        certification=None,
        repair_config=GraphRepairConfig(max_repair_edges=2, min_pair_coverage=0.05),
        planner_config=CompatibilityPlannerConfig(min_pair_coverage=0.05, pair_weight=10.0),
        methods=["compat_threshold"],
        H_intra=2,
        max_queries=None,
        seed=0,
    )
    summary = outputs["summary"].set_index("graph_variant")
    assert summary.loc["base", "path_coverage"] == 0.0
    assert summary.loc["repaired", "path_coverage"] == 1.0
    repaired_path = outputs["path_metrics"][
        (outputs["path_metrics"]["graph_variant"] == "repaired")
        & (outputs["path_metrics"]["method"] == "compat_threshold")
    ].iloc[0]
    assert repaired_path["path_edge_ids"] == "0 2 3"

