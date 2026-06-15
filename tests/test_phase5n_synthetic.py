from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3.edge_rollout import policy_action
from phase3.models import GCBCMLP
from phase5n.planner_subgoal_dataset import (
    TARGET_SOURCE_TO_ID,
    PlannerMixedGCBCDataset,
    build_planner_relevant_edge_weights,
    episode_bounds_from_terminals,
    sample_support_graph_planner_paths,
)


def _synthetic_edges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"edge_id": 0, "src": 1, "dst": 2, "cost": 1.0, "median_h": 2.0, "num_unique_starts": 2, "edge_bottleneck_score": 0.1},
            {"edge_id": 1, "src": 2, "dst": 3, "cost": 1.0, "median_h": 2.0, "num_unique_starts": 2, "edge_bottleneck_score": 0.8},
            {"edge_id": 2, "src": 1, "dst": 3, "cost": 5.0, "median_h": 5.0, "num_unique_starts": 1, "edge_bottleneck_score": 0.2},
        ]
    )


def _synthetic_segments() -> dict[str, np.ndarray]:
    return {
        "edge_id": np.asarray([0, 1, 2], dtype=np.int64),
        "ep_id": np.asarray([0, 0, 0], dtype=np.int64),
        "t": np.asarray([0, 2, 0], dtype=np.int64),
        "h": np.asarray([2, 2, 4], dtype=np.int64),
        "global_i": np.asarray([0, 2, 0], dtype=np.int64),
        "global_j": np.asarray([2, 4, 4], dtype=np.int64),
    }


def _synthetic_dataset() -> dict[str, np.ndarray]:
    observations = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [10.0], [11.0], [12.0]], dtype=np.float32)
    actions = np.ones((8, 1), dtype=np.float32)
    terminals = np.asarray([False, False, False, False, True, False, False, True])
    return {"observations": observations, "actions": actions, "terminals": terminals}


def test_planner_paths_prefer_supported_shortest_path_over_direct_expensive_edge():
    paths = sample_support_graph_planner_paths(_synthetic_edges(), queries=[(1, 3)], seed=0)
    row = paths.iloc[0]
    assert bool(row["reachable"])
    assert row["path_edge_ids"] == "0 1"
    assert int(row["first_edge_id"]) == 0


def test_planner_relevant_weights_lift_first_edge_usage():
    paths = sample_support_graph_planner_paths(_synthetic_edges(), queries=[(1, 3)] * 10, seed=0)
    weights, _ = build_planner_relevant_edge_weights(
        _synthetic_edges(),
        path_rows=paths,
        planner_usage_strength=0.0,
        planner_first_edge_strength=1.0,
        base_loss_weight_mode="none",
    )
    row0 = weights.set_index("edge_id").loc[0]
    row1 = weights.set_index("edge_id").loc[1]
    assert row0["planner_first_edge_count"] == 10
    assert row1["planner_first_edge_count"] == 0
    assert row0["loss_weight"] > row1["loss_weight"]


def test_episode_bounds_and_split_preserve_episode_semantics():
    dataset = _synthetic_dataset()
    bounds = episode_bounds_from_terminals(dataset["terminals"], dataset["observations"].shape[0])
    assert bounds.tolist() == [[0, 5], [5, 8]]
    ds = PlannerMixedGCBCDataset(
        dataset,
        _synthetic_edges(),
        _synthetic_segments(),
        max_examples=32,
        source_probabilities={"final_goal_hindsight": 1.0, "support_edge_local": 0.0, "planner_first_edge_replay": 0.0},
        seed=3,
    )
    train, val = ds.split(0.5, seed=0)
    assert train.episode_bounds.shape[0] >= 1
    assert val.episode_bounds.shape[0] >= 1
    for i in range(8):
        sample = train[i]
        assert int(sample["target_source_id"]) == TARGET_SOURCE_TO_ID["final_goal_hindsight"]
        assert sample["goal"].shape == sample["obs"].shape


def test_planner_replay_samples_only_real_segment_targets():
    paths = sample_support_graph_planner_paths(_synthetic_edges(), queries=[(1, 3)] * 5, seed=0)
    weights, _ = build_planner_relevant_edge_weights(
        _synthetic_edges(),
        path_rows=paths,
        planner_first_edge_strength=2.0,
        base_loss_weight_mode="none",
    )
    ds = PlannerMixedGCBCDataset(
        _synthetic_dataset(),
        _synthetic_edges(),
        _synthetic_segments(),
        planner_edge_weights=weights,
        max_examples=64,
        source_probabilities={"final_goal_hindsight": 0.0, "support_edge_local": 0.0, "planner_first_edge_replay": 1.0},
        seed=11,
    )
    for i in range(12):
        sample = ds[i]
        assert int(sample["target_source_id"]) == TARGET_SOURCE_TO_ID["planner_first_edge_replay"]
        assert int(sample["edge_id"]) in {0, 1, 2}
        assert torch.isfinite(sample["sample_weight"])
        assert float(sample["remaining_h"]) >= 1.0


def test_mixed_dataset_source_probabilities_can_include_all_sources():
    ds = PlannerMixedGCBCDataset(
        _synthetic_dataset(),
        _synthetic_edges(),
        _synthetic_segments(),
        max_examples=128,
        source_probabilities={"final_goal_hindsight": 0.3, "support_edge_local": 0.3, "planner_first_edge_replay": 0.4},
        seed=21,
    )
    seen = {int(ds[i]["target_source_id"]) for i in range(80)}
    assert TARGET_SOURCE_TO_ID["final_goal_hindsight"] in seen
    assert TARGET_SOURCE_TO_ID["support_edge_local"] in seen
    assert TARGET_SOURCE_TO_ID["planner_first_edge_replay"] in seen


def test_source_head_gcbc_uses_target_source_id_and_policy_action_infers_runtime_source():
    torch.manual_seed(0)
    model = GCBCMLP(
        obs_dim=2,
        action_dim=1,
        hidden_dims=[8],
        use_remaining_h=True,
        remaining_h_scale=10.0,
        num_target_sources=3,
        target_source_embedding_dim=4,
        target_source_head_mode="heads",
    )
    obs = torch.zeros(4, 2)
    goal = torch.ones(4, 2)
    remaining = torch.ones(4)
    edge = torch.zeros(4, dtype=torch.long)
    source0 = torch.zeros(4, dtype=torch.long)
    source2 = torch.full((4,), 2, dtype=torch.long)
    out0 = model(obs, goal, remaining, edge, source0)
    out2 = model(obs, goal, remaining, edge, source2)
    assert out0.shape == (4, 1)
    assert out2.shape == (4, 1)
    assert not torch.allclose(out0, out2)

    direct = policy_action(model, np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), remaining_h=5)
    edge_action = policy_action(
        model,
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        remaining_h=5,
        edge_id=0,
    )
    explicit = policy_action(
        model,
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        remaining_h=5,
        edge_id=0,
        target_source_id=2,
    )
    assert direct.shape == (1,)
    assert edge_action.shape == (1,)
    assert np.allclose(edge_action, explicit)
    assert not np.allclose(direct, edge_action)
