import sys
from pathlib import Path
from argparse import Namespace
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phase1.clustering import assign_clusters, fit_state_clusters
from phase3.edge_bc_dataset import build_edge_bc_examples
from phase3.edge_rollout import dst_cluster_success
from phase3.models import GCBCMLP
from phase3.train_gcbc import edge_loss_weight_values, weighted_action_mse
from phase3.reset_utils import (
    RESET_STATUS_ENV_UNAVAILABLE,
    env_unavailable_probe_result,
    probe_reset_capability,
)
from phase3f.natural_rollout import run_natural_start_episodes, write_natural_rollout_outputs
from phase3f.hierarchical_rollout import (
    build_support_planning_graph,
    choose_edge_subgoal,
    load_or_fit_runtime_cluster_model,
    run_hierarchical_support_episodes,
)
from phase3f.edge_memory import (
    extract_edge_attempts_from_traces,
    memory_failed_edge_counts,
    merge_edge_memory,
    summarize_edge_attempts,
)
from phase3f.edge_outcome_model import edge_outcome_penalty_map, fit_edge_outcome_scores
from scripts.eval_phase3_edge_execution import _write_rollout_skip


def _dataset(n=128):
    obs = np.arange(n, dtype=np.float32).reshape(n, 1)
    actions = (obs * 0.1 + 1.0).astype(np.float32)
    return {
        "observations": obs,
        "actions": actions,
        "next_observations": np.vstack([obs[1:], obs[-1:]]),
        "terminals": np.zeros(n, dtype=bool),
    }


def _edges():
    return pd.DataFrame(
        {
            "edge_id": [0, 1],
            "src": [0, 1],
            "dst": [1, 2],
            "num_segments": [1, 1],
            "num_episodes": [1, 1],
            "num_unique_starts": [1, 1],
            "num_unique_episodes": [1, 1],
            "median_h": [3.0, 1.0],
            "max_h": [3.0, 1.0],
            "edge_bottleneck_score": [0.1, 10.0],
        }
    )


def test_edge_segment_expands_to_one_step_bc_samples():
    segments = {
        "edge_id": np.asarray([0], dtype=np.int64),
        "ep_id": np.asarray([0], dtype=np.int64),
        "t": np.asarray([0], dtype=np.int64),
        "h": np.asarray([3], dtype=np.int64),
        "global_i": np.asarray([0], dtype=np.int64),
        "global_j": np.asarray([3], dtype=np.int64),
    }
    ds = build_edge_bc_examples(_dataset(8), _edges().iloc[:1], segments, sampling_mode="uniform_transition")
    assert len(ds) == 3
    first = ds[0]
    third = ds[2]
    assert torch.equal(first["obs"], torch.tensor([0.0]))
    assert torch.equal(first["goal"], torch.tensor([3.0]))
    assert torch.equal(first["action"], torch.tensor([1.0]))
    assert first["remaining_h"].item() == 3.0
    assert torch.equal(third["obs"], torch.tensor([2.0]))
    assert third["remaining_h"].item() == 1.0


def test_uniform_edge_sampling_not_dominated_by_large_edge():
    segments = {
        "edge_id": np.asarray([0, 1], dtype=np.int64),
        "ep_id": np.asarray([0, 0], dtype=np.int64),
        "t": np.asarray([0, 110], dtype=np.int64),
        "h": np.asarray([100, 1], dtype=np.int64),
        "global_i": np.asarray([0, 110], dtype=np.int64),
        "global_j": np.asarray([100, 111], dtype=np.int64),
    }
    ds = build_edge_bc_examples(_dataset(128), _edges(), segments, sampling_mode="uniform_edge", seed=3)
    sampled = ds.sample_edge_ids(2000, seed=4)
    small_edge_fraction = float(np.mean(sampled == 1))
    assert 0.4 <= small_edge_fraction <= 0.6


def test_bottleneck_weighted_sampling_prefers_high_bottleneck_edge():
    segments = {
        "edge_id": np.asarray([0, 1], dtype=np.int64),
        "ep_id": np.asarray([0, 0], dtype=np.int64),
        "t": np.asarray([0, 10], dtype=np.int64),
        "h": np.asarray([2, 2], dtype=np.int64),
        "global_i": np.asarray([0, 10], dtype=np.int64),
        "global_j": np.asarray([2, 12], dtype=np.int64),
    }
    ds = build_edge_bc_examples(_dataset(32), _edges(), segments, sampling_mode="bottleneck_weighted", seed=1)
    sampled = ds.sample_edge_ids(2000, seed=2)
    assert float(np.mean(sampled == 1)) > 0.9


def test_support_balanced_sampling_prefers_low_support_edge():
    segments = {
        "edge_id": np.asarray([0, 1], dtype=np.int64),
        "ep_id": np.asarray([0, 0], dtype=np.int64),
        "t": np.asarray([0, 10], dtype=np.int64),
        "h": np.asarray([2, 2], dtype=np.int64),
        "global_i": np.asarray([0, 10], dtype=np.int64),
        "global_j": np.asarray([2, 12], dtype=np.int64),
    }
    edges = _edges().copy()
    edges["num_unique_starts"] = [100, 1]
    ds = build_edge_bc_examples(_dataset(32), edges, segments, sampling_mode="support_balanced", seed=4)
    sampled = ds.sample_edge_ids(2000, seed=5)
    assert float(np.mean(sampled == 1)) > 0.85


def test_bottleneck_support_balanced_combines_support_and_bottleneck():
    segments = {
        "edge_id": np.asarray([0, 1, 2], dtype=np.int64),
        "ep_id": np.asarray([0, 0, 0], dtype=np.int64),
        "t": np.asarray([0, 10, 20], dtype=np.int64),
        "h": np.asarray([2, 2, 2], dtype=np.int64),
        "global_i": np.asarray([0, 10, 20], dtype=np.int64),
        "global_j": np.asarray([2, 12, 22], dtype=np.int64),
    }
    edges = pd.DataFrame(
        {
            "edge_id": [0, 1, 2],
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "num_segments": [100, 1, 1],
            "num_episodes": [5, 1, 1],
            "num_unique_starts": [100, 1, 1],
            "num_unique_episodes": [5, 1, 1],
            "median_h": [2.0, 2.0, 2.0],
            "max_h": [2.0, 2.0, 2.0],
            "edge_bottleneck_score": [0.0, 0.0, 10.0],
        }
    )
    ds = build_edge_bc_examples(
        _dataset(32),
        edges,
        segments,
        sampling_mode="bottleneck_support_balanced",
        seed=6,
    )
    sampled = ds.sample_edge_ids(3000, seed=7)
    assert float(np.mean(sampled == 2)) > float(np.mean(sampled == 1))
    assert float(np.mean(sampled == 2)) > 0.7


def test_support_bottleneck_loss_weights_prioritize_rare_bottleneck_edges():
    edges = pd.DataFrame(
        {
            "edge_id": [0, 1, 2],
            "num_unique_starts": [100, 1, 1],
            "edge_bottleneck_score": [0.0, 0.0, 10.0],
        }
    )
    weights = edge_loss_weight_values(
        edges,
        mode="support_bottleneck",
        strength=1.0,
        min_weight=0.25,
        max_weight=4.0,
    ).set_index("edge_id")
    assert weights.loc[2, "loss_weight"] > weights.loc[1, "loss_weight"]
    assert weights.loc[1, "loss_weight"] > weights.loc[0, "loss_weight"]


def test_weighted_action_mse_increases_high_weight_error_contribution():
    pred = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([[1.0], [3.0]])
    unweighted = torch.mean((pred - target) ** 2)
    weighted = weighted_action_mse(pred, target, torch.tensor([1.0, 3.0]))
    assert weighted.item() > unweighted.item()


def test_gcbc_mlp_forward_action_shape():
    model = GCBCMLP(obs_dim=3, action_dim=2, hidden_dims=[16, 16], use_remaining_h=True)
    obs = torch.zeros(5, 3)
    goal = torch.ones(5, 3)
    remaining_h = torch.ones(5)
    out = model(obs, goal, remaining_h)
    assert out.shape == (5, 2)


def test_dst_cluster_success_uses_cluster_assignment():
    observations = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 0.0],
            [10.0, 1.0],
        ],
        dtype=np.float32,
    )
    cluster_model = fit_state_clusters(
        observations,
        method="grid_xy",
        n_clusters=4,
        seed=0,
        state_dims=[0, 1],
        n_bins_x=2,
        n_bins_y=2,
    )
    labels = assign_clusters(observations, cluster_model)
    assert dst_cluster_success(observations[2], int(labels[2]), cluster_model)
    assert not dst_cluster_success(observations[2], int(labels[0]), cluster_model)


def test_reset_probe_detects_dummy_set_state():
    class DummyUnwrapped:
        def __init__(self):
            self.state = None

        def set_state(self, obs):
            self.state = np.asarray(obs).copy()

    class DummyEnv:
        def __init__(self):
            self.unwrapped = DummyUnwrapped()

    env = DummyEnv()
    result = probe_reset_capability(env, np.asarray([1.0, 2.0], dtype=np.float32))
    assert result["reset_supported"]
    assert result["env_available"]
    assert result["reset_probe_status"] == "reset_supported"
    assert result["method"] == "set_state(obs)"
    assert np.allclose(env.unwrapped.state, [1.0, 2.0])


def test_env_unavailable_status_uses_nullable_reset_supported():
    result = env_unavailable_probe_result(
        "env_construction_failed: No module named gymnasium",
        missing_packages=["gymnasium", "gym"],
        num_probe_states=3,
    )
    assert result["env_available"] is False
    assert result["reset_probe_status"] == RESET_STATUS_ENV_UNAVAILABLE
    assert result["reset_supported"] is None
    assert result["reset_method"] is None
    assert result["missing_packages"] == ["gymnasium", "gym"]


def test_rollout_skip_summary_preserves_env_unavailable_status(tmp_path):
    probe = env_unavailable_probe_result(
        "env_construction_failed",
        missing_packages=["gymnasium", "gym"],
        num_probe_states=1,
    )
    args = Namespace(
        dataset_name="dummy-v0",
        phase2_run_dir="results/phase2/dummy/core_plus_bottleneck_budget1_H1",
    )
    _write_rollout_skip(tmp_path, args, probe, RESET_STATUS_ENV_UNAVAILABLE)
    summary = json.loads((tmp_path / "edge_execution_summary.json").read_text())
    assert summary["rollout_skipped"] is True
    assert summary["skipped_reason"] == RESET_STATUS_ENV_UNAVAILABLE
    assert summary["reset_probe"]["reset_probe_status"] == RESET_STATUS_ENV_UNAVAILABLE
    assert summary["reset_probe"]["reset_supported"] is None


def test_natural_start_direct_rollout_uses_env_goal_and_writes_summary(tmp_path):
    class DummyActionSpace:
        shape = (1,)
        low = np.asarray([-1.0], dtype=np.float32)
        high = np.asarray([1.0], dtype=np.float32)

        def sample(self):
            return np.asarray([0.0], dtype=np.float32)

    class DummyEnv:
        def __init__(self):
            self.action_space = DummyActionSpace()
            self.x = 0.0
            self.goal = np.asarray([1.0], dtype=np.float32)

        def reset(self, seed=None, options=None):
            del seed, options
            self.x = 0.0
            return np.asarray([self.x], dtype=np.float32), {"goal": self.goal.copy()}

        def step(self, action):
            self.x = float(np.clip(self.x + float(action[0]), -10.0, 10.0))
            success = self.x >= 1.0
            return (
                np.asarray([self.x], dtype=np.float32),
                1.0 if success else 0.0,
                success,
                False,
                {"success": success},
            )

    class DummyPolicy:
        def __call__(self, obs, goal):
            del obs, goal
            return np.asarray([0.6], dtype=np.float32)

    episodes, traces = run_natural_start_episodes(
        DummyEnv(),
        DummyPolicy(),
        dataset_name="dummy-v0",
        method="direct_gcbc",
        num_episodes=1,
        max_steps=5,
        seed=0,
        action_mode="direct_gcbc",
    )
    assert episodes.loc[0, "success"] == 1.0
    assert episodes.loc[0, "num_steps"] == 2
    assert traces[0]["steps"][-1]["success"] == 1.0
    summary = write_natural_rollout_outputs(
        tmp_path,
        dataset_name="dummy-v0",
        method="direct_gcbc",
        episodes=episodes,
        traces=traces,
    )
    assert summary.loc[0, "success_rate"] == 1.0
    assert (tmp_path / "episode_traces.jsonl").read_text().strip()


def test_support_planning_graph_adds_only_support_bank_connectors():
    graph_edges = pd.DataFrame(
        {
            "edge_id": [5],
            "src": [1],
            "dst": [2],
            "median_h": [1.0],
            "cost": [1.0],
        }
    )
    bank_edges = pd.DataFrame(
        {
            "edge_id": [7, 8, 9],
            "src": [0, 9, 8],
            "dst": [1, 2, 9],
            "median_h": [1.0, 1.0, 1.0],
            "cost": [1.0, 1.0, 1.0],
        }
    )
    graph = build_support_planning_graph(
        graph_edges,
        bank_edges=bank_edges,
        start_cluster=0,
        goal_cluster=2,
        allow_bank_connectors=True,
    )
    assert graph.has_edge(0, 1)
    assert graph.has_edge(1, 2)
    assert graph.has_edge(9, 2)
    assert not graph.has_edge(8, 9)
    assert graph[0][1]["is_bank_connector"] is True


def test_hierarchical_support_rollout_switches_support_edges():
    observations = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    dataset = {
        "observations": observations,
        "actions": np.zeros((3, 2), dtype=np.float32),
        "next_observations": observations.copy(),
        "terminals": np.zeros(3, dtype=bool),
    }
    cluster_model = fit_state_clusters(
        observations,
        method="grid_xy",
        n_clusters=3,
        seed=0,
        state_dims=[0, 1],
        n_bins_x=3,
        n_bins_y=1,
    )
    graph_edges = pd.DataFrame({"edge_id": [5], "src": [1], "dst": [2], "median_h": [1.0], "cost": [1.0]})
    bank_edges = pd.DataFrame({"edge_id": [7], "src": [0], "dst": [1], "median_h": [1.0], "cost": [1.0]})
    graph_segments = {
        "edge_id": np.asarray([5], dtype=np.int64),
        "global_i": np.asarray([1], dtype=np.int64),
        "global_j": np.asarray([2], dtype=np.int64),
    }
    bank_segments = {
        "edge_id": np.asarray([7], dtype=np.int64),
        "global_i": np.asarray([0], dtype=np.int64),
        "global_j": np.asarray([1], dtype=np.int64),
    }

    class DummyActionSpace:
        shape = (2,)
        low = np.asarray([-1.0, -1.0], dtype=np.float32)
        high = np.asarray([1.0, 1.0], dtype=np.float32)

    class DummyEnv:
        action_space = DummyActionSpace()

        def __init__(self):
            self.obs = observations[0].copy()

        def reset(self, seed=None, options=None):
            del seed, options
            self.obs = observations[0].copy()
            return self.obs.copy(), {"goal": observations[2].copy()}

        def step(self, action):
            self.obs = self.obs + np.asarray(action, dtype=np.float32)
            success = bool(self.obs[0] >= 2.0)
            return self.obs.copy(), float(success), success, False, {"success": success}

    class StepPolicy:
        def __call__(self, obs, goal):
            delta = np.asarray(goal, dtype=np.float32) - np.asarray(obs, dtype=np.float32)
            return np.clip(delta, -1.0, 1.0).astype(np.float32)

    episodes, traces = run_hierarchical_support_episodes(
        DummyEnv(),
        StepPolicy(),
        dataset=dataset,
        cluster_model=cluster_model,
        graph_edges=graph_edges,
        graph_segments=graph_segments,
        bank_edges=bank_edges,
        bank_segments=bank_segments,
        dataset_name="dummy-v0",
        method="hierarchical_support",
        num_episodes=1,
        max_steps=5,
    )
    assert episodes.loc[0, "success"] == 1.0
    assert episodes.loc[0, "completed_edges"] >= 1
    assert traces[0]["steps"][0]["segment_source"] == "bank"


def test_policy_aware_subgoal_scoring_can_override_nearest_initiation():
    observations = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    actions = np.asarray([[10.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    segments = {
        "edge_id": np.asarray([4, 4], dtype=np.int64),
        "global_i": np.asarray([0, 2], dtype=np.int64),
        "global_j": np.asarray([1, 3], dtype=np.int64),
    }
    edge_attrs = {
        "segment_source": "graph",
        "segment_edge_id": 4,
        "policy_edge_id": 4,
    }

    class ZeroPolicy:
        def __call__(self, obs, goal):
            del obs, goal
            return np.asarray([0.0], dtype=np.float32)

    subgoal, reason, info = choose_edge_subgoal(
        observations,
        edge_attrs,
        segments,
        {},
        current_obs=observations[0],
        final_goal=observations[3],
        initiation_weight=1.0,
        downstream_weight=0.0,
        policy=ZeroPolicy(),
        actions=actions,
        policy_mse_weight=1.0,
        policy_mse_scale=1.0,
    )
    assert reason == "policy_aware_current_and_final_goal"
    assert np.allclose(subgoal, observations[3])
    assert info["selected_policy_action_mse"] == 0.0
    assert info["policy_mse_used"] == 1.0


def test_failure_penalty_changes_support_planning_route():
    edges = pd.DataFrame(
        {
            "edge_id": [0, 1, 2],
            "src": [0, 0, 2],
            "dst": [1, 2, 1],
            "median_h": [1.0, 2.0, 2.0],
            "cost": [1.0, 2.0, 2.0],
        }
    )
    base = build_support_planning_graph(edges)
    penalized = build_support_planning_graph(
        edges,
        failed_edge_counts={("graph", 0): 1},
        failure_penalty=10.0,
    )
    assert list(__import__("networkx").shortest_path(base, 0, 1, weight="cost")) == [0, 1]
    assert list(__import__("networkx").shortest_path(penalized, 0, 1, weight="cost")) == [0, 2, 1]


def test_runtime_cluster_model_cache_hits_second_load(tmp_path):
    dataset = {
        "observations": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    }
    cache_path = tmp_path / "cluster.pkl"
    _, hit1 = load_or_fit_runtime_cluster_model(
        dataset,
        cluster_method="grid_xy",
        n_clusters=3,
        seed=0,
        state_dims=[0, 1],
        cache_path=cache_path,
    )
    _, hit2 = load_or_fit_runtime_cluster_model(
        dataset,
        cluster_method="grid_xy",
        n_clusters=3,
        seed=0,
        state_dims=[0, 1],
        cache_path=cache_path,
    )
    assert hit1 is False
    assert hit2 is True
    assert cache_path.exists()


def test_edge_memory_extracts_attempts_and_summarizes_failures():
    traces = [
        {
            "episode_id": 0,
            "steps": [
                {
                    "cluster": 0,
                    "edge_src": 0,
                    "edge_dst": 1,
                    "edge_id": 10,
                    "segment_edge_id": 10,
                    "segment_source": "graph",
                    "subgoal_l2": 1.0,
                    "selected_policy_action_mse": 0.2,
                },
                {
                    "cluster": 1,
                    "edge_src": 0,
                    "edge_dst": 1,
                    "edge_id": 10,
                    "segment_edge_id": 10,
                    "segment_source": "graph",
                    "subgoal_l2": 0.1,
                    "selected_policy_action_mse": 0.4,
                },
                {
                    "cluster": 1,
                    "edge_src": 1,
                    "edge_dst": 2,
                    "edge_id": 11,
                    "segment_edge_id": 11,
                    "segment_source": "bank",
                    "subgoal_l2": 2.0,
                    "selected_policy_action_mse": 0.6,
                },
            ],
        }
    ]
    attempts = extract_edge_attempts_from_traces(traces)
    summary = summarize_edge_attempts(attempts).set_index(["segment_source", "segment_edge_id"])

    assert attempts.shape[0] == 2
    assert summary.loc[("graph", 10), "completed"] == 1
    assert summary.loc[("graph", 10), "timeouts"] == 0
    assert summary.loc[("bank", 11), "completed"] == 0
    assert summary.loc[("bank", 11), "timeouts"] == 1
    assert summary.loc[("bank", 11), "failure_excess"] == 1


def test_edge_memory_splits_restarted_same_edge_attempts():
    traces = [
        {
            "episode_id": 0,
            "steps": [
                {
                    "cluster": 0,
                    "edge_src": 0,
                    "edge_dst": 1,
                    "segment_edge_id": 5,
                    "segment_source": "bank",
                    "edge_step": 1,
                },
                {
                    "cluster": 0,
                    "edge_src": 0,
                    "edge_dst": 1,
                    "segment_edge_id": 5,
                    "segment_source": "bank",
                    "edge_step": 2,
                },
                {
                    "cluster": 0,
                    "edge_src": 0,
                    "edge_dst": 1,
                    "segment_edge_id": 5,
                    "segment_source": "bank",
                    "edge_step": 1,
                },
                {
                    "cluster": 1,
                    "edge_src": 0,
                    "edge_dst": 1,
                    "segment_edge_id": 5,
                    "segment_source": "bank",
                    "edge_step": 2,
                },
            ],
        }
    ]
    attempts = extract_edge_attempts_from_traces(traces)
    assert attempts.shape[0] == 2
    assert attempts["attempt_steps"].tolist() == [2, 2]
    assert attempts["completed"].tolist() == [0, 1]


def test_edge_memory_merge_produces_failed_edge_penalty_counts():
    existing = pd.DataFrame(
        {
            "segment_source": ["graph"],
            "segment_edge_id": [3],
            "edge_src": [0],
            "edge_dst": [1],
            "attempts": [2],
            "completed": [0],
            "timeouts": [2],
            "success_rate": [0.0],
            "failure_excess": [2],
            "mean_attempt_steps": [4.0],
            "mean_final_subgoal_l2": [1.0],
            "mean_selected_policy_action_mse": [0.5],
        }
    )
    new_summary = pd.DataFrame(
        {
            "segment_source": ["graph", "bank"],
            "segment_edge_id": [3, 7],
            "edge_src": [0, 9],
            "edge_dst": [1, 10],
            "attempts": [1, 1],
            "completed": [1, 0],
            "timeouts": [0, 1],
            "success_rate": [1.0, 0.0],
            "failure_excess": [0, 1],
            "mean_attempt_steps": [2.0, 3.0],
            "mean_final_subgoal_l2": [0.0, 2.0],
            "mean_selected_policy_action_mse": [0.1, 0.2],
        }
    )

    merged = merge_edge_memory(existing, new_summary)
    failed_counts = memory_failed_edge_counts(merged, mode="failure_excess")

    graph_row = merged[(merged["segment_source"] == "graph") & (merged["segment_edge_id"] == 3)].iloc[0]
    assert graph_row["attempts"] == 3
    assert graph_row["completed"] == 1
    assert graph_row["timeouts"] == 2
    assert graph_row["failure_excess"] == 1
    assert failed_counts[("graph", 3)] == 1
    assert failed_counts[("bank", 7)] == 1


def test_edge_outcome_scores_penalize_failed_edges_more_than_completed_edges():
    memory = pd.DataFrame(
        {
            "segment_source": ["graph", "graph"],
            "segment_edge_id": [0, 1],
            "edge_src": [0, 0],
            "edge_dst": [1, 2],
            "attempts": [2, 2],
            "completed": [2, 0],
            "timeouts": [0, 2],
            "success_rate": [1.0, 0.0],
            "failure_excess": [0, 2],
            "mean_attempt_steps": [2.0, 2.0],
            "mean_final_subgoal_l2": [0.1, 2.0],
            "mean_selected_policy_action_mse": [0.01, 0.10],
        }
    )
    scores = fit_edge_outcome_scores(memory, penalty_weight=3.0, uncertainty_weight=0.0)
    penalties = edge_outcome_penalty_map(scores)

    completed_penalty = penalties[("graph", 0)]
    failed_penalty = penalties[("graph", 1)]
    assert failed_penalty > completed_penalty
    assert scores.loc[scores["segment_edge_id"] == 1, "posterior_success_prob"].iloc[0] < 0.5


def test_edge_outcome_penalty_changes_support_planning_route():
    edges = pd.DataFrame(
        {
            "edge_id": [0, 1, 2],
            "src": [0, 0, 2],
            "dst": [1, 2, 1],
            "median_h": [1.0, 2.0, 2.0],
            "cost": [1.0, 2.0, 2.0],
        }
    )
    base = build_support_planning_graph(edges)
    outcome_penalized = build_support_planning_graph(
        edges,
        edge_risk_penalties={("graph", 0): 10.0},
    )
    assert list(__import__("networkx").shortest_path(base, 0, 1, weight="cost")) == [0, 1]
    assert list(__import__("networkx").shortest_path(outcome_penalized, 0, 1, weight="cost")) == [0, 2, 1]
