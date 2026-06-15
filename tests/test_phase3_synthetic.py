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
