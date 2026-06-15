from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from phase1.clustering import assign_clusters
from phase3.reset_utils import reset_env_to_state


def _as_edges(option_edges: str | pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(option_edges) if isinstance(option_edges, str) else option_edges.copy()


def _as_segments(edge_segments: str | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(edge_segments, str):
        with np.load(edge_segments) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    return {key: np.asarray(value) for key, value in edge_segments.items()}


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, bool]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, _, terminated, truncated, _ = out
        return np.asarray(obs, dtype=np.float32), bool(terminated or truncated)
    if isinstance(out, tuple) and len(out) == 4:
        obs, _, done, _ = out
        return np.asarray(obs, dtype=np.float32), bool(done)
    raise RuntimeError(f"Unsupported env.step return signature: {type(out)}")


def _clip_action(env: Any, action: np.ndarray) -> np.ndarray:
    space = getattr(env, "action_space", None)
    if space is None or not hasattr(space, "low") or not hasattr(space, "high"):
        return action
    return np.clip(action, np.asarray(space.low), np.asarray(space.high)).astype(np.float32)


def _dataset_state_ref(dataset: dict[str, Any], index: int) -> dict[str, Any]:
    ref: dict[str, Any] = {"observation": np.asarray(dataset["observations"])[int(index)]}
    for key in ("qpos", "qvel", "state", "states", "sim_state", "sim_states", "infos/qpos", "infos/qvel"):
        value = dataset.get(key)
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] > int(index):
            ref[key] = value[int(index)]
    return ref


def policy_action(
    policy: Any,
    obs: np.ndarray,
    goal: np.ndarray,
    remaining_h: float | None = None,
    edge_id: int | None = None,
    target_source_id: int | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    if hasattr(policy, "eval") and callable(policy.eval):
        policy.eval()
    if isinstance(policy, torch.nn.Module):
        dev = torch.device(device) if device is not None else next(policy.parameters()).device
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).reshape(1, -1)
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=dev).reshape(1, -1)
            rem_t = None if remaining_h is None else torch.as_tensor([float(remaining_h)], dtype=torch.float32, device=dev)
            edge_t = None if edge_id is None else torch.as_tensor([int(edge_id)], dtype=torch.long, device=dev)
            source_id = 0 if edge_id is None else 2
            if target_source_id is not None:
                source_id = int(target_source_id)
            source_t = torch.as_tensor([source_id], dtype=torch.long, device=dev)
            action = policy(obs_t, goal_t, rem_t, edge_t, source_t).detach().cpu().numpy()[0]
        return np.asarray(action, dtype=np.float32)
    action = policy(obs, goal)
    return np.asarray(action, dtype=np.float32)


def dst_cluster_success(final_obs: np.ndarray, dst: int, cluster_model: dict[str, Any]) -> bool:
    label = int(assign_clusters(np.asarray(final_obs, dtype=np.float32).reshape(1, -1), cluster_model)[0])
    return label == int(dst)


def _termination_nearest_success(
    final_obs: np.ndarray,
    terminations: np.ndarray,
    eps: float,
) -> tuple[bool, float]:
    if terminations.size == 0:
        return False, float("inf")
    distances = np.linalg.norm(terminations.reshape(terminations.shape[0], -1) - final_obs.reshape(1, -1), axis=1)
    min_distance = float(np.min(distances))
    return bool(min_distance <= float(eps)), min_distance


def _horizon_for_edge(row: Any, horizon_mode: str, fixed_H: int | None) -> int:
    if horizon_mode == "edge_median":
        return max(1, int(round(float(getattr(row, "median_h", 1)))))
    if horizon_mode == "edge_max":
        return max(1, int(round(float(getattr(row, "max_h", getattr(row, "median_h", 1))))))
    if horizon_mode == "fixed_H":
        if fixed_H is None:
            raise ValueError("fixed_H is required when horizon_mode='fixed_H'")
        return max(1, int(fixed_H))
    raise ValueError("horizon_mode must be edge_median, edge_max, or fixed_H")


def evaluate_edge_rollouts(
    env: Any,
    policy: Any,
    dataset: dict[str, Any],
    option_edges: str | pd.DataFrame,
    edge_segments: str | dict[str, np.ndarray],
    cluster_model: dict[str, Any],
    num_edges: int,
    starts_per_edge: int,
    horizon_mode: str = "edge_median",
    success_mode: str = "dst_cluster",
    fixed_H: int | None = None,
    seed: int = 0,
    termination_eps: float = 1.0,
    device: str | torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    edges = _as_edges(option_edges)
    segments = _as_segments(edge_segments)
    if edges.empty:
        empty = pd.DataFrame(
            columns=[
                "edge_id",
                "src",
                "dst",
                "num_trials",
                "success_rate",
                "mean_final_distance",
                "edge_bottleneck_score",
                "num_unique_starts",
                "num_unique_episodes",
                "median_h",
                "strict_compatibility_context",
            ]
        )
        return empty, _rollout_summary(empty)

    rng = np.random.default_rng(seed)
    edge_ids = edges["edge_id"].astype(int).to_numpy()
    take = min(int(num_edges), int(edge_ids.size)) if num_edges is not None else int(edge_ids.size)
    selected_edge_ids = rng.choice(edge_ids, size=take, replace=False) if take < edge_ids.size else edge_ids
    seg_edge_ids = np.asarray(segments.get("edge_id", []), dtype=np.int64)
    global_i = np.asarray(segments.get("global_i", []), dtype=np.int64)
    global_j = np.asarray(segments.get("global_j", []), dtype=np.int64)

    rows = []
    for edge_id in selected_edge_ids:
        edge_row = edges[edges["edge_id"].astype(int) == int(edge_id)].iloc[0]
        seg_idx = np.flatnonzero(seg_edge_ids == int(edge_id))
        if seg_idx.size == 0:
            continue
        n_trials = min(int(starts_per_edge), int(seg_idx.size))
        chosen = rng.choice(seg_idx, size=n_trials, replace=seg_idx.size < n_trials)
        successes = []
        distances = []
        horizon = _horizon_for_edge(edge_row, horizon_mode, fixed_H)
        termination_states = observations[global_j[seg_idx]]
        for idx in chosen:
            start_idx = int(global_i[idx])
            goal_idx = int(global_j[idx])
            start_obs = observations[start_idx]
            goal_obs = observations[goal_idx]
            reset_env_to_state(env, _dataset_state_ref(dataset, start_idx))
            obs = start_obs.copy()
            done = False
            for step in range(horizon):
                action = policy_action(
                    policy,
                    obs,
                    goal_obs,
                    remaining_h=max(1, horizon - step),
                    edge_id=int(edge_id),
                    device=device,
                )
                obs, done = _step_env(env, _clip_action(env, action))
                if done:
                    break
            final_distance = float(np.linalg.norm(obs.reshape(-1) - goal_obs.reshape(-1)))
            if success_mode == "dst_cluster":
                success = dst_cluster_success(obs, int(edge_row.dst), cluster_model)
            elif success_mode == "termination_nearest":
                success, final_distance = _termination_nearest_success(obs, termination_states, termination_eps)
            else:
                raise ValueError("success_mode must be dst_cluster or termination_nearest")
            successes.append(float(success))
            distances.append(final_distance)
        rows.append(
            {
                "edge_id": int(edge_id),
                "src": int(edge_row.src),
                "dst": int(edge_row.dst),
                "num_trials": int(len(successes)),
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "mean_final_distance": float(np.mean(distances)) if distances else float("inf"),
                "edge_bottleneck_score": float(getattr(edge_row, "edge_bottleneck_score", np.nan)),
                "num_unique_starts": int(getattr(edge_row, "num_unique_starts", 0)),
                "num_unique_episodes": int(getattr(edge_row, "num_unique_episodes", getattr(edge_row, "num_episodes", 0))),
                "median_h": float(getattr(edge_row, "median_h", np.nan)),
                "strict_compatibility_context": getattr(edge_row, "strict_compatibility_context", np.nan),
            }
        )
    metrics = pd.DataFrame(rows)
    return metrics, _rollout_summary(metrics)


def _rollout_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(
            [
                {
                    "overall_edge_success": 0.0,
                    "bottleneck_edge_success": 0.0,
                    "non_bottleneck_edge_success": 0.0,
                    "high_support_edge_success": 0.0,
                    "low_support_edge_success": 0.0,
                }
            ]
        )
    bottleneck_thr = float(metrics["edge_bottleneck_score"].median()) if "edge_bottleneck_score" in metrics else np.nan
    support_thr = float(metrics["num_unique_starts"].median()) if "num_unique_starts" in metrics else np.nan
    high_b = metrics[metrics["edge_bottleneck_score"] >= bottleneck_thr]
    low_b = metrics[metrics["edge_bottleneck_score"] < bottleneck_thr]
    high_s = metrics[metrics["num_unique_starts"] >= support_thr]
    low_s = metrics[metrics["num_unique_starts"] < support_thr]
    return pd.DataFrame(
        [
            {
                "overall_edge_success": float(metrics["success_rate"].mean()),
                "bottleneck_edge_success": float(high_b["success_rate"].mean()) if not high_b.empty else 0.0,
                "non_bottleneck_edge_success": float(low_b["success_rate"].mean()) if not low_b.empty else 0.0,
                "high_support_edge_success": float(high_s["success_rate"].mean()) if not high_s.empty else 0.0,
                "low_support_edge_success": float(low_s["success_rate"].mean()) if not low_s.empty else 0.0,
            }
        ]
    )
