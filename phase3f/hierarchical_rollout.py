from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from phase1.clustering import assign_clusters, fit_state_clusters
from phase3f.natural_rollout import (
    _clip_action,
    _goal_from_info,
    _reset_env,
    _step_env,
    _success_value,
)
from phase3.edge_rollout import policy_action


def fit_runtime_cluster_model(
    dataset: dict[str, Any],
    *,
    cluster_method: str,
    n_clusters: int,
    seed: int = 0,
    state_dims: list[int] | None = None,
) -> dict[str, Any]:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    return fit_state_clusters(
        observations,
        method=cluster_method,
        n_clusters=int(n_clusters),
        seed=int(seed),
        state_dims=state_dims,
    )


def load_or_fit_runtime_cluster_model(
    dataset: dict[str, Any],
    *,
    cluster_method: str,
    n_clusters: int,
    seed: int = 0,
    state_dims: list[int] | None = None,
    cache_path: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Load a cached runtime cluster model or fit and optionally cache it."""

    if cache_path is not None:
        path = Path(cache_path)
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f), True
    model = fit_runtime_cluster_model(
        dataset,
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        seed=seed,
        state_dims=state_dims,
    )
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(model, f)
    return model, False


def _read_edges(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_segments(path: str | Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with np.load(p) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _row_cost(row: Any) -> float:
    for name in ("cost", "median_h", "mean_h", "max_h"):
        value = getattr(row, name, None)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return 1.0


def _row_horizon(row: Any, multiplier: float, max_horizon: int | None) -> int:
    base = float(getattr(row, "median_h", getattr(row, "mean_h", 1.0)))
    horizon = max(1, int(np.ceil(float(multiplier) * base)))
    if max_horizon is not None:
        horizon = min(horizon, int(max_horizon))
    return horizon


def _edge_identity(row: Any, source: str) -> tuple[str, int, int]:
    edge_id = int(getattr(row, "edge_id"))
    bank_edge_id = getattr(row, "bank_edge_id", np.nan)
    if source == "graph" and bool(getattr(row, "is_repair_edge", False)) and pd.notna(bank_edge_id):
        return "bank", int(bank_edge_id), edge_id
    return source, edge_id, edge_id


def build_support_planning_graph(
    graph_edges: pd.DataFrame,
    *,
    bank_edges: pd.DataFrame | None = None,
    start_cluster: int | None = None,
    goal_cluster: int | None = None,
    allow_bank_connectors: bool = True,
    include_all_bank_edges: bool = False,
    failed_edge_counts: dict[tuple[str, int], int] | None = None,
    failure_penalty: float = 0.0,
    edge_risk_penalties: dict[tuple[str, int], float] | None = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    failed_edge_counts = failed_edge_counts or {}
    edge_risk_penalties = edge_risk_penalties or {}

    def maybe_add(row: Any, source: str, connector: bool) -> None:
        src = int(getattr(row, "src"))
        dst = int(getattr(row, "dst"))
        base_cost = _row_cost(row)
        segment_source, segment_edge_id, policy_edge_id = _edge_identity(row, source)
        failure_count = int(failed_edge_counts.get((segment_source, int(segment_edge_id)), 0))
        failure_cost = float(failure_penalty) * failure_count
        risk_penalty = float(edge_risk_penalties.get((segment_source, int(segment_edge_id)), 0.0))
        cost = float(base_cost + failure_cost + risk_penalty)
        attrs = {
            "src": src,
            "dst": dst,
            "base_cost": float(base_cost),
            "cost": cost,
            "failure_count": failure_count,
            "failure_cost": float(failure_cost),
            "edge_outcome_penalty": float(risk_penalty),
            "edge_risk_penalty": float(risk_penalty),
            "median_h": float(getattr(row, "median_h", cost)),
            "max_h": float(getattr(row, "max_h", getattr(row, "median_h", cost))),
            "edge_id": int(getattr(row, "edge_id")),
            "policy_edge_id": int(policy_edge_id),
            "segment_source": segment_source,
            "segment_edge_id": int(segment_edge_id),
            "is_bank_connector": bool(connector),
        }
        existing = graph.get_edge_data(src, dst)
        if existing is None or cost < float(existing.get("cost", np.inf)):
            graph.add_edge(src, dst, **attrs)

    if graph_edges is not None and not graph_edges.empty:
        for row in graph_edges.itertuples(index=False):
            maybe_add(row, "graph", False)

    if (allow_bank_connectors or include_all_bank_edges) and bank_edges is not None and not bank_edges.empty:
        start = None if start_cluster is None else int(start_cluster)
        goal = None if goal_cluster is None else int(goal_cluster)
        for row in bank_edges.itertuples(index=False):
            src = int(getattr(row, "src"))
            dst = int(getattr(row, "dst"))
            if include_all_bank_edges or (start is not None and src == start) or (goal is not None and dst == goal):
                maybe_add(row, "bank", True)
    return graph


def _segments_for_edge(
    segments: dict[str, np.ndarray],
    edge_id: int,
) -> np.ndarray:
    edge_ids = np.asarray(segments.get("edge_id", []), dtype=np.int64)
    if edge_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(edge_ids == int(edge_id))


def _segment_store(source: str, graph_segments: dict[str, np.ndarray], bank_segments: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return bank_segments if source == "bank" else graph_segments


def _candidate_policy_mse(
    policy: Any,
    init_obs: np.ndarray,
    term_obs: np.ndarray,
    actions: np.ndarray,
    init_idx: np.ndarray,
    *,
    edge_id: int,
    device: str | None,
) -> np.ndarray:
    values = np.zeros(init_obs.shape[0], dtype=np.float64)
    for i in range(init_obs.shape[0]):
        pred = policy_action(
            policy,
            init_obs[i],
            term_obs[i],
            remaining_h=1.0,
            edge_id=edge_id,
            device=device,
        )
        target = np.asarray(actions[int(init_idx[i])], dtype=np.float32).reshape(-1)
        values[i] = float(np.mean((np.asarray(pred, dtype=np.float32).reshape(-1) - target) ** 2))
    return values


def choose_edge_subgoal(
    observations: np.ndarray,
    edge_attrs: dict[str, Any],
    graph_segments: dict[str, np.ndarray],
    bank_segments: dict[str, np.ndarray],
    *,
    current_obs: np.ndarray,
    final_goal: np.ndarray,
    next_edge_attrs: dict[str, Any] | None = None,
    max_candidates: int = 256,
    initiation_weight: float = 1.0,
    downstream_weight: float = 0.25,
    policy: Any | None = None,
    actions: np.ndarray | None = None,
    device: str | None = None,
    policy_mse_weight: float = 0.0,
    policy_mse_scale: float = 0.05,
) -> tuple[np.ndarray | None, str, dict[str, float]]:
    store = _segment_store(str(edge_attrs["segment_source"]), graph_segments, bank_segments)
    seg_idx = _segments_for_edge(store, int(edge_attrs["segment_edge_id"]))
    if seg_idx.size == 0:
        return None, "missing_edge_segments", {}
    rng = np.random.default_rng(int(edge_attrs["segment_edge_id"]))
    if seg_idx.size > int(max_candidates):
        seg_idx = rng.choice(seg_idx, size=int(max_candidates), replace=False)
    global_j = np.asarray(store.get("global_j", []), dtype=np.int64)
    global_i = np.asarray(store.get("global_i", []), dtype=np.int64)
    init_idx = global_i[seg_idx]
    term_idx = global_j[seg_idx]
    init_obs = observations[init_idx]
    term_obs = observations[term_idx]
    init_flat = init_obs.reshape(init_obs.shape[0], -1)
    current_flat = np.asarray(current_obs, dtype=np.float32).reshape(1, -1)
    init_dist = np.linalg.norm(init_flat - current_flat, axis=1)
    downstream_dist = np.zeros_like(init_dist)
    reason = "nearest_current_and_final_goal"

    if next_edge_attrs is not None:
        next_store = _segment_store(str(next_edge_attrs["segment_source"]), graph_segments, bank_segments)
        next_seg_idx = _segments_for_edge(next_store, int(next_edge_attrs["segment_edge_id"]))
        if next_seg_idx.size > 0:
            if next_seg_idx.size > int(max_candidates):
                next_seg_idx = rng.choice(next_seg_idx, size=int(max_candidates), replace=False)
            next_global_i = np.asarray(next_store.get("global_i", []), dtype=np.int64)
            next_init_obs = observations[next_global_i[next_seg_idx]]
            term_flat = term_obs.reshape(term_obs.shape[0], -1)
            next_init_flat = next_init_obs.reshape(next_init_obs.shape[0], -1)
            distances = np.linalg.norm(term_flat[:, None, :] - next_init_flat[None, :, :], axis=2)
            downstream_dist = np.min(distances, axis=1)
            reason = "policy_aware_current_and_next_initiation"
    if reason == "nearest_current_and_final_goal":
        goal = np.asarray(final_goal, dtype=np.float32).reshape(1, -1)
        downstream_dist = np.linalg.norm(term_obs.reshape(term_obs.shape[0], -1) - goal.reshape(1, -1), axis=1)
        reason = "policy_aware_current_and_final_goal"

    policy_mse = np.zeros_like(init_dist)
    policy_mse_used = 0.0
    if policy is not None and actions is not None and float(policy_mse_weight) > 0.0:
        policy_mse = _candidate_policy_mse(
            policy,
            init_obs,
            term_obs,
            np.asarray(actions, dtype=np.float32),
            init_idx,
            edge_id=int(edge_attrs["policy_edge_id"]),
            device=device,
        )
        policy_mse_used = 1.0

    score = (
        float(initiation_weight) * init_dist
        + float(downstream_weight) * downstream_dist
        + float(policy_mse_weight) * (policy_mse / max(1e-6, float(policy_mse_scale)))
    )
    chosen = int(np.argmin(score))
    info = {
        "selected_init_distance": float(init_dist[chosen]),
        "selected_downstream_distance": float(downstream_dist[chosen]),
        "selected_policy_action_mse": float(policy_mse[chosen]) if policy_mse_used else float("nan"),
        "mean_candidate_policy_action_mse": float(np.mean(policy_mse)) if policy_mse_used else float("nan"),
        "selected_subgoal_score": float(score[chosen]),
        "num_subgoal_candidates": float(term_obs.shape[0]),
        "policy_mse_used": policy_mse_used,
    }
    return np.asarray(term_obs[chosen], dtype=np.float32), reason, info


def plan_cluster_path(graph: nx.DiGraph, start_cluster: int, goal_cluster: int) -> tuple[list[int], str]:
    start = int(start_cluster)
    goal = int(goal_cluster)
    if start == goal:
        return [start], "already_at_goal_cluster"
    if start not in graph:
        return [], "start_cluster_not_in_graph"
    if goal not in graph:
        return [], "goal_cluster_not_in_graph"
    try:
        return [int(x) for x in nx.shortest_path(graph, start, goal, weight="cost")], "planned"
    except nx.NetworkXNoPath:
        return [], "no_support_path"


def run_hierarchical_support_episodes(
    env: Any,
    policy: Any,
    *,
    dataset: dict[str, Any],
    cluster_model: dict[str, Any],
    graph_edges: pd.DataFrame,
    graph_segments: dict[str, np.ndarray],
    bank_edges: pd.DataFrame | None,
    bank_segments: dict[str, np.ndarray],
    dataset_name: str,
    method: str,
    num_episodes: int,
    max_steps: int,
    task_ids: list[int] | None = None,
    seed: int = 0,
    device: str | None = None,
    allow_bank_connectors: bool = True,
    edge_horizon_multiplier: float = 2.0,
    max_edge_horizon: int | None = None,
    max_replans: int = 5,
    subgoal_max_candidates: int = 256,
    allow_full_bank_fallback: bool = False,
    failure_penalty: float = 0.0,
    initiation_weight: float = 1.0,
    downstream_weight: float = 0.25,
    policy_mse_weight: float = 0.0,
    policy_mse_scale: float = 0.05,
    prior_failed_edge_counts: dict[tuple[str, int], int] | None = None,
    edge_risk_penalties: dict[tuple[str, int], float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    episode_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for episode_id in range(int(num_episodes)):
        task_id = int(task_ids[episode_id % len(task_ids)]) if task_ids else None
        ep_seed = int(seed + episode_id)
        total_reward = 0.0
        success = 0.0
        num_steps = 0
        replans = 0
        full_bank_fallbacks = 0
        completed_edges = 0
        planned_edges = 0
        failed_edge_attempts = 0
        failed_edge_counts: dict[tuple[str, int], int] = dict(prior_failed_edge_counts or {})
        current_failed_edges: set[tuple[str, int]] = set()
        prior_penalized_edges = len(failed_edge_counts)
        failure_reason = ""
        initial_goal_l2 = float("nan")
        final_goal_l2 = float("nan")
        step_trace: list[dict[str, Any]] = []
        start_cluster = -1
        goal_cluster = -1
        try:
            obs, reset_info = _reset_env(env, seed=ep_seed, task_id=task_id)
            final_goal = _goal_from_info(reset_info)
            if final_goal is None:
                raise RuntimeError("missing_goal_observation")
            initial_goal_l2 = float(np.linalg.norm(obs.reshape(-1) - final_goal.reshape(-1)))
            final_goal_l2 = initial_goal_l2
            goal_cluster = int(assign_clusters(final_goal.reshape(1, -1), cluster_model)[0])
            active_path: list[int] = []
            active_edges: list[dict[str, Any]] = []
            edge_cursor = 0
            edge_step = 0
            current_subgoal: np.ndarray | None = None
            current_subgoal_reason = ""
            subgoal_info: dict[str, float] = {}
            while num_steps < int(max_steps):
                current_cluster = int(assign_clusters(obs.reshape(1, -1), cluster_model)[0])
                if start_cluster < 0:
                    start_cluster = int(current_cluster)
                if success >= 1.0 or current_cluster == goal_cluster:
                    failure_reason = ""
                    break
                need_plan = not active_edges or edge_cursor >= len(active_edges)
                if not need_plan and current_cluster == int(active_edges[edge_cursor]["dst"]):
                    completed_edges += 1
                    edge_cursor += 1
                    edge_step = 0
                    current_subgoal = None
                    subgoal_info = {}
                    need_plan = edge_cursor >= len(active_edges)
                if need_plan:
                    graph = build_support_planning_graph(
                        graph_edges,
                        bank_edges=bank_edges,
                        start_cluster=current_cluster,
                        goal_cluster=goal_cluster,
                        allow_bank_connectors=allow_bank_connectors,
                        failed_edge_counts=failed_edge_counts,
                        failure_penalty=failure_penalty,
                        edge_risk_penalties=edge_risk_penalties,
                    )
                    active_path, plan_status = plan_cluster_path(graph, current_cluster, goal_cluster)
                    if (not active_path or len(active_path) <= 1) and allow_full_bank_fallback:
                        fallback_graph = build_support_planning_graph(
                            graph_edges,
                            bank_edges=bank_edges,
                            start_cluster=current_cluster,
                            goal_cluster=goal_cluster,
                            allow_bank_connectors=True,
                            include_all_bank_edges=True,
                            failed_edge_counts=failed_edge_counts,
                            failure_penalty=failure_penalty,
                            edge_risk_penalties=edge_risk_penalties,
                        )
                        fallback_path, fallback_status = plan_cluster_path(fallback_graph, current_cluster, goal_cluster)
                        if fallback_path and len(fallback_path) > 1:
                            graph = fallback_graph
                            active_path = fallback_path
                            plan_status = f"full_bank_fallback:{fallback_status}"
                            full_bank_fallbacks += 1
                    if not active_path or len(active_path) <= 1:
                        failure_reason = plan_status
                        break
                    active_edges = [dict(graph[active_path[i]][active_path[i + 1]]) for i in range(len(active_path) - 1)]
                    planned_edges += len(active_edges)
                    edge_cursor = 0
                    edge_step = 0
                    current_subgoal = None
                    subgoal_info = {}
                    replans += 1
                    if replans > int(max_replans):
                        failure_reason = "max_replans_exceeded"
                        break
                edge_attrs = active_edges[edge_cursor]
                if current_subgoal is None:
                    next_edge = active_edges[edge_cursor + 1] if edge_cursor + 1 < len(active_edges) else None
                    current_subgoal, current_subgoal_reason, subgoal_info = choose_edge_subgoal(
                        observations,
                        edge_attrs,
                        graph_segments,
                        bank_segments,
                        current_obs=obs,
                        final_goal=final_goal,
                        next_edge_attrs=next_edge,
                        max_candidates=subgoal_max_candidates,
                        initiation_weight=initiation_weight,
                        downstream_weight=downstream_weight,
                        policy=policy,
                        actions=actions,
                        device=device,
                        policy_mse_weight=policy_mse_weight,
                        policy_mse_scale=policy_mse_scale,
                    )
                    if current_subgoal is None:
                        failure_reason = current_subgoal_reason
                        break
                horizon = _row_horizon(pd.Series(edge_attrs), edge_horizon_multiplier, max_edge_horizon)
                remaining = max(1, horizon - edge_step)
                action = policy_action(
                    policy,
                    obs,
                    current_subgoal,
                    remaining_h=remaining,
                    edge_id=int(edge_attrs["policy_edge_id"]),
                    device=device,
                )
                action = _clip_action(env, action)
                subgoal_l2 = float(np.linalg.norm(obs.reshape(-1) - current_subgoal.reshape(-1)))
                obs, reward, terminated, truncated, info = _step_env(env, action)
                total_reward += float(reward)
                success = max(success, _success_value(info))
                final_goal_l2 = float(np.linalg.norm(obs.reshape(-1) - final_goal.reshape(-1)))
                num_steps += 1
                edge_step += 1
                new_cluster = int(assign_clusters(obs.reshape(1, -1), cluster_model)[0])
                step_trace.append(
                    {
                        "t": int(num_steps - 1),
                        "cluster": int(new_cluster),
                        "goal_cluster": int(goal_cluster),
                        "edge_src": int(edge_attrs["src"]),
                        "edge_dst": int(edge_attrs["dst"]),
                        "edge_id": int(edge_attrs["edge_id"]),
                        "segment_edge_id": int(edge_attrs["segment_edge_id"]),
                        "policy_edge_id": int(edge_attrs["policy_edge_id"]),
                        "segment_source": str(edge_attrs["segment_source"]),
                        "subgoal_reason": current_subgoal_reason,
                        "edge_step": int(edge_step),
                        "success": float(success),
                        "reward": float(reward),
                        "subgoal_l2": subgoal_l2,
                        "action_norm": float(np.linalg.norm(action.reshape(-1))),
                        "edge_failure_count": int(edge_attrs.get("failure_count", 0)),
                        "edge_failure_cost": float(edge_attrs.get("failure_cost", 0.0)),
                        "edge_outcome_penalty": float(edge_attrs.get("edge_outcome_penalty", 0.0)),
                        "edge_risk_penalty": float(edge_attrs.get("edge_risk_penalty", 0.0)),
                        "edge_planning_cost": float(edge_attrs.get("cost", 0.0)),
                        **subgoal_info,
                    }
                )
                if success >= 1.0 or terminated or truncated:
                    break
                if new_cluster == int(edge_attrs["dst"]):
                    completed_edges += 1
                    edge_cursor += 1
                    edge_step = 0
                    current_subgoal = None
                    subgoal_info = {}
                elif edge_step >= horizon:
                    fail_key = (str(edge_attrs["segment_source"]), int(edge_attrs["segment_edge_id"]))
                    failed_edge_counts[fail_key] = int(failed_edge_counts.get(fail_key, 0)) + 1
                    current_failed_edges.add(fail_key)
                    failed_edge_attempts += 1
                    active_edges = []
                    current_subgoal = None
                    subgoal_info = {}
                    edge_step = 0
            if success < 1.0 and not failure_reason:
                failure_reason = "max_steps_without_success"
        except Exception as exc:  # noqa: PERF203 - one failed episode should be diagnostic, not fatal.
            failure_reason = f"{type(exc).__name__}: {exc}"
        episode_rows.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "episode_id": int(episode_id),
                "seed": ep_seed,
                "task_id": task_id if task_id is not None else "",
                "num_steps": int(num_steps),
                "success": float(success),
                "total_reward": float(total_reward),
                "initial_goal_l2": float(initial_goal_l2),
                "final_goal_l2": float(final_goal_l2),
                "start_cluster": int(start_cluster),
                "goal_cluster": int(goal_cluster),
                "planned_edges": int(planned_edges),
                "completed_edges": int(completed_edges),
                "replans": int(replans),
                "full_bank_fallbacks": int(full_bank_fallbacks),
                "failed_edge_attempts": int(failed_edge_attempts),
                "unique_failed_edges": int(len(current_failed_edges)),
                "prior_penalized_edges": int(prior_penalized_edges),
                "failure_reason": failure_reason,
            }
        )
        traces.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "episode_id": int(episode_id),
                "seed": ep_seed,
                "task_id": task_id,
                "success": float(success),
                "failure_reason": failure_reason,
                "steps": step_trace,
            }
        )
    return pd.DataFrame(episode_rows), traces


def load_graph_artifacts(
    *,
    graph_edges_csv: str | Path,
    graph_segments_npz: str | Path | None = None,
    bank_edges_csv: str | Path | None = None,
    bank_segments_npz: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict[str, np.ndarray]]:
    graph_edges = _read_edges(graph_edges_csv)
    graph_segments = _read_segments(graph_segments_npz)
    bank_edges = _read_edges(bank_edges_csv)
    bank_segments = _read_segments(bank_segments_npz)
    return graph_edges, graph_segments, bank_edges, bank_segments
