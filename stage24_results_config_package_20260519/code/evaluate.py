from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from bars.external.gas_artifacts import resolve_gas_artifacts
from bars.external.gas_backbone import GASBackbone
from .boundary import build_boundary_scores, save_boundary_scores
from .graph_table import export_edges, export_nodes, load_gas_keygraph, save_edge_table
from .planner import PlanResult, plan


def _parse_task_ids(spec: str, task_ids: list[int]) -> list[int]:
    if spec.lower() == "all":
        return task_ids
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _load_df(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    pq = path.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    return None


def _csv_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _compact(values: Any, max_items: int = 80) -> str:
    if values is None:
        return ""
    if isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    if isinstance(values, list):
        arr = values[:max_items]
        return "|".join(str(x) for x in arr)
    return str(values)


def _episode_success(info: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    ep = info.get("episode", {}) if isinstance(info, dict) else {}
    success = bool(
        ep.get("success", False)
        or info.get("success", False)
        or info.get("goal_achieved", False)
        or info.get("is_success", False)
    )
    return success, ep


def _ensure_tables_and_scores(
    env_name: str,
    seed: int,
    keygraph_path: Path,
    stage22_artifact_root: Path,
    backbone: GASBackbone,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    key_graph = load_gas_keygraph(keygraph_path)
    out = stage22_artifact_root / env_name / f"seed{seed}"
    graph_dir = out / "graph"
    nodes_path = graph_dir / "gas_graph_nodes.csv"
    edges_path = graph_dir / "gas_graph_edges.csv"
    if nodes_path.exists() and edges_path.exists():
        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
    else:
        nodes = export_nodes(key_graph)
        edges = export_edges(key_graph)
        save_edge_table(nodes, edges, graph_dir)

    edge_scores = _load_df(out / "edge_scores.csv")
    boundary_scores = _load_df(out / "boundary_scores.csv")
    if boundary_scores is None and edge_scores is not None:
        try:
            artifacts = resolve_gas_artifacts(env_name, seed, backbone.artifact_root)
            dataset_phis = np.load(artifacts.dataset_embeddings) if artifacts.dataset_embeddings else None
            _, train_dataset, _ = backbone.load_env_and_dataset()
            terminals = np.asarray(train_dataset["terminals"]).astype(bool) if "terminals" in train_dataset else None
            boundary_scores, summary = build_boundary_scores(
                nodes,
                edges,
                dataset_phis=dataset_phis,
                terminals=terminals,
                fallback_psi=0.1,
            )
            save_boundary_scores(boundary_scores, summary, out)
        except Exception as exc:
            with open(out / "boundary_error.json", "w") as f:
                json.dump({"error": repr(exc)}, f, indent=2)
            boundary_scores = None
    return key_graph, nodes, edges, edge_scores, boundary_scores


def _should_trigger_progress_stall(
    no_path: bool,
    plan_result: Optional[PlanResult],
    best_dist: float,
    initial_dist: float,
    reached_subgoals: int,
    no_improve_replans: int,
    subgoal_failures: int,
    progress_frac: float,
    close_goal_phi_dist: float,
    stall_k: int,
) -> tuple[bool, str]:
    progress = (
        best_dist <= initial_dist * (1.0 - progress_frac)
        or reached_subgoals >= 1
        or best_dist <= close_goal_phi_dist
    )
    stall_reason = ""
    if no_path:
        stall_reason = "planner_no_path"
    elif plan_result is not None and plan_result.reject_reason == "budget_infeasible":
        stall_reason = "budget_infeasible"
    elif no_improve_replans >= stall_k:
        stall_reason = "no_improvement"
    elif subgoal_failures >= stall_k:
        stall_reason = "subgoal_failure"
    elif plan_result is not None and plan_result.path_len <= 1:
        stall_reason = "short_plan"
    if progress and stall_reason:
        return True, stall_reason
    return False, ""


def _should_trigger_progress_stall_v2(
    no_path: bool,
    plan_result: Optional[PlanResult],
    current_dist: float,
    best_dist: float,
    initial_dist: float,
    no_improve_replans: int,
    subgoal_failures: int,
    progress_frac: float,
    close_goal_phi_dist: float,
    stall_k: int,
) -> tuple[bool, str]:
    """Conservative direct-goal fallback.

    The first pilot showed that treating "one reached subgoal" as sufficient
    progress can send the actor directly to the final goal too early.  V2
    requires real goal-distance progress or already-close state before using
    the direct-goal recovery.
    """
    meaningful_progress = best_dist <= initial_dist * (1.0 - progress_frac)
    close_to_goal = min(best_dist, current_dist) <= close_goal_phi_dist
    progress_gate = meaningful_progress or close_to_goal

    stall_reason = ""
    if no_path:
        stall_reason = "planner_no_path"
    elif plan_result is not None and plan_result.reject_reason == "budget_infeasible":
        stall_reason = "budget_infeasible"
    elif no_improve_replans >= stall_k:
        stall_reason = "no_improvement"
    elif subgoal_failures >= stall_k:
        stall_reason = "subgoal_failure"
    elif close_to_goal and plan_result is not None and plan_result.path_len <= 1:
        stall_reason = "short_plan_close"

    if progress_gate and stall_reason:
        return True, stall_reason
    return False, ""


def _should_trigger_progress_stall_v3(
    no_path: bool,
    plan_result: Optional[PlanResult],
    steps_elapsed: int,
    current_dist: float,
    best_dist: float,
    initial_dist: float,
    reached_subgoals: int,
    no_improve_replans: int,
    subgoal_failures: int,
    progress_frac: float = 0.35,
    close_goal_phi_dist: float = 2.0,
    no_improve_k: int = 5,
    subgoal_failure_k: int = 3,
    min_steps_before_fallback: int = 100,
) -> tuple[bool, str]:
    if steps_elapsed < min_steps_before_fallback:
        return False, ""
    progress = (
        best_dist <= initial_dist * (1.0 - progress_frac)
        or reached_subgoals >= 1
        or current_dist <= close_goal_phi_dist
    )
    if not progress:
        return False, ""
    if no_path:
        return True, "no_path"
    if plan_result is not None and plan_result.reject_reason == "budget_infeasible":
        return True, "budget_reject"
    if current_dist <= close_goal_phi_dist and plan_result is not None and plan_result.path_len <= 1:
        return True, "short_plan_close"
    if no_improve_replans >= no_improve_k:
        return True, "no_improvement"
    if subgoal_failures >= subgoal_failure_k:
        return True, "repeated_subgoal_failure"
    return False, ""


def _local_drift_metrics(
    debug_rows: list[dict[str, Any]],
    window: int = 16,
    stall_frac: float = 0.03,
) -> dict[str, float | int]:
    target = np.asarray([float(r.get("target_dist_phi", np.nan)) for r in debug_rows], dtype=np.float64)
    target = target[np.isfinite(target)]
    if target.size < 2:
        return {"local_drift_score": 0.0, "progress_stall_count": 0, "oscillation_score": 0.0}
    w = int(max(2, min(window, target.size)))
    drift_scores = []
    stall_count = 0
    for start in range(0, target.size - w + 1):
        seg = target[start : start + w]
        denom = max(float(seg[0]), 1e-6)
        drift_scores.append(float((seg[-1] - seg[0]) / denom))
        progress = float((seg[0] - np.min(seg)) / denom)
        if progress < stall_frac:
            stall_count += 1
    deltas = np.diff(target)
    path_motion = float(np.sum(np.abs(deltas)))
    net_improvement = float(max(target[0] - target[-1], 0.0))
    oscillation = float(max(path_motion - net_improvement, 0.0) / max(path_motion, 1e-6))
    return {
        "local_drift_score": float(max(0.0, max(drift_scores) if drift_scores else 0.0)),
        "progress_stall_count": int(stall_count),
        "oscillation_score": oscillation,
    }


def _write_debug_trace(path: Path, rows: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            out = dict(extra)
            out.update(row)
            f.write(json.dumps(out, default=str) + "\n")


def run_episode(
    backbone: GASBackbone,
    env: Any,
    key_graph: Any,
    edge_scores: Optional[pd.DataFrame],
    boundary_scores: Optional[pd.DataFrame],
    env_name: str,
    seed: int,
    task_id: int,
    episode_id: int,
    variant: str,
    budget: float,
    lambda_exec: float,
    lambda_boundary: float,
    fallback_mode: str,
    recompute_paths_per_episode: bool,
    max_steps: int,
    max_plan_edges: int,
    subgoal_threshold: float,
    final_goal_threshold: float,
    progress_frac: float = 0.40,
    close_goal_phi_dist: float = 2.0,
    stall_k: int = 3,
    debug_trace_path: Optional[Path] = None,
    local_drift_window: int = 16,
    local_drift_threshold: float = 0.10,
    progress_stall_frac: float = 0.03,
) -> dict[str, Any]:
    start_time = time.time()
    eval_seed = seed + episode_id
    env, observation, goal, _, done, _ = backbone.setup_task_env(env, env_name, task_id, eval_seed, render_goal=False)
    observation = np.asarray(observation)
    goal = np.asarray(goal)
    phi_goal = np.asarray(backbone.get_phi(goal))
    phi_obs = np.asarray(backbone.get_phi(observation))
    initial_goal_dist = float(np.linalg.norm(phi_goal - phi_obs))
    best_goal_dist = initial_goal_dist
    last_best = best_goal_dist
    fallback_used = False
    fallback_trigger_reason = ""
    direct_goal_steps = 0
    no_path_count = 0
    budget_reject_count = 0
    replans = 0
    subgoals_attempted = 0
    subgoals_reached = 0
    subgoal_failures = 0
    no_improve_replans = 0
    planner_latencies: list[float] = []
    plan_lengths: list[int] = []
    total_reward = 0.0
    success = False
    info: dict[str, Any] = {}
    first_plan: Optional[PlanResult] = None
    last_plan: Optional[PlanResult] = None
    debug_rows = []
    base_graph = {"nodes": export_nodes(key_graph), "edges": export_edges(key_graph), "way_steps": getattr(key_graph, "way_steps", 8.0)}
    active_subgoals: list[np.ndarray] = []
    active_subgoal_idx = 0
    requested_variant = variant.lower()
    replan_on_local_drift = requested_variant in {
        "gas_shortest_replan_on_local_drift",
        "gas_reachability_budget_replan_on_local_drift",
    }
    adaptive_subgoal_horizon = requested_variant == "gas_shortest_adaptive_subgoal_horizon"
    drift_triggered_replans = 0
    last_local_drift_score = 0.0

    while not done and replans < max_steps and (len(debug_rows) < max_steps):
        phi_obs = np.asarray(backbone.get_phi(observation))
        cur_dist = float(np.linalg.norm(phi_goal - phi_obs))
        if cur_dist + 1e-6 < best_goal_dist:
            best_goal_dist = cur_dist
        if abs(last_best - best_goal_dist) < 1e-4:
            no_improve_replans += 1
        else:
            no_improve_replans = 0
            last_best = best_goal_dist

        use_direct = fallback_used or cur_dist <= final_goal_threshold
        plan_result: Optional[PlanResult] = None
        need_replan = (not active_subgoals) or active_subgoal_idx >= len(active_subgoals) or subgoal_failures >= stall_k
        if not use_direct and need_replan:
            plan_result = plan(
                base_graph,
                phi_obs,
                phi_goal,
                task_id=task_id,
                variant=variant,
                edge_scores=edge_scores,
                boundary_scores=boundary_scores,
                budget=budget,
                lambda_exec=lambda_exec,
                lambda_boundary=lambda_boundary,
                max_edges=max_plan_edges,
                virtual_nodes=True,
                force_closest=True,
            )
            replans += 1
            planner_latencies.append(plan_result.planner_latency_ms)
            plan_lengths.append(plan_result.path_len)
            if first_plan is None:
                first_plan = plan_result
            last_plan = plan_result
            if plan_result.no_path:
                no_path_count += 1
                if plan_result.reject_reason == "budget_infeasible":
                    budget_reject_count += 1
            trigger, reason = (False, "")
            if fallback_mode == "progress_stall":
                trigger, reason = _should_trigger_progress_stall(
                    plan_result.no_path,
                    plan_result,
                    best_goal_dist,
                    initial_goal_dist,
                    subgoals_reached,
                    no_improve_replans,
                    subgoal_failures,
                    progress_frac,
                    max(close_goal_phi_dist, final_goal_threshold),
                    stall_k,
                )
            elif fallback_mode == "progress_stall_v2":
                trigger, reason = _should_trigger_progress_stall_v2(
                    plan_result.no_path,
                    plan_result,
                    cur_dist,
                    best_goal_dist,
                    initial_goal_dist,
                    no_improve_replans,
                    subgoal_failures,
                    max(progress_frac, 0.55),
                    max(close_goal_phi_dist, final_goal_threshold),
                    max(stall_k, 5),
                )
            elif fallback_mode == "progress_stall_v3":
                trigger, reason = _should_trigger_progress_stall_v3(
                    plan_result.no_path,
                    plan_result,
                    len(debug_rows),
                    cur_dist,
                    best_goal_dist,
                    initial_goal_dist,
                    subgoals_reached,
                    no_improve_replans,
                    subgoal_failures,
                    progress_frac=0.35,
                    close_goal_phi_dist=max(close_goal_phi_dist, final_goal_threshold),
                    no_improve_k=max(stall_k, 5),
                    subgoal_failure_k=max(stall_k, 3),
                    min_steps_before_fallback=100,
                )
            elif fallback_mode == "no_path" and plan_result.no_path:
                trigger, reason = True, "no_path"
            if trigger:
                fallback_used = True
                fallback_trigger_reason = reason
                use_direct = True
            elif plan_result.no_path:
                break
            else:
                active_subgoals = [np.asarray(x) for x in plan_result.subgoal_phis]
                active_subgoal_idx = 0
                subgoal_failures = 0
        elif not use_direct:
            plan_result = last_plan

        if use_direct:
            target_phi = phi_goal
            chunk_base = max(1, int(getattr(key_graph, "way_steps", 8)))
        else:
            if not active_subgoals or active_subgoal_idx >= len(active_subgoals):
                target_phi = phi_goal
            else:
                target_phi = np.asarray(active_subgoals[active_subgoal_idx])
                subgoals_attempted += 1
            chunk_base = max(1, int(getattr(key_graph, "way_steps", 8)))
        if adaptive_subgoal_horizon and not use_direct and (subgoal_failures > 0 or no_improve_replans > 0):
            chunk = max(1, chunk_base // 2)
        else:
            chunk = chunk_base

        reached_this = False
        chunk_target_dists: list[float] = []
        for _ in range(chunk):
            action = backbone.sample_action(observation, target_phi, final_goal=use_direct)
            observation, reward, done, info = backbone.step_env(env, env_name, action)
            observation = np.asarray(observation)
            total_reward += float(reward)
            if use_direct:
                direct_goal_steps += 1
            phi_obs = np.asarray(backbone.get_phi(observation))
            cur_goal_dist = float(np.linalg.norm(phi_goal - phi_obs))
            if cur_goal_dist < best_goal_dist:
                best_goal_dist = cur_goal_dist
            target_dist = float(np.linalg.norm(np.asarray(target_phi) - phi_obs))
            chunk_target_dists.append(target_dist)
            ep_success, _ = _episode_success(info if isinstance(info, dict) else {})
            success = success or ep_success
            debug_rows.append(
                {
                    "step": len(debug_rows),
                    "goal_dist_phi": cur_goal_dist,
                    "target_dist_phi": target_dist,
                    "fallback": int(fallback_used),
                    "plan_edges": plan_result.path_len if plan_result is not None else 0,
                }
            )
            if success:
                done = True
                break
            if not use_direct and target_dist <= subgoal_threshold:
                subgoals_reached += 1
                active_subgoal_idx += 1
                reached_this = True
                break
            if done or len(debug_rows) >= max_steps:
                break
        if not use_direct and chunk_target_dists:
            start_target_dist = max(float(chunk_target_dists[0]), 1e-6)
            end_target_dist = float(chunk_target_dists[-1])
            best_target_dist = float(min(chunk_target_dists))
            chunk_drift_score = (end_target_dist - start_target_dist) / start_target_dist
            chunk_progress = (start_target_dist - best_target_dist) / start_target_dist
            last_local_drift_score = max(last_local_drift_score, chunk_drift_score)
            local_stall = chunk_progress < progress_stall_frac and len(chunk_target_dists) >= min(local_drift_window, chunk)
            should_refresh = not reached_this and (chunk_drift_score > local_drift_threshold or local_stall)
            if (replan_on_local_drift or adaptive_subgoal_horizon) and should_refresh:
                active_subgoals = []
                active_subgoal_idx = 0
                subgoal_failures = max(subgoal_failures, stall_k)
                drift_triggered_replans += 1
        if not use_direct and not reached_this:
            subgoal_failures += 1
        if len(debug_rows) >= max_steps:
            break

    success, ep_info = _episode_success(info if isinstance(info, dict) else {})
    final_phi = np.asarray(backbone.get_phi(observation))
    final_dist = float(np.linalg.norm(phi_goal - final_phi))
    first = first_plan or PlanResult(variant=variant, budget=budget)
    last = last_plan or first
    drift_metrics = _local_drift_metrics(debug_rows, window=local_drift_window, stall_frac=progress_stall_frac)
    if last_local_drift_score > drift_metrics["local_drift_score"]:
        drift_metrics["local_drift_score"] = float(last_local_drift_score)
    trace_path_str = str(debug_trace_path) if debug_trace_path is not None else ""
    if debug_trace_path is not None:
        _write_debug_trace(
            debug_trace_path,
            debug_rows,
            {
                "env": env_name,
                "seed": seed,
                "task_id": task_id,
                "episode_id": episode_id,
                "variant": variant,
                "fallback_mode": fallback_mode,
            },
        )
    row = {
        "env": env_name,
        "seed": seed,
        "task_id": task_id,
        "episode_id": episode_id,
        "variant": variant,
        "budget": budget,
        "lambda_exec": lambda_exec,
        "lambda_boundary": lambda_boundary,
        "success": int(success),
        "return": total_reward,
        "normalized_return": ep_info.get("normalized_return", ""),
        "steps": len(debug_rows),
        "duration_sec": time.time() - start_time,
        "actual_goal_raw": _compact(goal),
        "actual_goal_phi": _compact(phi_goal),
        "initial_goal_dist_phi": initial_goal_dist,
        "final_goal_dist_phi": final_dist,
        "best_goal_dist_phi": best_goal_dist,
        "goal_improvement_phi": initial_goal_dist - best_goal_dist,
        "fallback_used": int(fallback_used),
        "fallback_trigger_reason": fallback_trigger_reason,
        "direct_goal_steps": direct_goal_steps,
        "no_path_count": no_path_count,
        "budget_reject_count": budget_reject_count,
        "replans": replans,
        "first_plan_edges": first.path_len,
        "mean_plan_edges": float(np.mean(plan_lengths)) if plan_lengths else 0.0,
        "last_plan_edges": last.path_len,
        "first_plan_temporal_cost": first.temporal_cost,
        "first_plan_exec_risk": first.exec_risk,
        "first_plan_boundary_risk": first.boundary_risk,
        "first_plan_pred_success": first.predicted_success,
        "first_plan_reject_reason": first.reject_reason,
        "last_plan_reject_reason": last.reject_reason,
        "max_plan_edges": max_plan_edges,
        "subgoals_attempted": subgoals_attempted,
        "subgoals_reached": subgoals_reached,
        "subgoal_reach_rate": subgoals_reached / max(subgoals_attempted, 1),
        "planner_latency_mean_ms": float(np.mean(planner_latencies)) if planner_latencies else 0.0,
        "planner_latency_p95_ms": float(np.quantile(planner_latencies, 0.95)) if planner_latencies else 0.0,
        "edge_fail_position": "",
        "path_node_ids": _compact(first.node_ids),
        "path_edge_ids": _compact(first.edge_ids),
        "local_drift_score": drift_metrics["local_drift_score"],
        "progress_stall_count": drift_metrics["progress_stall_count"],
        "oscillation_score": drift_metrics["oscillation_score"],
        "drift_triggered_replans": drift_triggered_replans,
        "debug_trace_path": trace_path_str,
    }
    return row


def run_episode_official_control(
    backbone: GASBackbone,
    env: Any,
    key_graph: Any,
    env_name: str,
    seed: int,
    task_id: int,
    episode_id: int,
    variant: str,
    budget: float,
    lambda_exec: float,
    lambda_boundary: float,
    max_steps: int,
    subgoal_threshold: float,
    final_goal_threshold: float,
) -> dict[str, Any]:
    """Adapter logging around the official GAS per-step control protocol."""
    start_time = time.time()
    eval_seed = seed + episode_id
    env, observation, goal, _, done, _ = backbone.setup_task_env(env, env_name, task_id, eval_seed, render_goal=False)
    observation = np.asarray(observation)
    goal = np.asarray(goal)
    phi_goal = np.asarray(backbone.get_phi(goal))
    phi_obs = np.asarray(backbone.get_phi(observation))
    initial_goal_dist = float(np.linalg.norm(phi_goal - phi_obs))
    best_goal_dist = initial_goal_dist
    total_reward = 0.0
    success = False
    info: dict[str, Any] = {}
    no_path_count = 0
    path_lengths: list[int] = []
    shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs, force_closest=True)
    if shortest_path is None:
        no_path_count = 1
        shortest_path = np.asarray([phi_goal])
    else:
        shortest_path = np.asarray(shortest_path)
        path_lengths.append(int(len(shortest_path)))
    final_goal_on = False
    steps = 0
    subgoals_attempted = 0
    subgoals_reached = 0
    last_target_idx = -1

    while not done and steps < max_steps:
        phi_obs = np.asarray(backbone.get_phi(observation))
        cur_goal_dist = float(np.linalg.norm(phi_goal - phi_obs))
        if cur_goal_dist < best_goal_dist:
            best_goal_dist = cur_goal_dist
        if final_goal_on:
            target_phi = phi_goal
            target_idx = -1
        else:
            cached = key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
            if cached is not None:
                shortest_path = np.asarray(cached)
                path_lengths.append(int(len(shortest_path)))
            if len(shortest_path) <= final_goal_threshold:
                final_goal_on = True
                target_phi = phi_goal
                target_idx = -1
            else:
                distances = np.linalg.norm(np.asarray(shortest_path) - phi_obs[None, :], axis=1)
                valid_indices = np.where(distances <= subgoal_threshold)[0]
                target_idx = int(valid_indices[-1]) if len(valid_indices) else 0
                target_phi = np.asarray(shortest_path[target_idx])
                subgoals_attempted += 1
                if target_idx > last_target_idx:
                    subgoals_reached += max(0, target_idx - max(last_target_idx, 0))
                    last_target_idx = target_idx
        action = backbone.sample_action(observation, target_phi, final_goal=final_goal_on)
        observation, reward, done, info = backbone.step_env(env, env_name, action)
        observation = np.asarray(observation)
        total_reward += float(reward)
        steps += 1
        ep_success, _ = _episode_success(info if isinstance(info, dict) else {})
        success = success or ep_success
        if success:
            done = True
            break

    success, ep_info = _episode_success(info if isinstance(info, dict) else {})
    final_phi = np.asarray(backbone.get_phi(observation))
    final_dist = float(np.linalg.norm(phi_goal - final_phi))
    first_len = int(path_lengths[0]) if path_lengths else 0
    last_len = int(path_lengths[-1]) if path_lengths else first_len
    return {
        "env": env_name,
        "seed": seed,
        "task_id": task_id,
        "episode_id": episode_id,
        "variant": variant,
        "budget": budget,
        "lambda_exec": lambda_exec,
        "lambda_boundary": lambda_boundary,
        "success": int(success),
        "return": total_reward,
        "normalized_return": ep_info.get("normalized_return", ""),
        "steps": steps,
        "duration_sec": time.time() - start_time,
        "actual_goal_raw": _compact(goal),
        "actual_goal_phi": _compact(phi_goal),
        "initial_goal_dist_phi": initial_goal_dist,
        "final_goal_dist_phi": final_dist,
        "best_goal_dist_phi": best_goal_dist,
        "goal_improvement_phi": initial_goal_dist - best_goal_dist,
        "fallback_used": 0,
        "fallback_trigger_reason": "",
        "direct_goal_steps": 0,
        "no_path_count": no_path_count,
        "budget_reject_count": 0,
        "replans": steps,
        "first_plan_edges": first_len,
        "mean_plan_edges": float(np.mean(path_lengths)) if path_lengths else 0.0,
        "last_plan_edges": last_len,
        "first_plan_temporal_cost": "",
        "first_plan_exec_risk": "",
        "first_plan_boundary_risk": "",
        "first_plan_pred_success": "",
        "first_plan_reject_reason": "no_path" if no_path_count else "",
        "last_plan_reject_reason": "",
        "max_plan_edges": "",
        "subgoals_attempted": subgoals_attempted,
        "subgoals_reached": subgoals_reached,
        "subgoal_reach_rate": subgoals_reached / max(subgoals_attempted, 1),
        "planner_latency_mean_ms": 0.0,
        "planner_latency_p95_ms": 0.0,
        "edge_fail_position": "",
        "path_node_ids": "",
        "path_edge_ids": "",
        "local_drift_score": 0.0,
        "progress_stall_count": 0,
        "oscillation_score": 0.0,
        "drift_triggered_replans": 0,
        "debug_trace_path": "",
    }


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-ids", default="all")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--episodes-per-task", type=int, default=0)
    p.add_argument("--variant", default="gas_boundary_budget")
    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--lambda-exec", type=float, default=1.0)
    p.add_argument("--lambda-boundary", type=float, default=1.0)
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--artifact-root", default="artifacts/stage22")
    p.add_argument("--stage22-root", default="runs_stage22_eval")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--gpu", default="0")
    p.add_argument("--eval-on-cpu", type=int, default=0)
    p.add_argument("--fallback-mode", default="progress_stall_v2", choices=["none", "progress_stall", "progress_stall_v2", "progress_stall_v3", "no_path"])
    p.add_argument("--recompute-paths-per-episode", type=int, default=1)
    p.add_argument("--prefer-pretrained", type=int, default=1)
    p.add_argument("--train-if-missing", type=int, default=1)
    p.add_argument("--quick", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-plan-edges", type=int, default=20)
    p.add_argument("--subgoal-threshold", type=float, default=8.0)
    p.add_argument("--final-goal-threshold", type=float, default=2.0)
    p.add_argument("--control-mode", default="bars", choices=["bars", "official"])
    p.add_argument("--debug-jsonl", type=int, default=0)
    args = p.parse_args(argv)

    eval_gpu = "cpu" if args.eval_on_cpu else args.gpu
    backbone = GASBackbone.load_or_train(
        args.env,
        args.seed,
        args.gas_artifact_root,
        args.gas_repo_path,
        eval_gpu,
        prefer_pretrained=bool(args.prefer_pretrained),
        train_if_missing=bool(args.train_if_missing),
        quick=bool(args.quick),
    )
    artifacts = backbone.artifacts or resolve_gas_artifacts(args.env, args.seed, args.gas_artifact_root)
    if artifacts.keygraph is None:
        raise RuntimeError("Missing GAS keygraph")
    if artifacts.dataset_embeddings is None:
        try:
            backbone.export_dataset_embeddings(artifacts.features_dir / "dataset_embeddings.npy")
        except Exception:
            pass
    key_graph, _, _, edge_scores, boundary_scores = _ensure_tables_and_scores(
        args.env,
        args.seed,
        artifacts.keygraph,
        Path(args.artifact_root),
        backbone,
    )
    env, _, _ = backbone.load_env_and_dataset()
    task_ids = _parse_task_ids(args.task_ids, backbone.get_task_ids(env))
    run_dir = (
        Path(args.stage22_root)
        / args.env
        / f"seed{args.seed}"
        / args.variant
        / f"budget{args.budget:g}"
        / f"fallback_{args.fallback_mode}"
    )
    eval_csv = run_dir / "eval.csv"
    completed = set()
    if eval_csv.exists():
        try:
            old = pd.read_csv(eval_csv)
            completed = set(zip(old["task_id"].astype(int), old["episode_id"].astype(int)))
        except Exception:
            completed = set()
    if args.episodes_per_task:
        schedule = [(task_id, ep) for task_id in task_ids for ep in range(args.episodes)]
    else:
        schedule = [(task_ids[ep % len(task_ids)], ep) for ep in range(args.episodes)]
    for task_id, ep in schedule:
        if (task_id, ep) in completed:
            continue
        if args.control_mode == "official":
            official_key_graph = backbone.key_graph if backbone.key_graph is not None else key_graph
            row = run_episode_official_control(
                backbone,
                env,
                official_key_graph,
                args.env,
                args.seed,
                task_id,
                ep,
                args.variant,
                args.budget,
                args.lambda_exec,
                args.lambda_boundary,
                args.max_steps,
                args.subgoal_threshold,
                args.final_goal_threshold,
            )
        else:
            debug_trace_path = run_dir / "debug_traces" / f"task{task_id}_episode{ep}.jsonl" if args.debug_jsonl else None
            row = run_episode(
                backbone,
                env,
                key_graph,
                edge_scores,
                boundary_scores,
                args.env,
                args.seed,
                task_id,
                ep,
                args.variant,
                args.budget,
                args.lambda_exec,
                args.lambda_boundary,
                args.fallback_mode,
                bool(args.recompute_paths_per_episode),
                args.max_steps,
                args.max_plan_edges,
                args.subgoal_threshold,
                args.final_goal_threshold,
                debug_trace_path=debug_trace_path,
            )
        _csv_append(eval_csv, row)
        print(json.dumps({k: row[k] for k in ["env", "task_id", "episode_id", "variant", "budget", "success", "steps", "fallback_used"]}))
    from .diagnostics import summarize_eval

    summarize_eval(run_dir)


if __name__ == "__main__":
    main()
