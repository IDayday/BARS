#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stage30_official_gas_common import (
    ARCHIVED_PRE_STAGE30_STATUS,
    configure_official_env,
    final_goal_threshold,
    gas_source_identity,
    gas_config_overrides,
    load_dataset_arrays,
    parse_csv_list,
    parse_seed_list,
    protocol_lock_row,
    recover_node_dataset_indices,
    scan_official_artifacts,
    trajectory_ids_from_dataset,
    write_csv,
)


def _import_official_gas(gas_repo: Path) -> dict[str, Any]:
    sys.path.insert(0, str(gas_repo.resolve()))
    import jax
    import ogbench
    from K_utils.keygraph_utils import KeyGraph
    from M_utils.agents import agents_dict
    from M_utils.agents.gas import get_config
    from M_utils.flax_utils import restore_agent
    from O_utils.datasets import Dataset, GCDataset
    from O_utils.env_utils import EpisodeMonitor
    from O_utils.evaluation import flatten, setup_task_env, supply_rng

    return {
        "jax": jax,
        "ogbench": ogbench,
        "KeyGraph": KeyGraph,
        "agents_dict": agents_dict,
        "get_config": get_config,
        "restore_agent": restore_agent,
        "Dataset": Dataset,
        "GCDataset": GCDataset,
        "EpisodeMonitor": EpisodeMonitor,
        "flatten": flatten,
        "setup_task_env": setup_task_env,
        "supply_rng": supply_rng,
    }


def _load_official_components(art, gas_repo: Path, seed: int, eval_on_cpu: int) -> dict[str, Any]:
    mods = _import_official_gas(gas_repo)
    config = mods["get_config"]()
    for key, value in gas_config_overrides(art.env_name).items():
        config[key] = value
    dataset_dir = os.environ.get("OGBENCH_DATASET_DIR", "/mnt/project/offlinerl_datasets/ogbench")
    raw_env, train_dataset, _ = mods["ogbench"].make_env_and_datasets(art.env_name, dataset_dir=dataset_dir, compact_dataset=False)
    env = mods["EpisodeMonitor"](raw_env)
    env.reset(seed=seed)
    train_gc_dataset = mods["GCDataset"](mods["Dataset"].create(**train_dataset), config)
    example_batch = train_gc_dataset.sample(1)
    agent = mods["agents_dict"][config["agent_name"]].create(seed, example_batch["observations"], example_batch["actions"], config)
    policy_restore_path = os.path.dirname(str(art.policy_path))
    policy_restore_epoch = os.path.basename(str(art.policy_path)).split("_")[-1].split(".")[0]
    agent = mods["restore_agent"](agent, policy_restore_path, policy_restore_epoch)
    key_graph = mods["KeyGraph"]()
    key_graph.load_keygraph(os.path.dirname(str(art.keygraph_path)), os.path.basename(str(art.keygraph_path)).split("_")[-1].split(".")[0])
    if eval_on_cpu:
        agent = mods["jax"].device_put(agent, device=mods["jax"].devices("cpu")[0])
    return {"mods": mods, "config": config, "env": env, "agent": agent, "key_graph": key_graph}


def _shortest_path_indices(key_graph: Any, task_id: int, source: np.ndarray, force_closest: bool) -> tuple[list[int] | None, str]:
    if task_id not in key_graph.task_paths_dict:
        return None, "missing_task_paths"
    nodes = key_graph.nodes
    shortest_paths = key_graph.task_paths_dict[task_id]
    shortest_paths_dist = key_graph.task_paths_dist_dict[task_id]
    sp_keys = list(shortest_paths.keys())
    if not sp_keys:
        return None, "empty_task_paths"
    start_distances = np.linalg.norm(nodes[sp_keys] - source, axis=1)
    valid_indices = np.where(start_distances <= key_graph.way_steps)[0]
    if len(valid_indices) == 0:
        if not force_closest:
            return None, "no_source_within_way_steps"
        valid_indices = [int(np.argmin(start_distances))]
    best_total_distance = float("inf")
    best_path: list[int] | None = None
    for idx in valid_indices:
        path_key = sp_keys[int(idx)]
        path_distance = shortest_paths_dist[path_key]
        total_distance = float(start_distances[int(idx)] + path_distance)
        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_path = [int(x) for x in shortest_paths[path_key]]
    return best_path, "ok" if best_path is not None else "no_path"


def _is_exact_node_match(match: dict[str, Any] | None) -> bool:
    if not match:
        return False
    if "exact_embedding_match" in match:
        try:
            return bool(int(match["exact_embedding_match"]))
        except Exception:
            return False
    try:
        return float(match.get("embedding_match_dist", float("inf"))) <= 1e-5
    except Exception:
        return False


def _edge_metadata(key_graph: Any, u: int, v: int, node_map: dict[int, dict[str, Any]], traj_ids: dict[int, int]) -> dict[str, Any]:
    data = key_graph.graph.get_edge_data(u, v, default={}) or {}
    te_score = data.get("te_score", data.get("te", ""))
    tdr_distance = data.get("tdr_distance", data.get("distance", data.get("weight", "")))
    row: dict[str, Any] = {
        "u": u,
        "v": v,
        "edge_id": f"{u}->{v}",
        "edge_weight": data.get("weight", ""),
        "edge_weight_available": int("weight" in data),
        "edge_metadata_keys": ",".join(sorted(str(k) for k in data.keys())),
        "te_score": te_score,
        "te_score_available": int(te_score != ""),
        "tdr_distance": tdr_distance,
        "tdr_distance_available": int(tdr_distance != ""),
        "is_task_goal_edge": int(u >= key_graph.base_node_cnt or v >= key_graph.base_node_cnt),
    }
    mu = node_map.get(u)
    mv = node_map.get(v)
    u_exact = _is_exact_node_match(mu)
    v_exact = _is_exact_node_match(mv)
    row.update(
        {
            "u_dataset_idx": mu.get("dataset_idx", "") if mu else "",
            "v_dataset_idx": mv.get("dataset_idx", "") if mv else "",
            "u_embedding_match_dist": mu.get("embedding_match_dist", "") if mu else "",
            "v_embedding_match_dist": mv.get("embedding_match_dist", "") if mv else "",
            "u_embedding_match_tolerance": mu.get("embedding_match_tolerance", "") if mu else "",
            "v_embedding_match_tolerance": mv.get("embedding_match_tolerance", "") if mv else "",
            "u_exact_embedding_match": int(u_exact),
            "v_exact_embedding_match": int(v_exact),
            "edge_dataset_mapping_exact": int(u_exact and v_exact),
        }
    )
    if mu and mv and traj_ids and u_exact and v_exact:
        iu = int(mu["dataset_idx"])
        iv = int(mv["dataset_idx"])
        tu = traj_ids.get(iu)
        tv = traj_ids.get(iv)
        same = int(tu == tv) if tu is not None and tv is not None else ""
        dt = iv - iu if same == 1 else ""
        row.update(
            {
                "u_dataset_idx": iu,
                "v_dataset_idx": iv,
                "u_traj_id": tu if tu is not None else "",
                "v_traj_id": tv if tv is not None else "",
                "same_trajectory": same,
                "same_trajectory_available": int(same != ""),
                "dt": dt,
                "dt_available": int(dt != ""),
                "cross_trajectory": int(same == 0) if same != "" else "",
                "cross_trajectory_available": int(same != ""),
                "edge_category": "cross_trajectory_keygraph_edge"
                if same == 0
                else ("same_trajectory_temporal_like_edge" if same == 1 and abs(int(dt)) <= int(key_graph.way_steps) else "same_trajectory_nonlocal_edge"),
            }
        )
    else:
        row.update(
            {
                "same_trajectory": "",
                "same_trajectory_available": 0,
                "dt": "",
                "dt_available": 0,
                "cross_trajectory": "",
                "cross_trajectory_available": 0,
                "edge_category": "official_keygraph_edge_metadata_only",
            }
        )
    row["trajectory_semantics_valid"] = int(
        str(row.get("edge_dataset_mapping_exact", "")) in {"1", "1.0"}
        and str(row.get("same_trajectory_available", "")) in {"1", "1.0"}
    )
    return row


def _env_step_with_flags(env: Any, env_name: str, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any], int, int]:
    if env_name in ["kitchen-partial-v0"]:
        next_observation, reward, done, info = env.step(action)
        return next_observation[:30], reward, done, info, int(done), 0
    next_observation, reward, terminated, truncated, info = env.step(action)
    return next_observation, reward, bool(terminated or truncated), info, int(terminated), int(truncated)


def _dist_stats(values: list[float], prefix: str) -> dict[str, Any]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    if not xs:
        return {f"{prefix}_count": 0, f"{prefix}_mean": "", f"{prefix}_min": "", f"{prefix}_p50": "", f"{prefix}_max": ""}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        f"{prefix}_count": len(xs),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p50": float(np.quantile(arr, 0.5)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _official_task_ids(env: Any, env_name: str) -> list[int]:
    if env_name in ["kitchen-partial-v0"]:
        return [1]
    raw = getattr(env, "unwrapped", env)
    task_infos = getattr(raw, "task_infos", getattr(env, "task_infos", []))
    return list(range(1, len(task_infos) + 1)) if task_infos else [1]


def _path_edges_text(path_indices: list[int]) -> str:
    return " ".join(f"{u}->{v}" for u, v in zip(path_indices[:-1], path_indices[1:]))


def _path_nodes_text(path_indices: list[int]) -> str:
    return " ".join(str(x) for x in path_indices)


def _run_episode(
    *,
    art,
    seed: int,
    task_id: int,
    episode_id: int,
    components: dict[str, Any],
    actor_fn: Any,
    node_map: dict[int, dict[str, Any]],
    traj_ids: dict[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mods = components["mods"]
    env = components["env"]
    key_graph = components["key_graph"]
    agent = components["agent"]
    config = components["config"]
    get_phi_fn = agent.get_phi
    eval_seed = seed + episode_id
    env, observation, goal, _, done, _ = mods["setup_task_env"](env, art.env_name, task_id, False, seed=eval_seed)
    epsilon = 1e-10
    phi_obs = np.asarray(get_phi_fn(observation))
    phi_goal = np.asarray(get_phi_fn(goal))
    initial_goal_dist_phi = float(np.linalg.norm(phi_obs - phi_goal))
    path_indices, path_status = _shortest_path_indices(key_graph, task_id, phi_obs, force_closest=True)
    if path_indices is None:
        return (
            {
                "stage": "stage30_official_gas_instrumentation",
                "evidence_class": "OFFICIAL_GAS_EPISODE_TRACE",
                "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
                "env_name": art.env_name,
                "seed": seed,
                "task_id": task_id,
                "episode_id": episode_id,
                "eval_seed": eval_seed,
                "success": 0,
                "no_path": 1,
                "no_path_reason": path_status,
                "steps": 0,
                "planned_path_nodes": "",
                "planned_path_edges": "",
                "planned_path_len": 0,
                "final_active_path_nodes": "",
                "final_active_path_edges": "",
                "final_active_path_len": 0,
                "active_path_trace_count": 0,
                "planned_edge_count": 0,
                "num_subgoals": 0,
                "subgoal_reached_count": 0,
                "subgoal_reach_rate": 0.0,
                "first_failed_subgoal_index": "",
                "first_failed_subgoal": "",
                "first_failed_edge_id": "",
                "first_failed_edge": "",
                "first_failed_edge_source": "no_official_graph_path",
                "first_failed_edge_reliable": 0,
                "timeout": 0,
                "stuck": 0,
                "divergence": 0,
                "initial_goal_dist_phi": initial_goal_dist_phi,
                "final_goal_dist_phi": "",
                "final_goal_threshold": final_goal_threshold(art.env_name),
                "way_steps": int(config["way_steps"]),
            },
            [],
        )
    shortest_path_nodes = key_graph.nodes[path_indices]
    initial_path_indices = list(path_indices)
    active_path_indices = list(path_indices)
    active_path_trace: list[list[int]] = [list(path_indices)]
    final_goal_on = False
    step = 0
    terminated = 0
    truncated = 0
    max_reached_path_index = 0
    cached_path_miss_count = 0
    path_update_count = 0
    last_active_cur_node_idx = 0
    support_dists: list[float] = []
    used_edges: Counter[tuple[int, int]] = Counter()
    for u, v in zip(path_indices[:-1], path_indices[1:]):
        used_edges[(int(u), int(v))] += 1
    start_time = time.time()
    while not done:
        phi_obs = np.asarray(get_phi_fn(observation))
        if final_goal_on:
            cur_obs_goal = phi_goal
        else:
            cached_indices, cached_status = _shortest_path_indices(key_graph, task_id, phi_obs, force_closest=False)
            if cached_indices is not None:
                if cached_indices != path_indices:
                    path_update_count += 1
                    active_path_trace.append(list(cached_indices))
                    for u, v in zip(cached_indices[:-1], cached_indices[1:]):
                        used_edges[(int(u), int(v))] += 1
                path_indices = cached_indices
                active_path_indices = list(cached_indices)
                shortest_path_nodes = key_graph.nodes[path_indices]
            elif cached_status != "ok":
                cached_path_miss_count += 1
            distances = np.linalg.norm(np.asarray(shortest_path_nodes) - phi_obs, axis=1)
            valid_indices = np.where(distances <= int(config["way_steps"]))[0]
            cur_node_idx = int(valid_indices[-1]) if len(valid_indices) > 0 else 0
            last_active_cur_node_idx = cur_node_idx
            max_reached_path_index = max(max_reached_path_index, cur_node_idx)
            if len(shortest_path_nodes) <= final_goal_threshold(art.env_name):
                final_goal_on = True
                cur_obs_goal = phi_goal
            else:
                cur_obs_goal = shortest_path_nodes[cur_node_idx]
                support_dists.append(float(distances[cur_node_idx]))
        skills = (cur_obs_goal - phi_obs) / (np.linalg.norm(cur_obs_goal - phi_obs) + epsilon)
        action = actor_fn(observations=observation, goals=skills, temperature=0.0)
        action = np.clip(np.asarray(action), -1, 1)
        next_observation, _, done, info, terminated, truncated = _env_step_with_flags(env, art.env_name, action)
        step += 1
        observation = next_observation
    flat = mods["flatten"](info)
    success = int(round(float(flat.get("episode.success", 0.0))))
    final_phi = np.asarray(get_phi_fn(observation))
    final_goal_dist_phi = float(np.linalg.norm(final_phi - phi_goal))
    progress_norm = (initial_goal_dist_phi - final_goal_dist_phi) / max(initial_goal_dist_phi, 1e-10)
    first_failed_subgoal = ""
    first_failed_edge = ""
    first_failed_subgoal_idx: int | str = ""
    first_failed_edge_source = ""
    first_failed_edge_reliable = 0
    if not success:
        if final_goal_on:
            first_failed_edge_source = "final_goal_phase_no_keygraph_edge"
        elif active_path_indices:
            first_failed_edge_source = "active_final_path_trace"
            first_failed_subgoal_idx = min(last_active_cur_node_idx + 1, max(len(active_path_indices) - 1, 0))
            first_failed_subgoal = active_path_indices[first_failed_subgoal_idx] if active_path_indices else ""
            if first_failed_subgoal_idx > 0 and first_failed_subgoal_idx < len(active_path_indices):
                first_failed_edge = f"{active_path_indices[first_failed_subgoal_idx - 1]}->{active_path_indices[first_failed_subgoal_idx]}"
                first_failed_edge_reliable = 1
        else:
            first_failed_edge_source = "active_path_unavailable"
    episode_row: dict[str, Any] = {
        "stage": "stage30_official_gas_instrumentation",
        "evidence_class": "OFFICIAL_GAS_EPISODE_TRACE",
        "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
        "env_name": art.env_name,
        "seed": seed,
        "task_id": task_id,
        "episode_id": episode_id,
        "eval_seed": eval_seed,
        "success": success,
        "return": flat.get("episode.return", ""),
        "normalized_return": flat.get("episode.normalized_return", ""),
        "steps": step,
        "duration_sec": time.time() - start_time,
        "planned_path_nodes": _path_nodes_text(initial_path_indices),
        "planned_path_edges": _path_edges_text(initial_path_indices),
        "planned_path_len": len(initial_path_indices),
        "final_active_path_nodes": _path_nodes_text(active_path_indices),
        "final_active_path_edges": _path_edges_text(active_path_indices),
        "final_active_path_len": len(active_path_indices),
        "active_path_trace_count": len(active_path_trace),
        "planned_edge_count": max(0, len(initial_path_indices) - 1),
        "num_subgoals": max(0, len(initial_path_indices) - final_goal_threshold(art.env_name)),
        "subgoal_reached_count": max_reached_path_index + 1,
        "subgoal_reach_rate": (max_reached_path_index + 1) / max(1, len(initial_path_indices)),
        "first_failed_subgoal_index": first_failed_subgoal_idx,
        "first_failed_subgoal": first_failed_subgoal,
        "first_failed_edge_id": first_failed_edge,
        "first_failed_edge": first_failed_edge,
        "first_failed_edge_source": first_failed_edge_source,
        "first_failed_edge_reliable": first_failed_edge_reliable,
        "no_path": 0,
        "no_path_reason": "",
        "cached_path_miss_count": cached_path_miss_count,
        "path_update_count": path_update_count,
        "timeout": int(truncated),
        "terminated": int(terminated),
        "stuck": int((not success) and progress_norm < 0.05),
        "divergence": int((not success) and final_goal_dist_phi > initial_goal_dist_phi + 1e-6),
        "initial_goal_dist_phi": initial_goal_dist_phi,
        "final_goal_dist_phi": final_goal_dist_phi,
        "progress_norm": progress_norm,
        "final_goal_threshold": final_goal_threshold(art.env_name),
        "way_steps": int(config["way_steps"]),
    }
    episode_row.update(_dist_stats(support_dists, "selected_subgoal_phi_dist"))
    edge_rows: list[dict[str, Any]] = []
    for (u, v), count in sorted(used_edges.items()):
        meta = _edge_metadata(key_graph, u, v, node_map, traj_ids)
        meta.update(
            {
                "stage": "stage30_official_gas_instrumentation",
                "evidence_class": "OFFICIAL_GAS_PATH_EDGE_TRACE",
                "env_name": art.env_name,
                "seed": seed,
                "task_id": task_id,
                "episode_id": episode_id,
                "success": success,
                "path_usage_count": count,
                "path_usage": "success_path" if success else "failure_path",
                "edge_id": f"{u}->{v}",
                "failure_association": int(first_failed_edge == f"{u}->{v}"),
            }
        )
        edge_rows.append(meta)
    return episode_row, edge_rows


def _write_report(out_dir: Path, episode_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]]) -> None:
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        by_env[str(row["env_name"])].append(row)
    lines = [
        "# Stage30 Official GAS Instrumentation Report",
        "",
        "Status: OFFICIAL_GAS_NON_INVASIVE_TRACE.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "The control loop mirrors official `evaluate_with_graph`; logs are diagnostic only.",
        "",
        "## Episode Summary",
        "",
        "| env_name | episodes | success_rate | no_path_rate | timeout_rate | stuck_rate | divergence_rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for env_name, rows in sorted(by_env.items()):
        n = len(rows)
        def rate(key: str) -> float:
            return sum(float(r.get(key, 0) or 0) for r in rows) / max(1, n)
        lines.append(
            f"| {env_name} | {n} | {rate('success'):.4f} | {rate('no_path'):.4f} | {rate('timeout'):.4f} | {rate('stuck'):.4f} | {rate('divergence'):.4f} |"
        )
    edge_categories = Counter(str(r.get("edge_category", "")) for r in edge_rows)
    lines.extend(["", "## Edge Categories Observed", ""])
    for category, count in sorted(edge_categories.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Files", ""])
    lines.append(f"- protocol lock: `{out_dir / 'protocol_lock.csv'}`")
    lines.append(f"- episode traces: `{out_dir / 'official_gas_episode_traces.csv'}`")
    lines.append(f"- path edge traces: `{out_dir / 'official_gas_path_edges.csv'}`")
    (out_dir / "instrumentation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage30 non-invasive official GAS instrumentation.")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--out-root", default="runs_stage30_official_gas/instrumentation")
    parser.add_argument("--envs", default="antmaze-medium-navigate-v0")
    parser.add_argument("--seeds", default="44")
    parser.add_argument("--task-ids", default="auto")
    parser.add_argument("--max-task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--eval-on-cpu", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--fallback-mode", default="none")
    parser.add_argument("--recover-dataset-indices", type=int, default=0)
    parser.add_argument("--node-map-batch-size", type=int, default=4096)
    parser.add_argument("--node-map-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    os.environ.update(configure_official_env(args.gpu))
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = scan_official_artifacts(Path(args.artifact_root), parse_csv_list(args.envs), parse_seed_list(args.seeds))
    task_ids_arg = [int(x) for x in parse_csv_list(args.task_ids)]
    gas_repo = Path(args.gas_repo_path)
    source_identity = gas_source_identity(gas_repo)
    command_line = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    all_episode_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    for art in artifacts:
        components = _load_official_components(art, gas_repo, art.seed, args.eval_on_cpu)
        key_graph = components["key_graph"]
        task_ids = task_ids_arg or _official_task_ids(components["env"], art.env_name)
        if args.max_task_id > 0:
            task_ids = [int(x) for x in task_ids if int(x) <= args.max_task_id]
        protocol_rows.append(
            protocol_lock_row(
                art,
                gas_repo,
                stage="stage30_official_gas_instrumentation",
                evidence_class="OFFICIAL_GAS_PROTOCOL_LOCK",
                wrapper_status="OFFICIAL_GAS_NON_INVASIVE_TRACE_ACTIONS_UNCHANGED",
                command_line=command_line,
                task_id=",".join(str(x) for x in task_ids),
                episode_count=args.episodes,
                subgoal_horizon=int(key_graph.way_steps),
                eval_on_cpu=args.eval_on_cpu,
                gpu=args.gpu,
                source_identity=source_identity,
                extra={
                    "recover_dataset_indices": args.recover_dataset_indices,
                    "node_map_tolerance": args.node_map_tolerance,
                    "fallback_mode": args.fallback_mode,
                    "max_task_id": args.max_task_id,
                    "official_task_id_source": "auto_env_task_infos" if not task_ids_arg else "explicit_cli",
                },
            )
        )
        if args.recover_dataset_indices:
            node_map = recover_node_dataset_indices(
                key_graph.nodes,
                art.dataset_embeddings_path,
                base_node_count=int(key_graph.base_node_cnt),
                tolerance=float(args.node_map_tolerance),
                batch_size=args.node_map_batch_size,
            )
            traj_ids = trajectory_ids_from_dataset(art.dataset_npz_path)
        else:
            node_map = {}
            traj_ids = {}
        for task_id in task_ids:
            actor_fn = components["mods"]["supply_rng"](
                components["agent"].sample_actions,
                rng=components["mods"]["jax"].random.PRNGKey(art.seed),
            )
            for episode_id in range(args.episodes):
                episode_row, edge_rows = _run_episode(
                    art=art,
                    seed=art.seed,
                    task_id=task_id,
                    episode_id=episode_id,
                    components=components,
                    actor_fn=actor_fn,
                    node_map=node_map,
                    traj_ids=traj_ids,
                )
                all_episode_rows.append(episode_row)
                all_edge_rows.extend(edge_rows)
    write_csv(out_dir / "protocol_lock.csv", protocol_rows)
    write_csv(out_dir / "official_gas_episode_traces.csv", all_episode_rows)
    write_csv(out_dir / "official_gas_path_edges.csv", all_edge_rows)
    _write_report(out_dir, all_episode_rows, all_edge_rows)
    print(out_dir / "instrumentation_report.md")


if __name__ == "__main__":
    main()
