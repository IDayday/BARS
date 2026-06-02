#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stage30_official_gas_common import (
    ARCHIVED_PRE_STAGE30_STATUS,
    configure_official_env,
    gas_source_identity,
    load_dataset_arrays,
    parse_csv_list,
    parse_seed_list,
    protocol_lock_row,
    recover_node_dataset_indices,
    scan_official_artifacts,
    trajectory_ids_from_dataset,
    write_csv,
)
from stage30_official_gas_instrument import _edge_metadata, _env_step_with_flags, _load_official_components


MEANINGFUL_CATEGORY_ORDER = [
    "frequently_used_success_path_edges",
    "first_failed_failure_path_edges",
    "high_use_edges",
    "long_hop_edges",
    "low_cost_edges",
    "high_cost_edges",
    "cross_trajectory_edges",
    "same_trajectory_temporal_like_edges",
    "high_te_edges",
    "low_te_edges",
]


def _read_path_edge_csv(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _edge_key(row: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(row["u"]), int(row["v"])
    except Exception:
        return None


def _finite_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _build_edge_categories(key_graph: Any, node_map: dict[int, dict[str, Any]], traj_ids: dict[int, int], path_edge_rows: list[dict[str, str]]) -> dict[str, list[tuple[int, int]]]:
    categories: dict[str, list[tuple[int, int]]] = defaultdict(list)
    te_edges: list[tuple[float, tuple[int, int]]] = []
    weighted_edges: list[tuple[float, tuple[int, int]]] = []
    for u, v, data in key_graph.graph.edges(data=True):
        u = int(u)
        v = int(v)
        meta = _edge_metadata(key_graph, u, v, node_map, traj_ids)
        if u >= key_graph.base_node_cnt:
            continue
        cat = str(meta.get("edge_category", ""))
        if cat == "same_trajectory_temporal_like_edge":
            categories["same_trajectory_temporal_like_edges"].append((u, v))
        if cat == "cross_trajectory_keygraph_edge":
            categories["cross_trajectory_edges"].append((u, v))
        te_f = _finite_float(meta.get("te_score", ""))
        if math.isfinite(te_f):
            te_edges.append((te_f, (u, v)))
        weight_f = _finite_float(meta.get("edge_weight", ""))
        if math.isfinite(weight_f):
            weighted_edges.append((weight_f, (u, v)))
            if weight_f > float(key_graph.way_steps):
                categories["long_hop_edges"].append((u, v))
    if te_edges:
        te_edges.sort()
        n = len(te_edges)
        categories["low_te_edges"].extend(edge for _, edge in te_edges[: max(1, n // 5)])
        categories["high_te_edges"].extend(edge for _, edge in te_edges[-max(1, n // 5) :])
    if weighted_edges:
        weighted_edges.sort()
        n = len(weighted_edges)
        categories["low_cost_edges"].extend(edge for _, edge in weighted_edges[: max(1, n // 5)])
        categories["high_cost_edges"].extend(edge for _, edge in weighted_edges[-max(1, n // 5) :])

    success_counts: Counter[tuple[int, int]] = Counter()
    failed_counts: Counter[tuple[int, int]] = Counter()
    all_counts: Counter[tuple[int, int]] = Counter()
    for row in path_edge_rows:
        edge = _edge_key(row)
        if edge is None:
            continue
        try:
            count = int(float(row.get("path_usage_count", 1) or 1))
        except Exception:
            count = 1
        all_counts[edge] += count
        if str(row.get("path_usage", "")) == "success_path" or str(row.get("success", "")) in {"1", "1.0"}:
            success_counts[edge] += count
        if str(row.get("failure_association", "")) in {"1", "1.0"}:
            failed_counts[edge] += count
    categories["frequently_used_success_path_edges"].extend(edge for edge, _ in success_counts.most_common())
    categories["first_failed_failure_path_edges"].extend(edge for edge, _ in failed_counts.most_common())
    categories["high_use_edges"].extend(edge for edge, _ in all_counts.most_common())
    deduped: dict[str, list[tuple[int, int]]] = {}
    for category, edges in categories.items():
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int]] = []
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                out.append(edge)
        deduped[category] = out
    return deduped


def _sample_edges(edges: list[tuple[int, int]], n: int, seed: int) -> list[tuple[int, int]]:
    if len(edges) <= n:
        return list(edges)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(edges), size=n, replace=False)
    return [edges[int(i)] for i in idx]


def _set_state_from_dataset(env: Any, dataset: Any, dataset_idx: int, seed: int) -> tuple[bool, str, np.ndarray | None]:
    if dataset is None or "observations" not in dataset.files:
        return False, "dataset_observations_unavailable", None
    observation = np.asarray(dataset["observations"][dataset_idx]).copy()
    try:
        env.reset(seed=seed)
    except Exception:
        pass
    raw = getattr(env, "unwrapped", env)
    setter = getattr(raw, "set_state", None)
    if setter is None and hasattr(raw, "env"):
        setter = getattr(raw.env, "set_state", None)
    if setter is None:
        return False, "set_state_unavailable", observation
    if "qpos" in dataset.files and "qvel" in dataset.files:
        qpos = np.asarray(dataset["qpos"][dataset_idx]).copy()
        qvel = np.asarray(dataset["qvel"][dataset_idx]).copy()
    else:
        model = getattr(raw, "model", None)
        nq = int(getattr(model, "nq", 0) or 0)
        nv = int(getattr(model, "nv", 0) or 0)
        if nq <= 0 or nv <= 0:
            return False, "qpos_qvel_unavailable", observation
        obs = np.asarray(observation)
        if len(obs) >= nq + nv:
            qpos = obs[:nq].copy()
            qvel = obs[nq : nq + nv].copy()
        else:
            qpos = np.zeros(nq, dtype=np.float64)
            qpos[2:] = obs[: max(0, nq - 2)]
            qvel = obs[max(0, nq - 2) : max(0, nq - 2) + nv].copy()
    try:
        setter(qpos, qvel)
        return True, "set_state", observation
    except Exception as exc:
        return False, f"set_state_error:{type(exc).__name__}", observation


def _execute_edge(
    *,
    art,
    category: str,
    edge: tuple[int, int],
    edge_idx: int,
    components: dict[str, Any],
    node_map: dict[int, dict[str, Any]],
    traj_ids: dict[int, int],
    dataset: Any,
    horizon: int,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    u, v = edge
    key_graph = components["key_graph"]
    agent = components["agent"]
    mods = components["mods"]
    env = components["env"]
    meta = _edge_metadata(key_graph, u, v, node_map, traj_ids)
    row: dict[str, Any] = {
        "stage": "stage30_official_gas_edge_probe",
        "evidence_class": "OFFICIAL_GAS_EDGE_EXECUTION_PROBE",
        "pre_stage30_results_status": ARCHIVED_PRE_STAGE30_STATUS,
        "env_name": art.env_name,
        "seed": art.seed,
        "category": category,
        "sample_index": edge_idx,
        **meta,
        "horizon": horizon,
        "threshold": threshold,
    }
    u_info = node_map.get(u)
    if not u_info:
        row.update({"valid_probe": 0, "invalid_reason": "UNAVAILABLE_NODE_STATE_MAPPING", "reach": "", "set_state_success": 0})
        return row
    dataset_idx = int(u_info["dataset_idx"])
    ok, reason, observation = _set_state_from_dataset(env, dataset, dataset_idx, seed + edge_idx)
    row.update({"set_state_success": int(ok), "set_state_reason": reason})
    if not ok or observation is None:
        row.update({"valid_probe": 0, "invalid_reason": reason, "reach": ""})
        return row
    get_phi_fn = agent.get_phi
    actor_fn = mods["supply_rng"](agent.sample_actions, rng=mods["jax"].random.PRNGKey(seed + 7919 * (edge_idx + 1)))
    target_phi = np.asarray(key_graph.nodes[v])
    phi_obs = np.asarray(get_phi_fn(observation))
    initial_dist = float(np.linalg.norm(phi_obs - target_phi))
    min_dist = initial_dist
    done = False
    truncated = 0
    steps = 0
    action_norms: list[float] = []
    for step in range(horizon):
        phi_obs = np.asarray(get_phi_fn(observation))
        cur_dist = float(np.linalg.norm(phi_obs - target_phi))
        min_dist = min(min_dist, cur_dist)
        if cur_dist <= threshold:
            break
        skill = (target_phi - phi_obs) / (np.linalg.norm(target_phi - phi_obs) + 1e-10)
        action = actor_fn(observations=observation, goals=skill, temperature=0.0)
        action = np.clip(np.asarray(action), -1, 1)
        action_norms.append(float(np.linalg.norm(action)))
        observation, _, done, _, _, truncated = _env_step_with_flags(env, art.env_name, action)
        steps = step + 1
        if done:
            break
    final_phi = np.asarray(get_phi_fn(observation))
    final_dist = float(np.linalg.norm(final_phi - target_phi))
    min_dist = min(min_dist, final_dist)
    reach = int(min_dist <= threshold)
    progress_norm = (initial_dist - final_dist) / max(initial_dist, 1e-10)
    row.update(
        {
            "valid_probe": 1,
            "invalid_reason": "",
            "initial_dist": initial_dist,
            "final_dist": final_dist,
            "min_dist": min_dist,
            "reach": reach,
            "progress_norm": progress_norm,
            "steps": steps,
            "action_norm_mean": float(np.mean(action_norms)) if action_norms else "",
            "action_norm_max": float(np.max(action_norms)) if action_norms else "",
            "timeout": int(steps >= horizon and not reach),
            "stuck": int((not reach) and progress_norm < 0.05),
            "divergence": int((not reach) and final_dist > initial_dist + 1e-6),
            "env_done": int(done),
            "env_truncated": int(truncated),
        }
    )
    return row


def _write_report(out_dir: Path, rows: list[dict[str, Any]], required: int) -> None:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", ""))].append(row)
    lines = [
        "# Stage30 Official GAS Edge Execution Probe Report",
        "",
        "Status: OFFICIAL_GAS_EDGE_EXECUTION_PROBE.",
        f"Pre-Stage30 BARS/Stage28/Stage29 evidence: {ARCHIVED_PRE_STAGE30_STATUS}.",
        "A category is promotion-grade only when valid sampled edges meet the requested count.",
        "",
        "| category | rows | valid | reach_rate | set_state_rate | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category in MEANINGFUL_CATEGORY_ORDER:
        part = by_category.get(category, [])
        valid = [r for r in part if str(r.get("valid_probe", "")) in {"1", "1.0"}]
        reach = sum(float(r.get("reach", 0) or 0) for r in valid) / max(1, len(valid))
        set_state = sum(float(r.get("set_state_success", 0) or 0) for r in part) / max(1, len(part))
        status = "PASS_SAMPLE_COUNT" if len(valid) >= required else "INSUFFICIENT_SAMPLE"
        lines.append(f"| {category} | {len(part)} | {len(valid)} | {reach:.4f} | {set_state:.4f} | {status} |")
    lines.extend(["", "## Files", "", f"- edge probe CSV: `{out_dir / 'official_gas_edge_probe.csv'}`"])
    lines.append(f"- protocol lock: `{out_dir / 'protocol_lock.csv'}`")
    (out_dir / "edge_probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage30 official GAS edge execution probe.")
    parser.add_argument("--artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--out-root", default="runs_stage30_official_gas/edge_probe")
    parser.add_argument("--envs", default="antmaze-medium-navigate-v0")
    parser.add_argument("--seeds", default="44")
    parser.add_argument("--path-edge-csv", default="")
    parser.add_argument("--edges-per-category", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--eval-on-cpu", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--node-map-batch-size", type=int, default=4096)
    parser.add_argument("--node-map-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    os.environ.update(configure_official_env(args.gpu))
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_edge_rows = _read_path_edge_csv(Path(args.path_edge_csv) if args.path_edge_csv else None)
    artifacts = scan_official_artifacts(Path(args.artifact_root), parse_csv_list(args.envs), parse_seed_list(args.seeds))
    gas_repo = Path(args.gas_repo_path)
    source_identity = gas_source_identity(gas_repo)
    command_line = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    all_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    for art in artifacts:
        components = _load_official_components(art, gas_repo, art.seed, args.eval_on_cpu)
        key_graph = components["key_graph"]
        protocol_rows.append(
            protocol_lock_row(
                art,
                gas_repo,
                stage="stage30_official_gas_edge_probe",
                evidence_class="OFFICIAL_GAS_PROTOCOL_LOCK",
                wrapper_status="OFFICIAL_GAS_EDGE_EXECUTION_PROBE_OFFICIAL_POLICY",
                command_line=command_line,
                task_id="edge_probe",
                episode_count="",
                subgoal_horizon=int(key_graph.way_steps),
                eval_on_cpu=args.eval_on_cpu,
                gpu=args.gpu,
                source_identity=source_identity,
                extra={
                    "edges_per_category": args.edges_per_category,
                    "probe_horizon_arg": args.horizon,
                    "probe_threshold_arg": args.threshold,
                    "node_map_tolerance": args.node_map_tolerance,
                    "path_edge_csv": args.path_edge_csv,
                },
            )
        )
        node_map = recover_node_dataset_indices(
            key_graph.nodes,
            art.dataset_embeddings_path,
            base_node_count=int(key_graph.base_node_cnt),
            tolerance=float(args.node_map_tolerance),
            batch_size=args.node_map_batch_size,
        )
        traj_ids = trajectory_ids_from_dataset(art.dataset_npz_path)
        dataset = load_dataset_arrays(art.dataset_npz_path)
        categories = _build_edge_categories(key_graph, node_map, traj_ids, path_edge_rows)
        horizon = int(args.horizon if args.horizon > 0 else key_graph.way_steps)
        threshold = float(args.threshold if args.threshold > 0 else key_graph.way_steps)
        for category in MEANINGFUL_CATEGORY_ORDER:
            sampled = _sample_edges(categories.get(category, []), args.edges_per_category, art.seed + len(category))
            for i, edge in enumerate(sampled):
                all_rows.append(
                    _execute_edge(
                        art=art,
                        category=category,
                        edge=edge,
                        edge_idx=i,
                        components=components,
                        node_map=node_map,
                        traj_ids=traj_ids,
                        dataset=dataset,
                        horizon=horizon,
                        threshold=threshold,
                        seed=art.seed,
                    )
                )
    write_csv(out_dir / "official_gas_edge_probe.csv", all_rows)
    write_csv(out_dir / "protocol_lock.csv", protocol_rows)
    _write_report(out_dir, all_rows, args.edges_per_category)
    print(out_dir / "edge_probe_report.md")


if __name__ == "__main__":
    main()
