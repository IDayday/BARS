#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.external.gas_backbone import GASBackbone
from bars.gas_bars.bars_v3_planner import plan_bars_v3, summarize_plan_edges
from bars.gas_bars.bridge_graph import BRIDGE_EDGE_TYPES, RISKY_EDGE_TYPES, load_bridge_graph
from bars.gas_bars.planner import plan


def _csv_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _success(info: dict[str, Any]) -> bool:
    ep = info.get("episode", {}) if isinstance(info, dict) else {}
    return bool(ep.get("success", False) or info.get("success", False) or info.get("goal_achieved", False) or info.get("is_success", False))


def _compact(xs: Any, n: int = 80) -> str:
    arr = np.asarray(xs).reshape(-1).tolist() if xs is not None else []
    return "|".join(str(x) for x in arr[:n])


def _graph_for_variant(root: Path, variant: str):
    if variant in {"official_gas_shortest_G0", "gas_shortest"}:
        return load_bridge_graph(root / "G0.pkl")
    if variant.startswith("aggressive_shortest_"):
        gid = variant.rsplit("_", 1)[-1]
        return load_bridge_graph(root / f"{gid}.pkl")
    if variant == "oracle_bridge_shortest":
        return load_bridge_graph(root / "G_oracle.pkl")
    return load_bridge_graph(root / "G3.pkl")


def run_episode(
    backbone: GASBackbone,
    env: Any,
    graph_bundle: Any,
    bridge_scores: pd.DataFrame | None,
    boundary_scores: pd.DataFrame | None,
    variant: str,
    task_id: int,
    episode_id: int,
    seed: int,
    max_steps: int,
    max_plan_edges: int,
    p_bridge_min: float,
    bridge_risk_budget: float,
    max_risky_bridges: int,
) -> dict[str, Any]:
    start = time.time()
    env, obs, goal, _, done, _ = backbone.setup_task_env(env, backbone.env_name or "", task_id, seed + episode_id, render_goal=False)
    obs = np.asarray(obs)
    goal = np.asarray(goal)
    phi_goal = backbone.get_phi(goal)
    phi_obs = backbone.get_phi(obs)
    initial = float(np.linalg.norm(phi_goal - phi_obs))
    best = initial
    success = False
    total_reward = 0.0
    subgoals_attempted = 0
    subgoals_reached = 0
    subgoal_failures = 0
    replans = 0
    plan_lengths = []
    planner_latencies = []
    first_plan = None
    last_plan = None
    active = []
    active_idx = 0
    info: dict[str, Any] = {}
    graph = graph_bundle.to_dict()
    while not done and (subgoals_attempted + subgoals_reached + subgoal_failures + replans) < max_steps:
        phi_obs = backbone.get_phi(obs)
        cur_goal_dist = float(np.linalg.norm(phi_goal - phi_obs))
        best = min(best, cur_goal_dist)
        need_replan = not active or active_idx >= len(active) or subgoal_failures >= 3
        if need_replan:
            if variant in {"p_bridge_budget", "p_bridge_boundary"}:
                pr = plan_bars_v3(
                    graph,
                    phi_obs,
                    phi_goal,
                    variant=variant,
                    bridge_scores=bridge_scores,
                    boundary_scores=boundary_scores,
                    p_bridge_min=p_bridge_min,
                    bridge_risk_budget=bridge_risk_budget,
                    max_risky_bridges=max_risky_bridges,
                    max_edges=max_plan_edges,
                    virtual_nodes=True,
                    force_closest=True,
                )
            else:
                pr = plan(graph, phi_obs, phi_goal, variant="gas_shortest", max_edges=max_plan_edges, virtual_nodes=True, force_closest=True)
            replans += 1
            planner_latencies.append(pr.planner_latency_ms)
            plan_lengths.append(pr.path_len)
            if first_plan is None:
                first_plan = pr
            last_plan = pr
            if pr.no_path:
                break
            active = [np.asarray(x) for x in pr.subgoal_phis]
            active_idx = 0
            subgoal_failures = 0
        target = phi_goal if not active or active_idx >= len(active) else active[active_idx]
        subgoals_attempted += 1
        reached = False
        for _ in range(int(max(1, graph_bundle.way_steps))):
            action = backbone.sample_action(obs, target, final_goal=False)
            obs, reward, done, info = backbone.step_env(env, backbone.env_name or "", action)
            total_reward += float(reward)
            phi_obs = backbone.get_phi(obs)
            best = min(best, float(np.linalg.norm(phi_goal - phi_obs)))
            if _success(info):
                success = True
                done = True
                break
            if float(np.linalg.norm(phi_obs - target)) <= graph_bundle.way_steps:
                active_idx += 1
                subgoals_reached += 1
                reached = True
                break
            if done:
                break
        if success:
            break
        if not reached:
            subgoal_failures += 1
        if subgoals_attempted >= max_steps:
            break
    final_phi = backbone.get_phi(obs)
    final = float(np.linalg.norm(phi_goal - final_phi))
    first = first_plan or last_plan
    edge_summary = summarize_plan_edges(first, graph_bundle.edges) if first is not None else {}
    row = {
        "env": backbone.env_name,
        "seed": seed,
        "task_id": task_id,
        "episode_id": episode_id,
        "variant": variant,
        "success": int(success),
        "return": total_reward,
        "steps": int(subgoals_attempted * max(1, graph_bundle.way_steps)),
        "duration_sec": time.time() - start,
        "actual_goal_raw": _compact(goal),
        "actual_goal_phi": _compact(phi_goal),
        "initial_goal_dist_phi": initial,
        "final_goal_dist_phi": final,
        "best_goal_dist_phi": best,
        "goal_improvement_phi": initial - best,
        "fallback_used": 0,
        "no_path_count": int(first.no_path) if first is not None else 1,
        "budget_reject_count": int(first is not None and "budget" in first.reject_reason),
        "replans": replans,
        "first_plan_edges": first.path_len if first is not None else 0,
        "mean_plan_edges": float(np.mean(plan_lengths)) if plan_lengths else 0.0,
        "last_plan_edges": last_plan.path_len if last_plan is not None else 0,
        "first_plan_temporal_cost": first.temporal_cost if first is not None else 0.0,
        "first_plan_exec_risk": first.exec_risk if first is not None else 0.0,
        "first_plan_boundary_risk": first.boundary_risk if first is not None else 0.0,
        "first_plan_pred_success": first.predicted_success if first is not None else 0.0,
        "first_plan_reject_reason": first.reject_reason if first is not None else "missing_plan",
        "max_plan_edges": max_plan_edges,
        "subgoals_attempted": subgoals_attempted,
        "subgoals_reached": subgoals_reached,
        "subgoal_reach_rate": subgoals_reached / max(subgoals_attempted, 1),
        "planner_latency_mean_ms": float(np.mean(planner_latencies)) if planner_latencies else 0.0,
        "path_node_ids": _compact(first.node_ids if first is not None else []),
        "path_edge_ids": _compact(first.edge_ids if first is not None else []),
    }
    row.update(edge_summary)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--variant", default="p_bridge_budget")
    p.add_argument("--gas-artifact-root", default="artifacts/gas")
    p.add_argument("--artifact-root", default="artifacts/stage23")
    p.add_argument("--gas-repo-path", default="external_src/GAS")
    p.add_argument("--run-root", default="runs_stage23_integrated")
    p.add_argument("--gpu", default="cpu")
    p.add_argument("--prefer-pretrained", type=int, default=1)
    p.add_argument("--train-if-missing", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-plan-edges", type=int, default=20)
    p.add_argument("--p-bridge-min", type=float, default=0.6)
    p.add_argument("--bridge-risk-budget", type=float, default=1.0)
    p.add_argument("--max-risky-bridges", type=int, default=2)
    args = p.parse_args()
    bb = GASBackbone.load_or_train(args.env, args.seed, args.gas_artifact_root, args.gas_repo_path, args.gpu, prefer_pretrained=bool(args.prefer_pretrained), train_if_missing=bool(args.train_if_missing), quick=True)
    graph_root = Path(args.artifact_root) / args.env / f"seed{args.seed}" / "bridge_graphs"
    graph_bundle = _graph_for_variant(graph_root, args.variant)
    scores_path = Path(args.artifact_root) / args.env / f"seed{args.seed}" / "p_bridge" / "bridge_scores.csv"
    bridge_scores = pd.read_csv(scores_path) if scores_path.exists() else None
    boundary_path = Path(args.artifact_root) / args.env / f"seed{args.seed}" / "boundary_junction" / "bridge_boundary_junctions.csv"
    boundary_scores = pd.read_csv(boundary_path) if boundary_path.exists() else None
    env, _, _ = bb.load_env_and_dataset()
    task_ids = bb.get_task_ids(env)
    run_dir = Path(args.run_root) / args.env / f"seed{args.seed}" / args.variant / f"B{args.bridge_risk_budget:g}_K{args.max_risky_bridges}"
    eval_csv = run_dir / "eval.csv"
    for ep in range(args.episodes):
        row = run_episode(bb, env, graph_bundle, bridge_scores, boundary_scores, args.variant, task_ids[ep % len(task_ids)], ep, args.seed, args.max_steps, args.max_plan_edges, args.p_bridge_min, args.bridge_risk_budget, args.max_risky_bridges)
        _csv_append(eval_csv, row)
        print(json.dumps({k: row[k] for k in ["env", "variant", "episode_id", "success", "steps", "first_plan_edges"]}))
    df = pd.read_csv(eval_csv)
    summary = df.groupby(["env", "seed", "variant"], dropna=False).agg(episodes=("success", "count"), success=("success", "mean"), steps=("steps", "mean"), risky_bridge_count=("risky_bridge_count", "mean")).reset_index()
    summary.to_csv(run_dir / "grouped.csv", index=False)


if __name__ == "__main__":
    main()
