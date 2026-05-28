from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .diagnostics import write_failure
from .io import ensure_dir, write_csv, write_json
from .keygraph_tmd import TMDKeyGraph
from .repr_provider import TMDRepresentationProvider
from .tmd_agent_adapter import TMDAgentAdapter, _add_tmd_paths

TMD_GRAPH_MODES = {
    "tmd_graph_tmd_actor",
    "tmd_graph_gas_policy",
    "tmd_exec_graph_gas_policy",
    "gas_graph_tmd_exec_rescue_policy",
}
GAS_MODES = {
    "tmd_graph_gas_policy",
    "gas_direct_goal",
    "gas_graph_policy",
    "gas_graph_tmd_cost_policy",
    "tmd_exec_graph_gas_policy",
    "gas_graph_tmd_exec_rescue_policy",
}
TMD_PROVIDER_MODES = {
    "gas_graph_tmd_cost_policy",
}


def _parse_tasks(spec: str, available: int | None = None) -> list[int]:
    if spec.lower() == "all":
        return list(range(1, (available or 1) + 1))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out or [1]


def _success(info: dict) -> float:
    ep = info.get("episode", {}) if isinstance(info, dict) else {}
    for src in (info, ep):
        for key in ("success", "goal_achieved", "is_success"):
            if key in src:
                return float(src[key])
    return 0.0


def _task_count(env) -> int | None:
    unwrapped = getattr(env, "unwrapped", env)
    infos = getattr(unwrapped, "task_infos", None)
    if infos is None:
        infos = getattr(env, "task_infos", None)
    return len(infos) if infos is not None else None


def _step_env(env, action: np.ndarray):
    out = env.step(np.asarray(action))
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        return next_obs, reward, bool(terminated or truncated), info
    next_obs, reward, done, info = out
    return next_obs, reward, bool(done), info


def _reset_env(env, env_name: str, task_id: int, seed: int, gas=None):
    if gas is not None:
        env, obs, goal, _, _, _ = gas.setup_task_env(env, env_name, task_id, seed, render_goal=False)
        return env, np.asarray(obs, dtype=np.float32), np.asarray(goal, dtype=np.float32)
    obs, info = env.reset(seed=seed, options={"task_id": task_id, "render_goal": False})
    goal = np.asarray(info.get("goal", obs), dtype=np.float32)
    return env, np.asarray(obs, dtype=np.float32), goal


def _load_tmd_graph(args):
    tmd = TMDAgentAdapter.restore(
        args.tmd_checkpoint,
        {"env_name": args.env, "seed": args.seed, "dataset_root": args.dataset_root, "tmd_repo": args.tmd_repo},
    )
    provider = TMDRepresentationProvider(tmd, batch_size=256)
    graph = TMDKeyGraph.load(args.keygraph_path)
    return tmd, provider, graph


def _load_gas(args):
    from bars.external.gas_artifacts import resolve_gas_artifacts
    from bars.external.gas_backbone import GASBackbone

    gas_seed = int(args.gas_seed if args.gas_seed is not None else args.seed)
    artifacts = None
    policy_path = Path(args.gas_policy_path) if args.gas_policy_path else None
    if policy_path is None:
        artifacts = resolve_gas_artifacts(args.env, gas_seed, args.gas_artifact_root)
        policy_path = artifacts.policy_checkpoint
    if policy_path is None:
        raise FileNotFoundError(f"No GAS policy checkpoint found for {args.env} seed {gas_seed}")
    gas = GASBackbone(
        env_name=args.env,
        seed=gas_seed,
        artifact_root=Path(args.gas_artifact_root),
        gas_repo_path=Path(args.gas_repo),
        gpu=args.gas_gpu,
        artifacts=artifacts,
    )
    gas.load_policy(policy_path)
    keygraph_path = Path(args.gas_keygraph_path) if args.gas_keygraph_path else None
    if keygraph_path is None and artifacts is not None:
        keygraph_path = artifacts.keygraph
    if args.mode in {"gas_graph_policy", "gas_graph_tmd_cost_policy", "gas_graph_tmd_exec_rescue_policy"}:
        if keygraph_path is None:
            raise FileNotFoundError(f"No GAS keygraph found for {args.env} seed {gas_seed}")
        gas.load_keygraph(keygraph_path)
        args.gas_keygraph_path = str(keygraph_path)
    args.gas_seed = gas_seed
    args.gas_policy_path = str(policy_path)
    return gas


def _select_tmd_subgoal(graph: TMDKeyGraph, provider, task_id: int, obs: np.ndarray):
    psi = provider.encode(np.asarray(obs, dtype=np.float32))
    path = graph.get_shortest_path(task_id, psi, provider, graph.edge_distance_threshold)
    if path["no_path"] or len(path["raw_obs"]) == 0:
        return path, -1, None, -1
    dists = provider.distance_embeddings(psi[None, :], path["embeds"]).reshape(-1)
    valid = np.flatnonzero(dists <= graph.edge_distance_threshold)
    choose = int(valid[-1]) if len(valid) else 0
    return path, int(path["node_ids"][choose]), np.asarray(path["raw_obs"][choose], dtype=np.float32), choose


def _connect_components_by_phi(exec_graph, node_phi: np.ndarray) -> None:
    import networkx as nx

    components = [list(comp) for comp in nx.strongly_connected_components(exec_graph.subgraph(range(len(node_phi))).copy())]
    if len(components) <= 1:
        return
    while len(components) > 1:
        main = components[0]
        others = [node for comp in components[1:] for node in comp]
        dists = np.linalg.norm(node_phi[np.asarray(main)][:, None, :] - node_phi[np.asarray(others)][None, :, :], axis=-1)
        i, j = np.unravel_index(int(np.argmin(dists)), dists.shape)
        u, v = int(main[i]), int(others[j])
        weight = float(dists[i, j])
        exec_graph.add_edge(u, v, weight=weight, edge_type="gas_phi_component")
        exec_graph.add_edge(v, u, weight=weight, edge_type="gas_phi_component")
        for comp in components[1:]:
            if v in comp:
                components[0].extend(comp)
                components.remove(comp)
                break


def _train_observations(gas, limit: int) -> np.ndarray:
    _, train_dataset, _ = gas.load_env_and_dataset()
    observations = np.asarray(train_dataset["observations"], dtype=np.float32)
    if int(limit) > 0:
        observations = observations[: int(limit)]
    if len(observations) == 0:
        raise RuntimeError("GAS train dataset has no observations for TMD cost lookup")
    return observations


def _nearest_dataset_observations(gas, node_phi: np.ndarray, observations: np.ndarray, batch_size: int) -> np.ndarray:
    best_dist = np.full((len(node_phi),), np.inf, dtype=np.float32)
    best_idx = np.zeros((len(node_phi),), dtype=np.int64)
    for st in range(0, len(observations), int(batch_size)):
        ed = min(st + int(batch_size), len(observations))
        obs_phi = np.asarray(gas.get_phi(observations[st:ed]), dtype=np.float32)
        d = np.linalg.norm(node_phi[:, None, :] - obs_phi[None, :, :], axis=-1)
        local_idx = np.argmin(d, axis=1)
        local_dist = d[np.arange(len(node_phi)), local_idx]
        update = local_dist < best_dist
        best_dist[update] = local_dist[update]
        best_idx[update] = st + local_idx[update]
    return observations[best_idx]


def _apply_tmd_cost_to_gas_graph(gas, provider, args, out: Path) -> None:
    if gas.key_graph is None:
        raise RuntimeError("TMD-cost GAS policy requires a loaded GAS key graph")
    key_graph = gas.key_graph
    graph = key_graph.graph
    nodes = np.asarray(key_graph.nodes, dtype=np.float32)
    base_count = int(getattr(key_graph, "base_node_cnt", len(nodes)) or len(nodes))
    base_phi = nodes[:base_count]
    observations = _train_observations(gas, int(args.tmd_cost_lookup_observations))
    node_obs = _nearest_dataset_observations(gas, base_phi, observations, int(args.tmd_cost_lookup_batch_size))
    node_psi = provider.encode(node_obs)
    edges = [(int(u), int(v)) for u, v in graph.edges() if int(u) < base_count and int(v) < base_count]
    if not edges:
        write_json(out / "gas_tmd_cost_stats.json", {"num_edges": 0})
        return
    src = np.asarray([u for u, _ in edges], dtype=np.int64)
    dst = np.asarray([v for _, v in edges], dtype=np.int64)
    tmd_dist = np.asarray(
        provider.paired_distance(node_psi[src], node_psi[dst], batch_size=int(args.tmd_cost_batch_size)),
        dtype=np.float32,
    )
    normalizer = float(args.tmd_cost_normalizer)
    if normalizer <= 0:
        normalizer = float(np.median(tmd_dist))
    normalizer = max(normalizer, 1e-6)
    way_steps = float(gas.config["way_steps"])
    old_weights = []
    new_weights = []
    for (u, v), dist in zip(edges, tmd_dist):
        old_weight = float(graph[u][v].get("weight", np.linalg.norm(nodes[u] - nodes[v])))
        penalty = float(args.tmd_cost_weight) * way_steps * (float(dist) / normalizer)
        new_weight = old_weight + penalty
        graph[u][v]["weight"] = new_weight
        graph[u][v]["tmd_cost"] = float(dist)
        graph[u][v]["tmd_cost_penalty"] = penalty
        old_weights.append(old_weight)
        new_weights.append(new_weight)
    write_json(
        out / "gas_tmd_cost_stats.json",
        {
            "mode": args.mode,
            "num_base_nodes": int(base_count),
            "num_lookup_observations": int(len(observations)),
            "num_edges": int(len(edges)),
            "tmd_cost_weight": float(args.tmd_cost_weight),
            "tmd_cost_normalizer": float(normalizer),
            "tmd_cost_q50": float(np.quantile(tmd_dist, 0.50)),
            "tmd_cost_q90": float(np.quantile(tmd_dist, 0.90)),
            "old_weight_mean": float(np.mean(old_weights)),
            "new_weight_mean": float(np.mean(new_weights)),
        },
    )


def _build_tmd_exec_planner(graph: TMDKeyGraph, gas, goal: np.ndarray, args):
    import networkx as nx

    if graph.keynodes is None:
        raise RuntimeError("TMD key graph does not contain keynodes")
    raw_obs = graph.keynodes.raw_observations()
    node_phi = np.asarray(gas.get_phi(raw_obs), dtype=np.float32)
    goal_phi = np.asarray(gas.get_phi(goal), dtype=np.float32)
    threshold = float(gas.config["way_steps"]) * float(args.gas_exec_threshold_scale)
    exec_graph = nx.DiGraph()
    for i in range(len(node_phi)):
        exec_graph.add_node(i)

    source = args.exec_edge_source
    if source in {"gas_phi", "tmd_or_gas"}:
        d = np.linalg.norm(node_phi[:, None, :] - node_phi[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        src, dst = np.where(d <= threshold)
        for i, j in zip(src, dst):
            exec_graph.add_edge(int(i), int(j), weight=float(d[i, j]), edge_type="gas_phi")
    if source in {"tmd", "tmd_or_gas"}:
        for u, v, data in graph.graph.edges(data=True):
            if not isinstance(u, int) or not isinstance(v, int):
                continue
            gas_dist = float(np.linalg.norm(node_phi[int(u)] - node_phi[int(v)]))
            if gas_dist <= threshold:
                weight = gas_dist + float(args.tmd_edge_weight) * float(data.get("weight", 0.0))
                exec_graph.add_edge(int(u), int(v), weight=weight, edge_type="tmd_exec")

    if args.exec_connect_components:
        _connect_components_by_phi(exec_graph, node_phi)

    target = "target"
    exec_graph.add_node(target)
    target_d = np.linalg.norm(node_phi - goal_phi[None, :], axis=1)
    target_threshold = max(threshold, float(np.min(target_d)) * float(args.exec_target_min_scale))
    for i, dist in enumerate(target_d):
        if float(dist) <= target_threshold:
            exec_graph.add_edge(int(i), target, weight=float(dist), edge_type="target_forward")
            exec_graph.add_edge(target, int(i), weight=float(dist), edge_type="target_backward")

    lengths, paths = nx.single_source_dijkstra(exec_graph.reverse(copy=False), source=target, weight="weight")
    paths_to_goal = {}
    lengths_to_goal = {}
    for node, path in paths.items():
        if isinstance(node, int):
            paths_to_goal[int(node)] = [x for x in reversed(path) if isinstance(x, int)]
            lengths_to_goal[int(node)] = float(lengths[node])
    return {
        "graph": exec_graph,
        "node_phi": node_phi,
        "raw_obs": raw_obs,
        "goal_phi": goal_phi,
        "threshold": threshold,
        "paths_to_goal": paths_to_goal,
        "lengths_to_goal": lengths_to_goal,
        "target_threshold": target_threshold,
    }


def _select_tmd_exec_subgoal(planner, phi_obs: np.ndarray, final_goal_threshold: int):
    node_phi = planner["node_phi"]
    paths_to_goal = planner["paths_to_goal"]
    if not paths_to_goal:
        return {"no_path": True, "node_ids": [], "raw_obs": []}, -1, None, -1, 0
    keys = np.asarray(list(paths_to_goal.keys()), dtype=np.int64)
    start_d = np.linalg.norm(node_phi[keys] - np.asarray(phi_obs, dtype=np.float32)[None, :], axis=1)
    valid = np.flatnonzero(start_d <= float(planner["threshold"]))
    if len(valid) == 0:
        valid = np.asarray([int(np.argmin(start_d))], dtype=np.int64)
    best = None
    for idx in valid:
        node = int(keys[int(idx)])
        score = float(start_d[int(idx)]) + float(planner["lengths_to_goal"].get(node, np.inf))
        if best is None or score < best[0]:
            best = (score, node)
    if best is None:
        return {"no_path": True, "node_ids": [], "raw_obs": []}, -1, None, -1, 0
    path_ids = [int(x) for x in paths_to_goal[int(best[1])]]
    if not path_ids:
        return {"no_path": True, "node_ids": [], "raw_obs": []}, -1, None, -1, 0
    path_phi = node_phi[np.asarray(path_ids, dtype=np.int64)]
    d = np.linalg.norm(path_phi - np.asarray(phi_obs, dtype=np.float32)[None, :], axis=1)
    reachable = np.flatnonzero(d <= float(planner["threshold"]))
    choose = int(reachable[-1]) if len(reachable) else 0
    if len(path_ids) <= int(final_goal_threshold):
        choose = len(path_ids) - 1
    selected = int(path_ids[choose])
    path = {"no_path": False, "node_ids": path_ids, "raw_obs": planner["raw_obs"][np.asarray(path_ids, dtype=np.int64)]}
    return path, selected, np.asarray(planner["raw_obs"][selected], dtype=np.float32), choose, len(path_ids)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate tmd-test graph modes.")
    p.add_argument(
        "--mode",
        choices=[
            "tmd_graph_gas_policy",
            "tmd_graph_tmd_actor",
            "tmd_exec_graph_gas_policy",
            "gas_graph_tmd_exec_rescue_policy",
            "gas_direct_goal",
            "gas_graph_policy",
            "gas_graph_tmd_cost_policy",
            "tmd_full_gas_low",
        ],
        required=True,
    )
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--tasks", default="1")
    p.add_argument("--keygraph-path", required=True)
    p.add_argument("--tmd-checkpoint", required=True)
    p.add_argument("--gas-policy-path")
    p.add_argument("--gas-tdr-path")
    p.add_argument("--gas-keygraph-path")
    p.add_argument("--gas-artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    p.add_argument("--gas-repo", default="external_src/GAS")
    p.add_argument("--gas-seed", type=int)
    p.add_argument("--gas-gpu", default="cpu")
    p.add_argument("--gas-final-goal-threshold", type=int, default=2)
    p.add_argument("--gas-exec-threshold-scale", type=float, default=1.0)
    p.add_argument("--exec-edge-source", choices=["gas_phi", "tmd", "tmd_or_gas"], default="gas_phi")
    p.add_argument("--exec-connect-components", action="store_true")
    p.add_argument("--exec-target-min-scale", type=float, default=1.2)
    p.add_argument("--tmd-edge-weight", type=float, default=0.0)
    p.add_argument("--hybrid-switch-steps", type=int, default=200)
    p.add_argument("--hybrid-min-path-len", type=int, default=15)
    p.add_argument("--hybrid-rescue-max-steps", type=int, default=0)
    p.add_argument("--hybrid-rescue-cooldown-steps", type=int, default=0)
    p.add_argument("--hybrid-rescue-return-min-improve", type=int, default=0)
    p.add_argument("--hybrid-max-rescue-bursts", type=int, default=0)
    p.add_argument("--final-tmd-actor-after-steps", type=int, default=0)
    p.add_argument("--final-tmd-actor-max-steps", type=int, default=0)
    p.add_argument("--tmd-cost-weight", type=float, default=1.0)
    p.add_argument("--tmd-cost-normalizer", type=float, default=0.0)
    p.add_argument("--tmd-cost-lookup-observations", type=int, default=50000)
    p.add_argument("--tmd-cost-lookup-batch-size", type=int, default=2048)
    p.add_argument("--tmd-cost-batch-size", type=int, default=256)
    p.add_argument("--rescue-low-level", choices=["gas", "tmd_actor"], default="gas")
    p.add_argument("--sticky-subgoal-steps", type=int, default=0)
    p.add_argument("--fallback", default="none")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-root", default="/mnt/project/offlinerl_datasets/ogbench")
    p.add_argument("--tmd-repo", default="external_src/tmd-release")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args(argv)
    out = ensure_dir(args.output_dir)
    write_json(out / "eval_args.json", vars(args))
    try:
        if args.fallback != "none":
            raise ValueError("fallback must be none for tmd-test smoke")
        if args.mode == "tmd_full_gas_low":
            raise NotImplementedError("tmd_full_gas_low is intentionally not enabled until full GAS low-level integration is specified")
        tmd = provider = graph = None
        gas = None
        if args.mode in TMD_GRAPH_MODES:
            tmd, provider, graph = _load_tmd_graph(args)
        elif args.mode in TMD_PROVIDER_MODES:
            tmd = TMDAgentAdapter.restore(
                args.tmd_checkpoint,
                {"env_name": args.env, "seed": args.seed, "dataset_root": args.dataset_root, "tmd_repo": args.tmd_repo},
            )
            provider = TMDRepresentationProvider(tmd, batch_size=256)
        if args.mode in GAS_MODES:
            gas = _load_gas(args)
            env, _, _ = gas.load_env_and_dataset()
        if args.mode == "gas_graph_tmd_cost_policy":
            assert gas is not None and provider is not None
            _apply_tmd_cost_to_gas_graph(gas, provider, args, out)
        write_json(out / "eval_args.json", vars(args))
        _add_tmd_paths(args.tmd_repo)
        if gas is None:
            import ogbench

            env = ogbench.make_env_and_datasets(args.env, dataset_dir=args.dataset_root, env_only=True)
        task_count = _task_count(env)
        rows = []
        debug_rows = []
        for task_id in _parse_tasks(args.tasks, task_count):
            for ep in range(args.episodes):
                reset_seed = int(args.seed * 1000 + ep)
                env, obs, goal = _reset_env(env, args.env, task_id, reset_seed, gas=gas)
                done = False
                steps = 0
                no_path_steps = 0
                subgoal_switch_count = 0
                final_goal_mode_steps = 0
                last_node = None
                active_subgoal = None
                active_node = -1
                active_index = -1
                active_path_len = 0
                active_age = 0
                info = {}
                initial_goal_dist = float(np.linalg.norm(np.asarray(obs) - goal))
                best_goal_dist = initial_goal_dist
                gas_phi_goal = gas.get_phi(goal) if args.mode in {"gas_direct_goal", "gas_graph_policy", "gas_graph_tmd_cost_policy", "tmd_exec_graph_gas_policy", "gas_graph_tmd_exec_rescue_policy"} else None
                gas_shortest_path = None
                gas_final_goal_on = False
                tmd_rescue_on = False
                tmd_rescue_activated = 0
                tmd_rescue_first_step = -1
                tmd_rescue_steps = 0
                tmd_rescue_elapsed = 0
                tmd_rescue_entry_path_len = 0
                tmd_rescue_bursts = 0
                tmd_rescue_returns = 0
                hybrid_cooldown_remaining = 0
                tmd_final_rescue_first_step = -1
                tmd_final_rescue_steps = 0
                tmd_exec_planner = None
                if args.mode in {"gas_graph_policy", "gas_graph_tmd_cost_policy", "gas_graph_tmd_exec_rescue_policy"}:
                    assert gas is not None and gas.key_graph is not None and gas_phi_goal is not None
                    phi_obs = gas.get_phi(obs)
                    gas.key_graph.precompute_shortest_paths_to_all_tasks({task_id: goal}, {task_id: gas_phi_goal})
                    gas_shortest_path = gas.key_graph.get_shortest_path(task_id=task_id, source=phi_obs, force_closest=True)
                if args.mode in {"tmd_exec_graph_gas_policy", "gas_graph_tmd_exec_rescue_policy"}:
                    assert gas is not None and graph is not None
                    tmd_exec_planner = _build_tmd_exec_planner(graph, gas, goal, args)
                while not done:
                    path_len = 0
                    selected_node = -2
                    subgoal_index = -1
                    final_goal_active = 0
                    tmd_rescue_active = 0
                    tmd_final_rescue_active = 0
                    if args.mode == "gas_graph_tmd_exec_rescue_policy":
                        assert gas is not None and gas_phi_goal is not None and gas_shortest_path is not None
                        assert tmd_exec_planner is not None
                        if tmd_rescue_on and (
                            int(args.hybrid_rescue_max_steps) > 0
                            or int(args.hybrid_rescue_return_min_improve) > 0
                        ):
                            phi_obs = gas.get_phi(obs)
                            cached_path = gas.key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
                            if cached_path is not None:
                                gas_shortest_path = cached_path
                            gas_path_len = len(gas_shortest_path)
                            rescue_exhausted = (
                                int(args.hybrid_rescue_max_steps) > 0
                                and tmd_rescue_elapsed >= int(args.hybrid_rescue_max_steps)
                            )
                            rescue_improved = (
                                int(args.hybrid_rescue_return_min_improve) > 0
                                and gas_path_len <= tmd_rescue_entry_path_len - int(args.hybrid_rescue_return_min_improve)
                            )
                            rescue_near_final = gas_path_len <= int(args.gas_final_goal_threshold)
                            if rescue_exhausted or rescue_improved or rescue_near_final:
                                tmd_rescue_on = False
                                tmd_rescue_returns += 1
                                hybrid_cooldown_remaining = int(args.hybrid_rescue_cooldown_steps)
                                active_subgoal = None
                                active_age = 0
                                path_len = gas_path_len
                                if rescue_near_final:
                                    gas_final_goal_on = True
                        if not tmd_rescue_on:
                            if hybrid_cooldown_remaining > 0:
                                hybrid_cooldown_remaining -= 1
                            phi_obs = gas.get_phi(obs)
                            cached_path = gas.key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
                            if cached_path is not None:
                                gas_shortest_path = cached_path
                            path_len = len(gas_shortest_path)
                            distances = np.linalg.norm(np.asarray(gas_shortest_path) - phi_obs, axis=1)
                            valid = np.flatnonzero(distances <= float(gas.config["way_steps"]))
                            selected_node = int(valid[-1]) if len(valid) else 0
                            if len(gas_shortest_path) <= int(args.gas_final_goal_threshold):
                                gas_final_goal_on = True
                            if (
                                steps >= int(args.hybrid_switch_steps)
                                and not gas_final_goal_on
                                and path_len >= int(args.hybrid_min_path_len)
                                and hybrid_cooldown_remaining <= 0
                                and (
                                    int(args.hybrid_max_rescue_bursts) <= 0
                                    or tmd_rescue_bursts < int(args.hybrid_max_rescue_bursts)
                                )
                            ):
                                tmd_rescue_on = True
                                tmd_rescue_activated = 1
                                tmd_rescue_first_step = steps + 1
                                tmd_rescue_elapsed = 0
                                tmd_rescue_entry_path_len = path_len
                                tmd_rescue_bursts += 1
                                active_subgoal = None
                                active_age = 0
                        if not tmd_rescue_on:
                            if gas_final_goal_on:
                                final_goal_active = 1
                                use_tmd_final_rescue = (
                                    int(args.final_tmd_actor_after_steps) > 0
                                    and final_goal_mode_steps >= int(args.final_tmd_actor_after_steps)
                                    and (
                                        int(args.final_tmd_actor_max_steps) <= 0
                                        or tmd_final_rescue_steps < int(args.final_tmd_actor_max_steps)
                                    )
                                )
                                final_goal_mode_steps += 1
                                if use_tmd_final_rescue:
                                    assert tmd is not None
                                    tmd_final_rescue_active = 1
                                    if tmd_final_rescue_first_step < 0:
                                        tmd_final_rescue_first_step = steps + 1
                                    tmd_final_rescue_steps += 1
                                    action = tmd.sample_actions(
                                        obs,
                                        goal,
                                        temperature=args.temperature,
                                        seed=args.seed * 100000 + task_id * 1000 + ep * 100 + steps,
                                    )
                                else:
                                    action = gas.sample_action(obs, gas_phi_goal)
                            else:
                                action = gas.sample_action(obs, np.asarray(gas_shortest_path)[selected_node])
                        else:
                            tmd_rescue_active = 1
                            tmd_rescue_steps += 1
                            tmd_rescue_elapsed += 1
                            use_active = (
                                int(args.sticky_subgoal_steps) > 0
                                and active_subgoal is not None
                                and active_age < int(args.sticky_subgoal_steps)
                            )
                            if use_active:
                                selected_node = active_node
                                subgoal = active_subgoal
                                subgoal_index = active_index
                                path_len = active_path_len
                                active_age += 1
                            else:
                                phi_obs = gas.get_phi(obs)
                                path, selected_node, subgoal, subgoal_index, path_len = _select_tmd_exec_subgoal(
                                    tmd_exec_planner,
                                    phi_obs,
                                    args.gas_final_goal_threshold,
                                )
                                active_subgoal = subgoal
                                active_node = selected_node
                                active_index = subgoal_index
                                active_path_len = path_len
                                active_age = 1
                            if subgoal is None:
                                no_path_steps += 1
                                active_subgoal = None
                                action = np.zeros(env.action_space.shape, dtype=np.float32)
                            elif path_len <= int(args.gas_final_goal_threshold):
                                final_goal_mode_steps += 1
                                final_goal_active = 1
                                if args.rescue_low_level == "tmd_actor":
                                    assert tmd is not None
                                    action = tmd.sample_actions(
                                        obs,
                                        goal,
                                        temperature=args.temperature,
                                        seed=args.seed * 100000 + task_id * 1000 + ep * 100 + steps,
                                    )
                                else:
                                    action = gas.sample_action(obs, gas_phi_goal)
                            else:
                                if args.rescue_low_level == "tmd_actor":
                                    assert tmd is not None
                                    action = tmd.sample_actions(
                                        obs,
                                        subgoal,
                                        temperature=args.temperature,
                                        seed=args.seed * 100000 + task_id * 1000 + ep * 100 + steps,
                                    )
                                else:
                                    gas_phi_subgoal = gas.get_phi(subgoal)
                                    action = gas.sample_action(obs, gas_phi_subgoal)
                        if last_node is not None and selected_node != last_node:
                            subgoal_switch_count += 1
                        last_node = selected_node
                    elif args.mode == "tmd_exec_graph_gas_policy":
                        assert gas is not None and gas_phi_goal is not None and tmd_exec_planner is not None
                        use_active = (
                            int(args.sticky_subgoal_steps) > 0
                            and active_subgoal is not None
                            and active_age < int(args.sticky_subgoal_steps)
                        )
                        if use_active:
                            selected_node = active_node
                            subgoal = active_subgoal
                            subgoal_index = active_index
                            path_len = active_path_len
                            active_age += 1
                        else:
                            phi_obs = gas.get_phi(obs)
                            path, selected_node, subgoal, subgoal_index, path_len = _select_tmd_exec_subgoal(
                                tmd_exec_planner,
                                phi_obs,
                                args.gas_final_goal_threshold,
                            )
                            active_subgoal = subgoal
                            active_node = selected_node
                            active_index = subgoal_index
                            active_path_len = path_len
                            active_age = 1
                        if subgoal is None:
                            no_path_steps += 1
                            active_subgoal = None
                            action = np.zeros(env.action_space.shape, dtype=np.float32)
                        elif path_len <= int(args.gas_final_goal_threshold):
                            final_goal_mode_steps += 1
                            final_goal_active = 1
                            action = gas.sample_action(obs, gas_phi_goal)
                        else:
                            gas_phi_subgoal = gas.get_phi(subgoal)
                            action = gas.sample_action(obs, gas_phi_subgoal)
                        if last_node is not None and selected_node != last_node:
                            subgoal_switch_count += 1
                        last_node = selected_node
                    elif args.mode in TMD_GRAPH_MODES:
                        assert graph is not None and provider is not None
                        use_active = (
                            int(args.sticky_subgoal_steps) > 0
                            and active_subgoal is not None
                            and active_age < int(args.sticky_subgoal_steps)
                        )
                        if use_active:
                            selected_node = active_node
                            subgoal = active_subgoal
                            subgoal_index = active_index
                            path_len = active_path_len
                            active_age += 1
                        else:
                            path, selected_node, subgoal, subgoal_index = _select_tmd_subgoal(graph, provider, task_id, obs)
                            path_len = 0 if path["no_path"] else len(path["node_ids"])
                            active_subgoal = subgoal
                            active_node = selected_node
                            active_index = subgoal_index
                            active_path_len = path_len
                            active_age = 1
                        if subgoal is None:
                            no_path_steps += 1
                            active_subgoal = None
                            action = np.zeros(env.action_space.shape, dtype=np.float32)
                        elif subgoal_index >= path_len - 1:
                            final_goal_mode_steps += 1
                            final_goal_active = 1
                            if args.mode == "tmd_graph_tmd_actor":
                                assert tmd is not None
                                action = tmd.sample_actions(obs, goal, temperature=args.temperature, seed=args.seed + ep + steps + 17)
                            else:
                                assert gas is not None
                                gas_phi_goal = gas.get_phi(goal)
                                action = gas.sample_action(obs, gas_phi_goal)
                        elif args.mode == "tmd_graph_tmd_actor":
                            assert tmd is not None
                            action = tmd.sample_actions(obs, subgoal, temperature=args.temperature, seed=args.seed + ep + steps + 17)
                        else:
                            assert gas is not None
                            gas_phi_subgoal = gas.get_phi(subgoal)
                            action = gas.sample_action(obs, gas_phi_subgoal)
                        if last_node is not None and selected_node != last_node:
                            subgoal_switch_count += 1
                        last_node = selected_node
                    elif args.mode == "gas_direct_goal":
                        assert gas is not None and gas_phi_goal is not None
                        action = gas.sample_action(obs, gas_phi_goal)
                    elif args.mode in {"gas_graph_policy", "gas_graph_tmd_cost_policy"}:
                        assert gas is not None and gas_phi_goal is not None and gas_shortest_path is not None
                        phi_obs = gas.get_phi(obs)
                        if not gas_final_goal_on:
                            cached_path = gas.key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
                            if cached_path is not None:
                                gas_shortest_path = cached_path
                            path_len = len(gas_shortest_path)
                            distances = np.linalg.norm(np.asarray(gas_shortest_path) - phi_obs, axis=1)
                            valid = np.flatnonzero(distances <= float(gas.config["way_steps"]))
                            selected_node = int(valid[-1]) if len(valid) else 0
                            if len(gas_shortest_path) <= int(args.gas_final_goal_threshold):
                                gas_final_goal_on = True
                        if gas_final_goal_on:
                            final_goal_mode_steps += 1
                            final_goal_active = 1
                            action = gas.sample_action(obs, gas_phi_goal)
                        else:
                            action = gas.sample_action(obs, np.asarray(gas_shortest_path)[selected_node])
                    else:
                        raise NotImplementedError(args.mode)
                    if gas is not None:
                        next_obs, reward, done, info = gas.step_env(env, args.env, np.asarray(action))
                    else:
                        next_obs, reward, done, info = _step_env(env, np.asarray(action))
                    steps += 1
                    goal_dist = float(np.linalg.norm(np.asarray(next_obs) - goal))
                    best_goal_dist = min(best_goal_dist, goal_dist)
                    debug_rows.append(
                        {
                            "episode": ep,
                            "step": steps,
                            "task_id": task_id,
                            "selected_node": selected_node,
                            "path_len": path_len,
                            "final_goal_mode": final_goal_active,
                            "tmd_rescue_active": tmd_rescue_active,
                            "tmd_final_rescue_active": tmd_final_rescue_active,
                            "tmd_rescue_first_step": tmd_rescue_first_step,
                            "goal_dist": goal_dist,
                            "mode": args.mode,
                        }
                    )
                    obs = next_obs
                rows.append(
                    {
                        "env": args.env,
                        "seed": args.seed,
                        "gas_seed": args.gas_seed if args.mode in GAS_MODES else "",
                        "mode": args.mode,
                        "task_id": task_id,
                        "episode": ep,
                        "episodes": 1,
                        "success": _success(info),
                        "steps": steps,
                        "no_path_rate": no_path_steps / max(steps, 1),
                        "goal_distance_improvement": initial_goal_dist - best_goal_dist,
                        "subgoal_switch_count": subgoal_switch_count,
                        "final_goal_mode_steps": final_goal_mode_steps,
                        "tmd_rescue_activated": tmd_rescue_activated,
                        "tmd_rescue_first_step": tmd_rescue_first_step,
                        "tmd_rescue_steps": tmd_rescue_steps,
                        "tmd_rescue_bursts": tmd_rescue_bursts,
                        "tmd_rescue_returns": tmd_rescue_returns,
                        "tmd_final_rescue_first_step": tmd_final_rescue_first_step,
                        "tmd_final_rescue_steps": tmd_final_rescue_steps,
                    }
                )
        write_csv(out / "eval.csv", rows)
        write_csv(out / "step_debug.csv", debug_rows)
        return 0
    except Exception as exc:
        write_failure(out / "eval_error.json", "evaluate_tmd_graph", exc, vars(args))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
