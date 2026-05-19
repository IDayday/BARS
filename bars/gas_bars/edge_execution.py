from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from bars.external.gas_backbone import GASBackbone
from .bridge_graph import BRIDGE_EDGE_TYPES, node_phis


def _node_obs_lookup(backbone: GASBackbone, nodes: pd.DataFrame, dataset_embeddings: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    _, train_dataset, _ = backbone.load_env_and_dataset()
    observations = np.asarray(train_dataset["observations"])
    phis = np.asarray(dataset_embeddings, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=1).fit(phis)
    node_phi = node_phis(nodes)
    _, idx = nn.kneighbors(node_phi)
    node_ids = nodes["node_id"].to_numpy(np.int64)
    return {int(n): int(i[0]) for n, i in zip(node_ids, idx)}, observations


def _unwrap_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def try_set_state_from_observation(env: Any, observation: np.ndarray) -> tuple[bool, str]:
    """Best-effort arbitrary reset for MuJoCo-like AntMaze states."""
    raw = _unwrap_env(env)
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    model = getattr(raw, "model", None)
    if model is None and hasattr(raw, "env"):
        model = getattr(raw.env, "model", None)
    nq = int(getattr(model, "nq", 0) or 0)
    nv = int(getattr(model, "nv", 0) or 0)
    setter = getattr(raw, "set_state", None)
    if setter is None and hasattr(raw, "env"):
        setter = getattr(raw.env, "set_state", None)
    if setter is None:
        return False, "set_state_unavailable"
    if nq <= 0 or nv <= 0:
        return False, "model_nq_nv_unavailable"
    if obs.size >= nq + nv:
        qpos = obs[:nq].copy()
        qvel = obs[nq : nq + nv].copy()
    elif obs.size == nq + nv - 2:
        qpos = np.zeros(nq, dtype=np.float64)
        qpos[2:] = obs[: max(0, nq - 2)]
        qvel = obs[max(0, nq - 2) : max(0, nq - 2) + nv].copy()
    else:
        return False, f"obs_dim_{obs.size}_not_compatible_with_nq_{nq}_nv_{nv}"
    try:
        setter(qpos, qvel)
        if hasattr(raw, "do_simulation"):
            pass
        return True, "set_state"
    except Exception as exc:
        return False, f"set_state_error:{type(exc).__name__}:{exc}"


def _edge_proxy(row: pd.Series, way_steps: float) -> tuple[int, float, float, str]:
    phi_dist = float(row.get("phi_dist", 0.0) or 0.0)
    et = str(row.get("edge_type", ""))
    if et in {"safe_local", "same_traj_temporal", "virtual_connector"}:
        p = math.exp(-max(phi_dist - way_steps, 0.0) / max(way_steps, 1e-6))
    elif et in BRIDGE_EDGE_TYPES:
        p = math.exp(-phi_dist / max(way_steps * 1.5, 1e-6))
    else:
        p = math.exp(-phi_dist / max(way_steps, 1e-6))
    return int(p >= 0.5), phi_dist, phi_dist, "weak_proxy_no_reset"


def execute_edge(
    backbone: GASBackbone,
    env: Any,
    nodes: pd.DataFrame,
    edge: pd.Series,
    node_to_obs: Optional[dict[int, int]],
    observations: Optional[np.ndarray],
    horizon: int,
    reach_threshold: float,
    way_steps: float,
) -> dict[str, Any]:
    start_time = time.time()
    row = edge.to_dict()
    row.setdefault("edge_type", str(edge.get("edge_type", "")))
    row.setdefault("graph_id", str(edge.get("graph_id", "")))
    if node_to_obs is None or observations is None:
        success, min_dist, final_dist, mode = _edge_proxy(edge, way_steps)
        row.update({"success": success, "min_dist": min_dist, "final_dist": final_dist, "steps": 0, "timeout": 0, "reset_mode": mode, "duration_sec": time.time() - start_time})
        return row
    node_id = nodes["node_id"].to_numpy(np.int64)
    phis = node_phis(nodes)
    idx = {int(n): i for i, n in enumerate(node_id)}
    u = int(edge["u"])
    v = int(edge["v"])
    if u not in node_to_obs or v not in idx:
        success, min_dist, final_dist, mode = _edge_proxy(edge, way_steps)
        row.update({"success": success, "min_dist": min_dist, "final_dist": final_dist, "steps": 0, "timeout": 0, "reset_mode": mode, "duration_sec": time.time() - start_time})
        return row
    start_obs = np.asarray(observations[node_to_obs[u]])
    target_phi = phis[idx[v]]
    try:
        env.reset(seed=int(row.get("edge_id", 0)) % 1_000_000)
        ok, reason = try_set_state_from_observation(env, start_obs)
        if not ok:
            success, min_dist, final_dist, mode = _edge_proxy(edge, way_steps)
            row.update({"success": success, "min_dist": min_dist, "final_dist": final_dist, "steps": 0, "timeout": 0, "reset_mode": f"weak_proxy_{reason}", "duration_sec": time.time() - start_time})
            return row
        obs = start_obs.copy()
        min_dist = float("inf")
        final_dist = float("inf")
        steps = 0
        for steps in range(1, horizon + 1):
            action = backbone.sample_action(obs, target_phi, final_goal=False)
            obs, _, done, _ = backbone.step_env(env, backbone.env_name or "", action)
            phi = backbone.get_phi(obs)
            final_dist = float(np.linalg.norm(phi - target_phi))
            min_dist = min(min_dist, final_dist)
            if final_dist <= reach_threshold:
                break
            if done:
                break
        success = int(min_dist <= reach_threshold)
        row.update({"success": success, "min_dist": min_dist, "final_dist": final_dist, "steps": steps, "timeout": int(steps >= horizon and not success), "reset_mode": "set_state", "duration_sec": time.time() - start_time})
        return row
    except Exception as exc:
        success, min_dist, final_dist, mode = _edge_proxy(edge, way_steps)
        row.update({"success": success, "min_dist": min_dist, "final_dist": final_dist, "steps": 0, "timeout": 0, "reset_mode": f"{mode}_after_error:{type(exc).__name__}", "error": repr(exc), "duration_sec": time.time() - start_time})
        return row


def sample_edge_pool(edges: pd.DataFrame, local_n: int = 200, bridge_n: int = 300, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    parts = []
    for name, n in [
        ("safe_local", local_n),
        ("same_traj_temporal", local_n),
        ("gas_cross", bridge_n),
        ("aggressive_tdr_bridge", bridge_n),
        ("bottleneck_bridge", bridge_n),
    ]:
        sub = edges[edges["edge_type"].astype(str).eq(name)] if "edge_type" in edges else pd.DataFrame()
        if len(sub):
            take = min(len(sub), n)
            parts.append(sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    if not parts:
        return edges.sample(n=min(len(edges), local_n + bridge_n), random_state=random_state) if len(edges) else edges
    return pd.concat(parts, ignore_index=True).drop_duplicates("edge_id")


def run_edge_execution(
    backbone: GASBackbone,
    graph: dict[str, Any],
    out_csv: str | Path,
    local_n: int = 200,
    bridge_n: int = 300,
    horizon: Optional[int] = None,
    reach_threshold: Optional[float] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    nodes = graph["nodes"]
    edges = graph["edges"]
    way_steps = float(graph.get("way_steps", 8.0))
    horizon = int(horizon or way_steps)
    reach_threshold = float(reach_threshold or way_steps)
    artifacts = backbone.artifacts
    node_to_obs = None
    observations = None
    if artifacts is not None and artifacts.dataset_embeddings is not None:
        try:
            emb = np.load(artifacts.dataset_embeddings)
            node_to_obs, observations = _node_obs_lookup(backbone, nodes, emb)
        except Exception:
            node_to_obs, observations = None, None
    env, _, _ = backbone.load_env_and_dataset()
    pool = sample_edge_pool(edges, local_n=local_n, bridge_n=bridge_n, random_state=random_state)
    rows = []
    for _, edge in pool.iterrows():
        rows.append(execute_edge(backbone, env, nodes, edge, node_to_obs, observations, horizon, reach_threshold, way_steps))
    df = pd.DataFrame(rows)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    summary = {
        "edges": int(len(df)),
        "success_rate": float(df["success"].mean()) if len(df) else 0.0,
        "set_state_rate": float((df.get("reset_mode", pd.Series(dtype=str)).astype(str) == "set_state").mean()) if len(df) else 0.0,
    }
    (path.parent / "edge_exec_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return df
