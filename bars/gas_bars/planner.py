from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .graph_table import add_virtual_start_goal_edges, export_edges, export_nodes


@dataclass
class PlanResult:
    node_ids: list[int] = field(default_factory=list)
    edge_ids: list[int] = field(default_factory=list)
    subgoal_phis: list[np.ndarray] = field(default_factory=list)
    temporal_cost: float = 0.0
    exec_risk: float = 0.0
    boundary_risk: float = 0.0
    total_risk: float = 0.0
    predicted_success: float = 0.0
    path_len: int = 0
    no_path: bool = True
    reject_reason: str = "unplanned"
    planner_latency_ms: float = 0.0
    variant: str = "gas_shortest"
    budget: float = 0.0
    lambdas: dict[str, float] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return not self.no_path

    def to_row(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "budget": self.budget,
            "lambda_exec": self.lambdas.get("exec", 0.0),
            "lambda_boundary": self.lambdas.get("boundary", 0.0),
            "no_path": int(self.no_path),
            "reject_reason": self.reject_reason,
            "path_len": self.path_len,
            "temporal_cost": self.temporal_cost,
            "exec_risk": self.exec_risk,
            "boundary_risk": self.boundary_risk,
            "total_risk": self.total_risk,
            "predicted_success": self.predicted_success,
            "planner_latency_ms": self.planner_latency_ms,
            "path_node_ids": "|".join(map(str, self.node_ids)),
            "path_edge_ids": "|".join(map(str, self.edge_ids)),
        }


@dataclass
class _PlanGraph:
    edges: pd.DataFrame
    out: dict[int, list[int]]
    inc: dict[int, list[int]]
    eid_to_row: dict[int, int]
    edge_id: np.ndarray
    u: np.ndarray
    v: np.ndarray
    temporal: np.ndarray
    r_exec: np.ndarray


def _phi_cols(nodes: pd.DataFrame) -> list[str]:
    return sorted([c for c in nodes.columns if c.startswith("phi_")], key=lambda c: int(c.split("_")[1]))


def _node_phis(nodes: pd.DataFrame) -> np.ndarray:
    return nodes[_phi_cols(nodes)].to_numpy(np.float32)


def _as_tables(
    graph: Any,
    source_phi: np.ndarray,
    goal_phi: np.ndarray,
    virtual_nodes: bool,
    force_closest: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    if isinstance(graph, dict):
        nodes = graph["nodes"].copy()
        edges = graph["edges"].copy()
        way_steps = graph.get("way_steps", None)
        if virtual_nodes:
            nodes, edges = _add_virtual_to_tables(nodes, edges, source_phi, goal_phi, way_steps, force_closest=force_closest)
    elif hasattr(graph, "nodes") and hasattr(graph, "edges"):
        nodes = graph.nodes.copy()
        edges = graph.edges.copy()
        if virtual_nodes:
            nodes, edges = _add_virtual_to_tables(nodes, edges, source_phi, goal_phi, getattr(graph, "way_steps", None), force_closest=force_closest)
    elif hasattr(graph, "graph") and hasattr(graph, "nodes"):
        if virtual_nodes:
            tables = add_virtual_start_goal_edges(
                graph,
                source_phi,
                goal_phi,
                way_steps=getattr(graph, "way_steps", None),
                force_closest=force_closest,
            )
            nodes, edges = tables.nodes, tables.edges
        else:
            nodes, edges = export_nodes(graph), export_edges(graph)
    else:
        raise TypeError("graph must be a GAS keygraph, GASGraphTables, or dict with nodes/edges")
    phis = _node_phis(nodes)
    if virtual_nodes and (nodes["node_type"] == "virtual_start").any():
        start = int(nodes.loc[nodes["node_type"] == "virtual_start", "node_id"].iloc[-1])
        goal = int(nodes.loc[nodes["node_type"] == "virtual_goal", "node_id"].iloc[-1])
    else:
        start = int(nodes.iloc[np.argmin(np.linalg.norm(phis - source_phi[None, :], axis=1))]["node_id"])
        goal = int(nodes.iloc[np.argmin(np.linalg.norm(phis - goal_phi[None, :], axis=1))]["node_id"])
    return nodes, edges, start, goal


def _add_virtual_to_tables(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    source_phi: np.ndarray,
    goal_phi: np.ndarray,
    way_steps: Optional[float],
    k: int = 16,
    force_closest: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phis = _node_phis(nodes)
    start_id = int(nodes["node_id"].max()) + 1 if len(nodes) else 0
    goal_id = start_id + 1
    rows = []
    for node_id, node_type, phi in ((start_id, "virtual_start", source_phi), (goal_id, "virtual_goal", goal_phi)):
        row = {"node_id": node_id, "node_type": node_type}
        row.update({f"phi_{i}": float(v) for i, v in enumerate(np.asarray(phi).reshape(-1))})
        rows.append(row)
    nodes = pd.concat([nodes, pd.DataFrame(rows)], ignore_index=True)
    way = float(way_steps if way_steps is not None else np.median(edges["phi_dist"]) if len(edges) else 1.0)
    base_ids = nodes.loc[~nodes["node_type"].isin(["virtual_start", "virtual_goal"]), "node_id"].to_numpy(np.int64)
    base_phis = phis[: len(base_ids)]
    ds = np.linalg.norm(base_phis - np.asarray(source_phi)[None, :], axis=1)
    dg = np.linalg.norm(base_phis - np.asarray(goal_phi)[None, :], axis=1)
    sidx = np.where(ds <= way)[0]
    gidx = np.where(dg <= way)[0]
    if force_closest or len(sidx) == 0:
        sidx = np.unique(np.concatenate([sidx, np.argsort(ds)[:k]]))
    if force_closest or len(gidx) == 0:
        gidx = np.unique(np.concatenate([gidx, np.argsort(dg)[:k]]))
    next_eid = int(edges["edge_id"].max()) + 1 if len(edges) else 0
    erows = []
    for ix in sidx[:k]:
        dist = float(ds[ix])
        erows.append({"edge_id": next_eid, "u": start_id, "v": int(base_ids[ix]), "gas_weight": dist, "temporal_cost": dist, "phi_dist": dist, "is_bidirectional_partner": 0, "edge_source": "gas_goal_connector"})
        next_eid += 1
    for ix in gidx[:k]:
        dist = float(dg[ix])
        erows.append({"edge_id": next_eid, "u": int(base_ids[ix]), "v": goal_id, "gas_weight": dist, "temporal_cost": dist, "phi_dist": dist, "is_bidirectional_partner": 0, "edge_source": "gas_goal_connector"})
        next_eid += 1
    dist = float(np.linalg.norm(np.asarray(goal_phi) - np.asarray(source_phi)))
    if dist <= max(way, 1e-6):
        erows.append({"edge_id": next_eid, "u": start_id, "v": goal_id, "gas_weight": dist, "temporal_cost": dist, "phi_dist": dist, "is_bidirectional_partner": 0, "edge_source": "gas_goal_connector"})
    edges = pd.concat([edges, pd.DataFrame(erows)], ignore_index=True)
    return nodes, edges


def _merge_scores(edges: pd.DataFrame, edge_scores: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = edges.copy()
    if edge_scores is not None and "edge_id" in edge_scores:
        cols = [c for c in ["edge_id", "p_exec", "r_exec", "local_support", "same_traj_support"] if c in edge_scores.columns]
        out = out.merge(edge_scores[cols].drop_duplicates("edge_id"), on="edge_id", how="left", suffixes=("", "_score"))
    if "p_exec" not in out:
        dist = out["phi_dist"].to_numpy(np.float32)
        scale = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
        out["p_exec"] = np.exp(-dist / max(scale, 1e-6))
    fallback = np.exp(-out["phi_dist"].to_numpy(np.float32) / max(float(np.median(out["phi_dist"])) if len(out) else 1.0, 1e-6))
    out["p_exec"] = out["p_exec"].where(out["p_exec"].notna(), fallback)
    if "r_exec" not in out:
        out["r_exec"] = -np.log(np.clip(out["p_exec"].to_numpy(np.float32), 1e-6, 1.0))
    risk_fallback = -np.log(np.clip(out["p_exec"].to_numpy(np.float32), 1e-6, 1.0))
    out["r_exec"] = out["r_exec"].where(out["r_exec"].notna(), risk_fallback)
    if "temporal_cost" not in out:
        out["temporal_cost"] = out.get("gas_weight", out["phi_dist"])
    return out.reset_index(drop=True)


def _adj(edges: pd.DataFrame) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, int]]:
    out: dict[int, list[int]] = {}
    inc: dict[int, list[int]] = {}
    eid_to_row: dict[int, int] = {}
    for i, r in edges.iterrows():
        eid = int(r["edge_id"])
        eid_to_row[eid] = int(i)
        out.setdefault(int(r["u"]), []).append(eid)
        inc.setdefault(int(r["v"]), []).append(eid)
    return out, inc, eid_to_row


def _build_plan_graph(edges: pd.DataFrame) -> _PlanGraph:
    edge_id = edges["edge_id"].to_numpy(np.int64)
    u = edges["u"].to_numpy(np.int64)
    v = edges["v"].to_numpy(np.int64)
    temporal = edges["temporal_cost"].to_numpy(np.float64)
    r_exec = edges["r_exec"].to_numpy(np.float64)
    out: dict[int, list[int]] = {}
    inc: dict[int, list[int]] = {}
    eid_to_row: dict[int, int] = {}
    for row, (eid, uu, vv) in enumerate(zip(edge_id, u, v)):
        eid_i = int(eid)
        u_i = int(uu)
        v_i = int(vv)
        eid_to_row[eid_i] = row
        out.setdefault(u_i, []).append(row)
        inc.setdefault(v_i, []).append(row)
    return _PlanGraph(edges, out, inc, eid_to_row, edge_id, u, v, temporal, r_exec)


def _boundary_map(boundary_scores: Optional[pd.DataFrame]) -> dict[tuple[int, int], float]:
    if boundary_scores is None or len(boundary_scores) == 0:
        return {}
    return {
        (int(r.prev_edge_id), int(r.next_edge_id)): float(r.boundary_cost)
        for r in boundary_scores.itertuples(index=False)
    }


def _finish(
    res: PlanResult,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    start_time: float,
) -> PlanResult:
    phis = _node_phis(nodes)
    node_to_row = {int(row.node_id): i for i, row in enumerate(nodes[["node_id"]].itertuples(index=False))}
    res.subgoal_phis = [phis[node_to_row[n]] for n in res.node_ids[1:]]
    res.path_len = len(res.edge_ids)
    res.total_risk = float(res.exec_risk + res.boundary_risk)
    res.predicted_success = float(math.exp(-res.total_risk)) if not res.no_path else 0.0
    res.planner_latency_ms = (time.perf_counter() - start_time) * 1000.0
    return res


def _empty(reason: str, variant: str, budget: float, lambdas: dict[str, float], start_time: float) -> PlanResult:
    return PlanResult(
        no_path=True,
        reject_reason=reason,
        variant=variant,
        budget=float(budget),
        lambdas=lambdas,
        planner_latency_ms=(time.perf_counter() - start_time) * 1000.0,
    )


def _edge_values(pg: _PlanGraph, eids: list[int], bmap: dict[tuple[int, int], float]) -> tuple[float, float, float]:
    if not eids:
        return 0.0, 0.0, 0.0
    rows = [pg.eid_to_row[e] for e in eids]
    temporal = float(pg.temporal[rows].sum())
    exec_risk = float(pg.r_exec[rows].sum())
    boundary = float(sum(bmap.get((a, b), -math.log(0.1)) for a, b in zip(eids[:-1], eids[1:])))
    return temporal, exec_risk, boundary


def _node_dijkstra(pg: _PlanGraph, start: int, goal: int, objective: str, lambda_exec: float, max_edges: int):
    if not pg.out.get(start):
        return None, "no_start_connection"
    if not pg.inc.get(goal):
        return None, "no_goal_connection"
    pq = [(0.0, start, 0, [], [start])]
    best: dict[int, float] = {start: 0.0}
    hit_edge_limit = False
    while pq:
        cost, node, depth, epath, npath = heapq.heappop(pq)
        if node == goal:
            return (epath, npath), ""
        if depth >= max_edges:
            if pg.out.get(node):
                hit_edge_limit = True
            continue
        if cost > best.get(node, float("inf")) + 1e-9:
            continue
        for row in pg.out.get(node, []):
            eid = int(pg.edge_id[row])
            w = float(pg.temporal[row])
            if objective == "reachability":
                w += lambda_exec * float(pg.r_exec[row])
            nxt = int(pg.v[row])
            nc = cost + w
            if nc < best.get(nxt, float("inf")):
                best[nxt] = nc
                heapq.heappush(pq, (nc, nxt, depth + 1, epath + [eid], npath + [nxt]))
    return None, "max_edges_exceeded" if hit_edge_limit else "disconnected"


def _dominates(a: tuple[float, float, int], b: tuple[float, float, int], eps: float) -> bool:
    return a[0] <= b[0] + eps and a[1] <= b[1] + eps and a[2] <= b[2]


def _budget_node(pg: _PlanGraph, start: int, goal: int, budget: float, max_edges: int, max_labels_per_node: int, risk_epsilon: float):
    if not pg.out.get(start):
        return None, "no_start_connection"
    if not pg.inc.get(goal):
        return None, "no_goal_connection"
    labels: dict[int, list[tuple[float, float, int, list[int], list[int]]]] = {start: [(0.0, 0.0, 0, [], [start])]}
    pq = [(0.0, 0.0, start, 0, [], [start])]
    best_over_budget = False
    hit_edge_limit = False
    while pq:
        cost, risk, node, depth, epath, npath = heapq.heappop(pq)
        if node == goal:
            return (epath, npath), ""
        if depth >= max_edges:
            if pg.out.get(node):
                hit_edge_limit = True
            continue
        for row in pg.out.get(node, []):
            eid = int(pg.edge_id[row])
            nrisk = risk + float(pg.r_exec[row])
            if nrisk > budget + risk_epsilon:
                best_over_budget = True
                continue
            ncost = cost + float(pg.temporal[row])
            nxt = int(pg.v[row])
            label_key = (ncost, nrisk, depth + 1)
            cur = labels.setdefault(nxt, [])
            if any(_dominates((c, rr, d), label_key, risk_epsilon) for c, rr, d, _, _ in cur):
                continue
            cur[:] = [lab for lab in cur if not _dominates(label_key, (lab[0], lab[1], lab[2]), risk_epsilon)]
            cur.append((ncost, nrisk, depth + 1, epath + [eid], npath + [nxt]))
            cur.sort(key=lambda x: (x[0], x[1]))
            del cur[max_labels_per_node:]
            heapq.heappush(pq, (ncost, nrisk, nxt, depth + 1, epath + [eid], npath + [nxt]))
    if best_over_budget:
        return None, "budget_infeasible"
    return None, "max_edges_exceeded" if hit_edge_limit else "disconnected"


def _line_graph(
    pg: _PlanGraph,
    start: int,
    goal: int,
    bmap: dict[tuple[int, int], float],
    budget: float,
    lambda_exec: float,
    lambda_boundary: float,
    max_edges: int,
    max_labels_per_node: int,
    risk_epsilon: float,
    hard_budget: bool,
):
    if not pg.out.get(start):
        return None, "no_start_connection"
    if not pg.inc.get(goal):
        return None, "no_goal_connection"
    fallback_boundary = -math.log(0.1)
    if hard_budget:
        labels: dict[int, list[tuple[float, float, int, list[int], list[int]]]] = {}
        pq = []
        for row in pg.out[start]:
            eid = int(pg.edge_id[row])
            risk = float(pg.r_exec[row])
            if risk > budget + risk_epsilon:
                continue
            cost = float(pg.temporal[row])
            node = int(pg.v[row])
            lab = (cost, risk, 1, [eid], [start, node])
            labels.setdefault(eid, []).append(lab)
            heapq.heappush(pq, (cost, risk, eid, lab))
        best_over_budget = len(pq) == 0
        hit_edge_limit = False
        while pq:
            cost, risk, last_eid, lab = heapq.heappop(pq)
            _, _, depth, epath, npath = lab
            node = npath[-1]
            if node == goal:
                return (epath, npath), ""
            if depth >= max_edges:
                if pg.out.get(node):
                    hit_edge_limit = True
                continue
            for nrow in pg.out.get(node, []):
                ne = int(pg.edge_id[nrow])
                b = float(bmap.get((last_eid, ne), fallback_boundary))
                nrisk = risk + float(pg.r_exec[nrow]) + b
                if nrisk > budget + risk_epsilon:
                    best_over_budget = True
                    continue
                ncost = cost + float(pg.temporal[nrow])
                nxt = int(pg.v[nrow])
                key = (ncost, nrisk, depth + 1)
                cur = labels.setdefault(ne, [])
                if any(_dominates((c, rsk, d), key, risk_epsilon) for c, rsk, d, _, _ in cur):
                    continue
                nlab = (ncost, nrisk, depth + 1, epath + [ne], npath + [nxt])
                cur[:] = [x for x in cur if not _dominates(key, (x[0], x[1], x[2]), risk_epsilon)]
                cur.append(nlab)
                cur.sort(key=lambda x: (x[0], x[1]))
                del cur[max_labels_per_node:]
                heapq.heappush(pq, (ncost, nrisk, ne, nlab))
        if best_over_budget:
            return None, "budget_infeasible"
        return None, "max_edges_exceeded" if hit_edge_limit else "disconnected"

    pq = []
    best: dict[int, float] = {}
    for row in pg.out[start]:
        eid = int(pg.edge_id[row])
        obj = float(pg.temporal[row]) + lambda_exec * float(pg.r_exec[row])
        node = int(pg.v[row])
        heapq.heappush(pq, (obj, eid, 1, [eid], [start, node]))
        best[eid] = obj
    hit_edge_limit = False
    while pq:
        obj, last_eid, depth, epath, npath = heapq.heappop(pq)
        node = npath[-1]
        if node == goal:
            return (epath, npath), ""
        if depth >= max_edges:
            if pg.out.get(node):
                hit_edge_limit = True
            continue
        if obj > best.get(last_eid, float("inf")) + 1e-9:
            continue
        for nrow in pg.out.get(node, []):
            ne = int(pg.edge_id[nrow])
            b = float(bmap.get((last_eid, ne), fallback_boundary))
            nobj = obj + float(pg.temporal[nrow]) + lambda_exec * float(pg.r_exec[nrow]) + lambda_boundary * b
            if nobj < best.get(ne, float("inf")):
                best[ne] = nobj
                heapq.heappush(pq, (nobj, ne, depth + 1, epath + [ne], npath + [int(pg.v[nrow])]))
    return None, "max_edges_exceeded" if hit_edge_limit else "disconnected"


def plan(
    graph: Any,
    source_phi: np.ndarray,
    goal_phi: np.ndarray,
    task_id: Optional[int] = None,
    variant: str = "gas_shortest",
    edge_scores: Optional[pd.DataFrame] = None,
    boundary_scores: Optional[pd.DataFrame] = None,
    budget: float = 2.0,
    lambda_exec: float = 1.0,
    lambda_boundary: float = 1.0,
    max_edges: int = 20,
    virtual_nodes: bool = True,
    force_closest: bool = True,
    max_labels_per_node: int = 64,
    risk_epsilon: float = 1e-6,
) -> PlanResult:
    start_time = time.perf_counter()
    requested_variant = variant.lower()
    alias = {
        "gas_reachability_budget_calibrated": "gas_reachability_budget",
        "gas_reachability_soft_calibrated": "gas_reachability_soft",
        "gas_shortest_replan_on_local_drift": "gas_shortest",
        "gas_shortest_adaptive_subgoal_horizon": "gas_shortest",
        "gas_reachability_budget_replan_on_local_drift": "gas_reachability_budget",
        "gas_shortest_subgoal_refresh_on_target_distance_increase": "gas_shortest",
        "gas_shortest_nearest_reachable_subgoal_on_path": "gas_shortest",
        "gas_shortest_drift_replan_with_cooldown": "gas_shortest",
        "gas_boundary_scaled_budget": "gas_boundary_budget",
        "gas_boundary_soft_scaled": "gas_boundary_soft",
    }
    variant = alias.get(requested_variant, requested_variant)
    lambdas = {"exec": float(lambda_exec), "boundary": float(lambda_boundary)}
    try:
        nodes, edges, start, goal = _as_tables(graph, np.asarray(source_phi), np.asarray(goal_phi), virtual_nodes, force_closest)
        edges = _merge_scores(edges, edge_scores)
        bmap = _boundary_map(boundary_scores)
        pg = _build_plan_graph(edges)
        if start == goal:
            res = PlanResult([start], [], [], 0.0, 0.0, 0.0, 0.0, 1.0, 0, False, "", 0.0, requested_variant, budget, lambdas)
            return _finish(res, nodes, edges, start_time)
        if variant == "gas_lagrangian_sweep":
            best: Optional[PlanResult] = None
            for le in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
                for lb in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
                    cand = plan(graph, source_phi, goal_phi, task_id, "gas_boundary_soft", edge_scores, boundary_scores, budget, le, lb, max_edges, virtual_nodes, force_closest)
                    if not cand.no_path and cand.total_risk <= budget + risk_epsilon:
                        if best is None or cand.temporal_cost < best.temporal_cost:
                            best = cand
            if best is None:
                return _empty("budget_infeasible", requested_variant, budget, lambdas, start_time)
            best.variant = requested_variant
            best.lambdas = lambdas
            best.planner_latency_ms = (time.perf_counter() - start_time) * 1000.0
            return best
        if variant == "gas_shortest":
            found, reason = _node_dijkstra(pg, start, goal, "shortest", lambda_exec, max_edges)
        elif variant == "gas_reachability_soft":
            found, reason = _node_dijkstra(pg, start, goal, "reachability", lambda_exec, max_edges)
        elif variant == "gas_reachability_budget":
            shortest, shortest_reason = _node_dijkstra(pg, start, goal, "shortest", lambda_exec, max_edges)
            if shortest is not None:
                epath0, _ = shortest
                _, risk0, _ = _edge_values(pg, epath0, {})
                if risk0 <= budget + risk_epsilon:
                    found, reason = shortest, ""
                else:
                    found, reason = _budget_node(pg, start, goal, budget, max_edges, max_labels_per_node, risk_epsilon)
            else:
                found, reason = _budget_node(pg, start, goal, budget, max_edges, max_labels_per_node, risk_epsilon)
                if found is None and shortest_reason in {"no_start_connection", "no_goal_connection"}:
                    reason = shortest_reason
        elif variant == "gas_boundary_soft":
            found, reason = _line_graph(pg, start, goal, bmap, budget, lambda_exec, lambda_boundary, max_edges, max_labels_per_node, risk_epsilon, hard_budget=False)
        elif variant == "gas_boundary_budget":
            found, reason = _line_graph(pg, start, goal, bmap, budget, lambda_exec, lambda_boundary, max_edges, max_labels_per_node, risk_epsilon, hard_budget=True)
        else:
            raise ValueError(f"Unknown Stage22 planner variant: {variant}")
        if found is None:
            if reason == "disconnected" and pg.out.get(start) and pg.inc.get(goal):
                reason = "max_edges_exceeded" if max_edges <= 1 else reason
            return _empty(reason, requested_variant, budget, lambdas, start_time)
        epath, npath = found
        temporal, exec_risk, boundary = _edge_values(pg, epath, bmap)
        if variant == "gas_reachability_budget":
            boundary = 0.0
        res = PlanResult(
            node_ids=[int(x) for x in npath],
            edge_ids=[int(x) for x in epath],
            temporal_cost=temporal,
            exec_risk=exec_risk,
            boundary_risk=boundary if "boundary" in variant or variant == "gas_lagrangian_sweep" else 0.0,
            no_path=False,
            reject_reason="",
            variant=requested_variant,
            budget=float(budget),
            lambdas=lambdas,
        )
        return _finish(res, nodes, edges, start_time)
    except Exception as exc:
        res = _empty(f"planner_error:{type(exc).__name__}:{exc}", requested_variant if "requested_variant" in locals() else variant, budget, lambdas, start_time)
        return res
