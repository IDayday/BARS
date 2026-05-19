from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .bridge_graph import BRIDGE_EDGE_TYPES, RISKY_EDGE_TYPES
from .planner import PlanResult, _as_tables, _finish


def _is_risky_edge(edge_type: str) -> bool:
    return str(edge_type) in RISKY_EDGE_TYPES


def _bridge_risk(row: Any, default_bridge_p: float = 0.5) -> float:
    et = str(getattr(row, "edge_type", "safe_local"))
    if not _is_risky_edge(et):
        return 0.0
    p = getattr(row, "p_bridge", None)
    if p is None or (isinstance(p, float) and math.isnan(p)):
        p = getattr(row, "p_exec", default_bridge_p)
    try:
        p = float(p)
    except Exception:
        p = default_bridge_p
    return float(-math.log(max(min(p, 1.0), 1e-6)))


def _merge_bridge_scores(edges: pd.DataFrame, bridge_scores: Optional[pd.DataFrame], p_bridge_min: float) -> pd.DataFrame:
    out = edges.copy()
    if "edge_type" not in out:
        out["edge_type"] = out.get("edge_source", "safe_local")
    if bridge_scores is not None and len(bridge_scores) and "edge_id" in bridge_scores:
        cols = [c for c in ["edge_id", "p_bridge", "bridge_score", "bridge_label", "edge_exec_success"] if c in bridge_scores.columns]
        out = out.merge(bridge_scores[cols].drop_duplicates("edge_id"), on="edge_id", how="left", suffixes=("", "_score"))
    if "p_bridge" not in out:
        out["p_bridge"] = np.where(out["edge_type"].astype(str).isin(RISKY_EDGE_TYPES), 0.5, 1.0)
    out["p_bridge"] = pd.to_numeric(out["p_bridge"], errors="coerce")
    out.loc[~out["edge_type"].astype(str).isin(RISKY_EDGE_TYPES), "p_bridge"] = 1.0
    risky = out["edge_type"].astype(str).isin(RISKY_EDGE_TYPES)
    accepted = (~risky) | (out["p_bridge"].fillna(0.0) >= p_bridge_min)
    return out.loc[accepted].reset_index(drop=True)


def _boundary_lookup(boundary_scores: Optional[pd.DataFrame]) -> dict[tuple[int, int], float]:
    if boundary_scores is None or len(boundary_scores) == 0:
        return {}
    psi_col = "psi_bridge" if "psi_bridge" in boundary_scores.columns else "psi"
    out: dict[tuple[int, int], float] = {}
    for r in boundary_scores.itertuples(index=False):
        try:
            psi = float(getattr(r, psi_col))
            out[(int(r.prev_edge_id), int(r.next_edge_id))] = float(-math.log(max(min(psi, 1.0), 1e-6)))
        except Exception:
            continue
    return out


def _adj(edges: pd.DataFrame) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, r in enumerate(edges[["u", "v"]].itertuples(index=False)):
        out.setdefault(int(r.u), []).append(i)
    return out


def _dominates(a: tuple[float, float, int, int], b: tuple[float, float, int, int], eps: float) -> bool:
    return a[0] <= b[0] + eps and a[1] <= b[1] + eps and a[2] <= b[2] and a[3] <= b[3]


def plan_bars_v3(
    graph: Any,
    source_phi: np.ndarray,
    goal_phi: np.ndarray,
    task_id: Optional[int] = None,
    variant: str = "p_bridge_budget",
    bridge_scores: Optional[pd.DataFrame] = None,
    boundary_scores: Optional[pd.DataFrame] = None,
    p_bridge_min: float = 0.6,
    bridge_risk_budget: float = 1.0,
    max_risky_bridges: int = 2,
    max_edges: int = 20,
    virtual_nodes: bool = True,
    force_closest: bool = True,
    max_labels_per_node: int = 64,
    risk_epsilon: float = 1e-6,
) -> PlanResult:
    """Bridge-risk constrained planner for Stage23.

    The key Stage23 distinction is that local GAS edges do not consume risk
    budget. Risk is accumulated only on risky bridge-like edges and, for the
    boundary variant, on junctions touching at least one risky edge.
    """
    del task_id
    start_time = time.perf_counter()
    requested_variant = variant
    lambdas = {"exec": 0.0, "boundary": 0.0}
    try:
        nodes, edges, start, goal = _as_tables(graph, np.asarray(source_phi), np.asarray(goal_phi), virtual_nodes, force_closest)
        edges = _merge_bridge_scores(edges, bridge_scores, p_bridge_min=p_bridge_min)
        if "temporal_cost" not in edges:
            edges["temporal_cost"] = edges.get("gas_weight", edges.get("phi_dist", 1.0))
        if "edge_id" not in edges:
            edges["edge_id"] = np.arange(len(edges))
        bmap = _boundary_lookup(boundary_scores)
        out = _adj(edges)
        edge_ids = edges["edge_id"].to_numpy(np.int64)
        u = edges["u"].to_numpy(np.int64)
        v = edges["v"].to_numpy(np.int64)
        temporal = pd.to_numeric(edges["temporal_cost"], errors="coerce").fillna(1.0).to_numpy(np.float64)
        edge_type = edges["edge_type"].astype(str).tolist()
        bridge_risk = np.array([_bridge_risk(r) for r in edges.itertuples(index=False)], dtype=np.float64)
        risky_flag = np.array([_is_risky_edge(et) for et in edge_type], dtype=bool)
        edge_to_row = {int(eid): i for i, eid in enumerate(edge_ids)}
        if start == goal:
            res = PlanResult([start], [], [], 0.0, 0.0, 0.0, 0.0, 1.0, 0, False, "", 0.0, requested_variant, bridge_risk_budget, lambdas)
            return _finish(res, nodes, edges, start_time)
        labels: dict[int, list[tuple[float, float, int, int, int, list[int], list[int]]]] = {
            int(start): [(0.0, 0.0, 0, 0, -1, [], [int(start)])]
        }
        pq: list[tuple[float, float, int, int, int, int, list[int], list[int]]] = [(0.0, 0.0, 0, 0, int(start), -1, [], [int(start)])]
        over_budget = False
        over_k = False
        hit_edge_limit = False
        while pq:
            cost, risk, k_risky, depth, node, last_eid, epath, npath = heapq.heappop(pq)
            if node == goal:
                boundary_risk = max(0.0, risk - float(sum(bridge_risk[edge_to_row[e]] for e in epath if e in edge_to_row)))
                res = PlanResult(
                    node_ids=[int(x) for x in npath],
                    edge_ids=[int(x) for x in epath],
                    temporal_cost=float(cost),
                    exec_risk=float(sum(bridge_risk[edge_to_row[e]] for e in epath if e in edge_to_row)),
                    boundary_risk=float(boundary_risk),
                    no_path=False,
                    reject_reason="",
                    variant=requested_variant,
                    budget=float(bridge_risk_budget),
                    lambdas=lambdas,
                )
                return _finish(res, nodes, edges, start_time)
            if depth >= max_edges:
                if out.get(node):
                    hit_edge_limit = True
                continue
            for row in out.get(node, []):
                eid = int(edge_ids[row])
                nrisk = risk + float(bridge_risk[row])
                nk = k_risky + int(risky_flag[row])
                if "boundary" in requested_variant and last_eid >= 0:
                    prev_row = edge_to_row.get(last_eid)
                    touches_bridge = bool(risky_flag[row] or (prev_row is not None and risky_flag[prev_row]))
                    if touches_bridge:
                        nrisk += float(bmap.get((last_eid, eid), 0.0))
                if nrisk > bridge_risk_budget + risk_epsilon:
                    over_budget = True
                    continue
                if nk > max_risky_bridges:
                    over_k = True
                    continue
                nxt = int(v[row])
                ncost = cost + float(temporal[row])
                key = (ncost, nrisk, nk, depth + 1)
                cur = labels.setdefault(nxt, [])
                if any(_dominates((c, r, k, d), key, risk_epsilon) for c, r, k, d, _, _, _ in cur):
                    continue
                cur[:] = [lab for lab in cur if not _dominates(key, (lab[0], lab[1], lab[2], lab[3]), risk_epsilon)]
                nlab = (ncost, nrisk, nk, depth + 1, eid, epath + [eid], npath + [nxt])
                cur.append(nlab)
                cur.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
                del cur[max_labels_per_node:]
                heapq.heappush(pq, (ncost, nrisk, nk, depth + 1, nxt, eid, epath + [eid], npath + [nxt]))
        reason = "bridge_budget_infeasible" if over_budget else ("risky_bridge_count_infeasible" if over_k else ("max_edges_exceeded" if hit_edge_limit else "disconnected"))
        return PlanResult(
            no_path=True,
            reject_reason=reason,
            variant=requested_variant,
            budget=float(bridge_risk_budget),
            lambdas=lambdas,
            planner_latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )
    except Exception as exc:
        return PlanResult(
            no_path=True,
            reject_reason=f"planner_error:{type(exc).__name__}:{exc}",
            variant=requested_variant,
            budget=float(bridge_risk_budget),
            lambdas=lambdas,
            planner_latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )


def summarize_plan_edges(plan: PlanResult, edges: pd.DataFrame) -> dict[str, Any]:
    if plan.no_path or not plan.edge_ids:
        return {"risky_bridge_count": 0, "bridge_edge_count": 0, "gas_cross_count": 0}
    edge_types = edges.set_index("edge_id")["edge_type"].astype(str).to_dict() if "edge_type" in edges else {}
    types = [edge_types.get(int(e), "") for e in plan.edge_ids]
    return {
        "risky_bridge_count": int(sum(t in RISKY_EDGE_TYPES for t in types)),
        "bridge_edge_count": int(sum(t in BRIDGE_EDGE_TYPES for t in types)),
        "gas_cross_count": int(sum(t == "gas_cross" for t in types)),
    }
