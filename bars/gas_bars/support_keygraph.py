from __future__ import annotations

import copy
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


SUPPORTED_MODES = {"annotate", "penalize", "prune", "penalize_and_prune"}
MISSING_SCORE_POLICIES = {"protect", "penalize", "prune"}


@dataclass(frozen=True)
class GASKeygraphPatchResult:
    key_graph: Any
    edge_audit: pd.DataFrame
    summary: dict[str, Any]


def load_keygraph_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict):
        obj = type("LoadedGASKeyGraph", (), {})()
        for key, value in data.items():
            setattr(obj, key, value)
        return obj
    return data


def save_keygraph_pickle(key_graph: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = getattr(key_graph, "__dict__", key_graph)
    with out.open("wb") as fh:
        pickle.dump(payload, fh)


def load_edge_scores_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"u", "v"} - set(df.columns)
    if missing:
        raise ValueError(f"edge score table is missing columns: {sorted(missing)}")
    out = df.copy()
    out["u"] = pd.to_numeric(out["u"], errors="raise").astype(int)
    out["v"] = pd.to_numeric(out["v"], errors="raise").astype(int)
    return out


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _edge_lookup(edge_scores: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in edge_scores.to_dict("records"):
        rows[(int(row["u"]), int(row["v"]))] = row
    return rows


def _goal_or_task_edge(key_graph: Any, u: int, v: int) -> bool:
    base_count = int(getattr(key_graph, "base_node_cnt", len(getattr(key_graph, "nodes", []))) or 0)
    return int(u) >= base_count or int(v) >= base_count


def _edge_supported(row: dict[str, Any] | None, support_column: str, min_support: float) -> bool:
    if row is None:
        return False
    return _finite_float(row.get(support_column), 0.0) >= float(min_support)


def _score_value(row: dict[str, Any] | None, column: str | None) -> float:
    if row is None or not column:
        return 0.0
    return _finite_float(row.get(column), 0.0)


def _recompute_paths_to_existing_task_nodes(key_graph: Any) -> dict[str, int]:
    """Recompute GAS cached task paths after edge weights or topology change.

    Official GAS stores task paths inside the keygraph pickle and its evaluator
    reads those cached dictionaries.  Any graph patch must refresh these fields;
    otherwise an evaluation run silently follows the old paths.
    """
    graph = getattr(key_graph, "graph", None)
    if graph is None:
        raise ValueError("key_graph has no graph")
    task_node_idx_dict = getattr(key_graph, "task_node_idx_dict", {}) or {}
    new_paths: dict[int, dict[int, list[int]]] = {}
    new_dists: dict[int, dict[int, float]] = {}
    reachable_counts: dict[str, int] = {}
    for raw_task_id, raw_target_idx in task_node_idx_dict.items():
        task_id = int(raw_task_id)
        target_idx = int(raw_target_idx)
        if not graph.has_node(target_idx):
            new_paths[task_id] = {}
            new_dists[task_id] = {}
            reachable_counts[str(task_id)] = 0
            continue
        lengths, paths = nx.single_source_dijkstra(graph, source=target_idx, weight="weight")
        task_paths: dict[int, list[int]] = {}
        task_dists: dict[int, float] = {}
        for node_idx, path in paths.items():
            node_idx = int(node_idx)
            if node_idx == target_idx:
                continue
            task_paths[node_idx] = [int(x) for x in path[::-1]]
            task_dists[node_idx] = float(lengths[node_idx])
        new_paths[task_id] = task_paths
        new_dists[task_id] = task_dists
        reachable_counts[str(task_id)] = len(task_paths)
    key_graph.task_paths_dict = new_paths
    key_graph.task_paths_dist_dict = new_dists
    return reachable_counts


def patch_gas_keygraph_with_support(
    key_graph: Any,
    edge_scores: pd.DataFrame,
    *,
    mode: str = "penalize",
    support_column: str = "local_support",
    min_support: float = 1.0,
    unsupported_penalty: float | None = None,
    risk_column: str | None = "r_exec",
    risk_weight: float = 0.0,
    missing_score_policy: str = "protect",
    protect_goal_edges: bool = True,
    recompute_task_paths: bool = True,
    copy_keygraph: bool = True,
) -> GASKeygraphPatchResult:
    mode = str(mode)
    missing_score_policy = str(missing_score_policy)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")
    if missing_score_policy not in MISSING_SCORE_POLICIES:
        raise ValueError(f"Unsupported missing_score_policy {missing_score_policy!r}")
    if support_column not in edge_scores.columns:
        raise ValueError(f"edge score table has no support column {support_column!r}")
    if risk_column and risk_column not in edge_scores.columns:
        raise ValueError(f"edge score table has no risk column {risk_column!r}")

    kg = copy.deepcopy(key_graph) if copy_keygraph else key_graph
    graph = getattr(kg, "graph", None)
    if graph is None:
        raise ValueError("key_graph has no graph")
    way_steps = _finite_float(getattr(kg, "way_steps", 1.0), 1.0)
    penalty = float(unsupported_penalty if unsupported_penalty is not None else max(way_steps, 1.0))
    lookup = _edge_lookup(edge_scores)

    audit_rows: list[dict[str, Any]] = []
    removed: list[tuple[int, int]] = []
    for u_raw, v_raw, attrs in list(graph.edges(data=True)):
        u = int(u_raw)
        v = int(v_raw)
        score_row = lookup.get((u, v))
        goal_edge = _goal_or_task_edge(kg, u, v)
        protected = bool(protect_goal_edges and goal_edge)
        missing_score = score_row is None
        supported = _edge_supported(score_row, support_column, min_support)
        if protected:
            effective_supported = True
            missing_action = "protected_goal_edge"
        elif not missing_score:
            effective_supported = supported
            missing_action = ""
        elif missing_score_policy == "protect":
            effective_supported = True
            missing_action = "protected_missing_score"
        elif missing_score_policy == "penalize":
            effective_supported = False
            missing_action = "penalized_missing_score"
        else:
            effective_supported = False
            missing_action = "pruned_missing_score"

        old_weight = _finite_float(attrs.get("weight"), _finite_float(attrs.get("gas_weight"), 0.0))
        if old_weight <= 0.0:
            nodes = np.asarray(getattr(kg, "nodes"))
            old_weight = float(np.linalg.norm(nodes[u] - nodes[v])) if len(nodes) > max(u, v) else 0.0
        risk = _score_value(score_row, risk_column)
        add_penalty = 0.0 if effective_supported else penalty
        new_weight = float(old_weight)
        if mode in {"penalize", "penalize_and_prune"}:
            new_weight = float(old_weight + float(risk_weight) * risk + add_penalty)
        elif mode == "annotate":
            new_weight = old_weight

        should_prune = mode in {"prune", "penalize_and_prune"} and not effective_supported and not protected
        audit_rows.append(
            {
                "u": u,
                "v": v,
                "goal_or_task_edge": int(goal_edge),
                "protected": int(protected),
                "score_available": int(not missing_score),
                "missing_score_action": missing_action,
                "support_column": support_column,
                "support_value": _score_value(score_row, support_column),
                "min_support": float(min_support),
                "supported": int(supported),
                "effective_supported": int(effective_supported),
                "risk_column": risk_column or "",
                "risk_value": risk,
                "old_weight": old_weight,
                "unsupported_penalty": add_penalty,
                "risk_penalty": float(float(risk_weight) * risk),
                "new_weight": new_weight,
                "pruned": int(should_prune),
            }
        )
        if should_prune:
            removed.append((u, v))
        else:
            attrs["bars_support_supported"] = int(effective_supported)
            attrs["bars_support_raw_supported"] = int(supported)
            attrs["bars_support_value"] = _score_value(score_row, support_column)
            attrs["bars_support_score_available"] = int(not missing_score)
            attrs["bars_support_risk"] = risk
            attrs["bars_support_original_weight"] = old_weight
            attrs["bars_support_patched_weight"] = new_weight
            attrs["weight"] = new_weight
    if removed:
        graph.remove_edges_from(removed)

    reachable_counts = _recompute_paths_to_existing_task_nodes(kg) if recompute_task_paths else {}
    audit = pd.DataFrame(audit_rows)
    non_goal = audit[audit["goal_or_task_edge"] == 0] if not audit.empty else audit
    summary = {
        "mode": mode,
        "support_column": support_column,
        "min_support": float(min_support),
        "unsupported_penalty": penalty,
        "risk_column": risk_column or "",
        "risk_weight": float(risk_weight),
        "missing_score_policy": missing_score_policy,
        "protect_goal_edges": int(protect_goal_edges),
        "recomputed_task_paths": int(recompute_task_paths),
        "num_edges_before": int(len(audit_rows)),
        "num_edges_after": int(graph.number_of_edges()),
        "num_edges_pruned": int(len(removed)),
        "num_scored_edges": int(audit["score_available"].sum()) if not audit.empty else 0,
        "num_effective_unsupported_edges": int((audit["effective_supported"] == 0).sum()) if not audit.empty else 0,
        "unsupported_edge_rate": float((audit["effective_supported"] == 0).mean()) if not audit.empty else 0.0,
        "non_goal_unsupported_edge_rate": float((non_goal["effective_supported"] == 0).mean()) if not non_goal.empty else 0.0,
        "mean_weight_delta": float((audit["new_weight"] - audit["old_weight"]).mean()) if not audit.empty else 0.0,
        "max_weight_delta": float((audit["new_weight"] - audit["old_weight"]).max()) if not audit.empty else 0.0,
        "task_reachable_node_counts": reachable_counts,
    }
    return GASKeygraphPatchResult(kg, audit, summary)


def write_patch_outputs(result: GASKeygraphPatchResult, out_keygraph_path: str | Path) -> dict[str, str]:
    out_keygraph = Path(out_keygraph_path)
    out_keygraph.parent.mkdir(parents=True, exist_ok=True)
    save_keygraph_pickle(result.key_graph, out_keygraph)
    edge_audit_path = out_keygraph.with_name(out_keygraph.stem + "_edge_audit.csv")
    summary_path = out_keygraph.with_name(out_keygraph.stem + "_summary.json")
    result.edge_audit.to_csv(edge_audit_path, index=False)
    summary_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "keygraph_path": str(out_keygraph),
        "edge_audit_path": str(edge_audit_path),
        "summary_path": str(summary_path),
    }
