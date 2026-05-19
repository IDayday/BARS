from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _node_phis(nodes: pd.DataFrame) -> np.ndarray:
    cols = [c for c in nodes.columns if c.startswith("phi_")]
    cols = sorted(cols, key=lambda c: int(c.split("_")[1]))
    return nodes[cols].to_numpy(np.float32)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _terminal_traj(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    terminals = np.asarray(terminals).astype(bool)
    if len(terminals):
        terminals[-1] = True
    traj = np.zeros(len(terminals), dtype=np.int32)
    step = np.zeros(len(terminals), dtype=np.int32)
    tid = 0
    t = 0
    for i in range(len(terminals)):
        traj[i] = tid
        step[i] = t
        if terminals[i]:
            tid += 1
            t = 0
        else:
            t += 1
    return traj, step


def build_boundary_scores(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    dataset_phis: Optional[np.ndarray] = None,
    terminals: Optional[np.ndarray] = None,
    psi_min: float = 0.05,
    fallback_psi: float = 0.1,
    unsupported_cross_penalty: float = 2.0,
    same_traj_bonus: float = 0.2,
    neighbor_radius: Optional[float] = None,
    max_neighbors: int = 64,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phis = _node_phis(nodes)
    u = edges["u"].to_numpy(np.int64)
    v = edges["v"].to_numpy(np.int64)
    edge_ids = edges["edge_id"].to_numpy(np.int64)
    vec = phis[v] - phis[u]
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    dirs = vec / norm

    dep_support: dict[int, np.ndarray] = {}
    arr_support: dict[int, np.ndarray] = {}
    dep_traj: dict[int, set[int]] = {}
    arr_traj: dict[int, set[int]] = {}
    if dataset_phis is not None and len(dataset_phis) >= 2:
        try:
            from sklearn.neighbors import NearestNeighbors

            data = np.asarray(dataset_phis, dtype=np.float32)
            radius = float(neighbor_radius or np.median(edges["phi_dist"].to_numpy(np.float32)) * 0.5 or 1.0)
            nn = NearestNeighbors(n_neighbors=min(max_neighbors, len(data))).fit(data)
            traj = None
            if terminals is not None:
                traj, _ = _terminal_traj(terminals)
            delta_f = np.zeros_like(data)
            delta_f[:-1] = data[1:] - data[:-1]
            delta_b = np.zeros_like(data)
            delta_b[1:] = data[1:] - data[:-1]
            for row_i, eid in enumerate(edge_ids):
                near_u = nn.kneighbors(phis[u[row_i]][None, :], return_distance=False)[0]
                near_v = nn.kneighbors(phis[v[row_i]][None, :], return_distance=False)[0]
                du = delta_f[near_u]
                dv = delta_b[near_v]
                dep_cos = (du * dirs[row_i]).sum(axis=1) / (np.linalg.norm(du, axis=1) + 1e-8)
                arr_cos = (dv * dirs[row_i]).sum(axis=1) / (np.linalg.norm(dv, axis=1) + 1e-8)
                dep = near_u[dep_cos > 0.2]
                arr = near_v[arr_cos > 0.2]
                if len(dep):
                    dep_support[int(eid)] = dep
                    if traj is not None:
                        dep_traj[int(eid)] = set(int(x) for x in traj[dep])
                if len(arr):
                    arr_support[int(eid)] = arr
                    if traj is not None:
                        arr_traj[int(eid)] = set(int(x) for x in traj[arr])
        except Exception:
            pass

    incoming: dict[int, list[int]] = {}
    outgoing: dict[int, list[int]] = {}
    eid_to_row = {int(eid): i for i, eid in enumerate(edge_ids)}
    for i, eid in enumerate(edge_ids):
        incoming.setdefault(int(v[i]), []).append(int(eid))
        outgoing.setdefault(int(u[i]), []).append(int(eid))

    rows = []
    for mid in sorted(set(incoming) & set(outgoing)):
        for prev_eid in incoming[mid]:
            prev_i = eid_to_row[prev_eid]
            for next_eid in outgoing[mid]:
                next_i = eid_to_row[next_eid]
                if prev_eid == next_eid:
                    continue
                corridor_cos = float(np.clip(np.dot(dirs[prev_i], dirs[next_i]), -1.0, 1.0))
                arr = arr_support.get(prev_eid)
                dep = dep_support.get(next_eid)
                overlap = 0
                support_type = "direction_fallback"
                support_score = 0.0
                if arr is not None and dep is not None:
                    overlap = len(set(arr.tolist()) & set(dep.tolist()))
                    same_traj = len(arr_traj.get(prev_eid, set()) & dep_traj.get(next_eid, set()))
                    support_score = math.log1p(overlap) + (same_traj_bonus if same_traj else 0.0)
                    support_type = "overlap" if overlap else ("same_traj" if same_traj else "unsupported_cross")
                penalty = unsupported_cross_penalty if support_type == "unsupported_cross" else 0.0
                raw = 1.25 * corridor_cos + support_score - penalty
                if support_type == "direction_fallback":
                    psi = min(float(fallback_psi), float(_sigmoid(raw)))
                else:
                    psi = float(_sigmoid(raw))
                psi = float(np.clip(psi, psi_min, 1.0))
                rows.append(
                    {
                        "prev_edge_id": int(prev_eid),
                        "next_edge_id": int(next_eid),
                        "psi": psi,
                        "boundary_cost": float(-np.log(max(psi, 1e-8))),
                        "support_type": support_type,
                        "overlap_count": int(overlap),
                        "corridor_cos": corridor_cos,
                    }
                )
    df = pd.DataFrame(rows)
    summary = boundary_summary(df, edges)
    summary.update(
        {
            "psi_min": float(psi_min),
            "fallback_psi": float(fallback_psi),
            "unsupported_cross_penalty": float(unsupported_cross_penalty),
            "same_traj_bonus": float(same_traj_bonus),
        }
    )
    return df, summary


def boundary_summary(boundary_scores: pd.DataFrame, edges: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    out: dict[str, Any] = {"num_pairs": int(len(boundary_scores))}
    if edges is not None:
        out["num_edges"] = int(len(edges))
    if len(boundary_scores) == 0:
        out["coverage"] = 0.0
        out["supported_pair_rate"] = 0.0
        return out
    supported = boundary_scores["support_type"].isin(["overlap", "same_traj"])
    out["coverage"] = float(len(boundary_scores) / max(1, len(edges) if edges is not None else len(boundary_scores)))
    out["supported_pair_rate"] = float(supported.mean())
    psi = boundary_scores["psi"].to_numpy(np.float32)
    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        out[f"psi_q{int(q * 100):02d}"] = float(np.quantile(psi, q))
    out["psi_mean"] = float(np.mean(psi))
    return out


def save_boundary_scores(
    boundary_scores: pd.DataFrame,
    summary: dict[str, Any],
    out_dir: str | os.PathLike[str],
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    csv_path = out / "boundary_scores.csv"
    boundary_scores.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)
    try:
        pq_path = out / "boundary_scores.parquet"
        boundary_scores.to_parquet(pq_path, index=False)
        paths["parquet"] = str(pq_path)
    except Exception:
        pass
    summary_path = out / "boundary_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    paths["summary"] = str(summary_path)
    return paths
