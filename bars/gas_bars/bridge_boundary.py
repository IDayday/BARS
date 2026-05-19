from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .bridge_graph import BRIDGE_EDGE_TYPES, RISKY_EDGE_TYPES


def filter_bridge_junctions(boundary_scores: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if len(boundary_scores) == 0:
        return boundary_scores.copy()
    edge_types = edges.set_index("edge_id")["edge_type"].astype(str).to_dict()
    out = boundary_scores.copy()
    out["prev_edge_type"] = out["prev_edge_id"].map(edge_types).fillna("")
    out["next_edge_type"] = out["next_edge_id"].map(edge_types).fillna("")
    prev_bridge = out["prev_edge_type"].isin(RISKY_EDGE_TYPES)
    next_bridge = out["next_edge_type"].isin(RISKY_EDGE_TYPES)
    out = out.loc[prev_bridge | next_bridge].copy()
    if "psi_bridge" not in out:
        out["psi_bridge"] = pd.to_numeric(out.get("psi", 0.5), errors="coerce").fillna(0.5)
    return out.reset_index(drop=True)


def synthesize_bridge_junctions(edges: pd.DataFrame, max_pairs_per_mid: int = 256) -> pd.DataFrame:
    """Create bridge-junction diagnostic pairs without all-pair local boundary scoring."""
    if len(edges) == 0 or "edge_type" not in edges:
        return pd.DataFrame()
    incoming: dict[int, list[int]] = {}
    outgoing: dict[int, list[int]] = {}
    edge_by_id = {}
    for r in edges.itertuples(index=False):
        eid = int(r.edge_id)
        incoming.setdefault(int(r.v), []).append(eid)
        outgoing.setdefault(int(r.u), []).append(eid)
        edge_by_id[eid] = r
    rows = []
    for mid in sorted(set(incoming) & set(outgoing)):
        count = 0
        for prev_eid in incoming[mid]:
            prev = edge_by_id[prev_eid]
            prev_type = str(getattr(prev, "edge_type", ""))
            for next_eid in outgoing[mid]:
                nxt = edge_by_id[next_eid]
                next_type = str(getattr(nxt, "edge_type", ""))
                if prev_eid == next_eid:
                    continue
                if prev_type not in RISKY_EDGE_TYPES and next_type not in RISKY_EDGE_TYPES:
                    continue
                # Minimal structural psi: support junctions that bridge into/out of
                # a local edge more than bridge-to-bridge handoffs.
                bridge_bridge = prev_type in RISKY_EDGE_TYPES and next_type in RISKY_EDGE_TYPES
                psi = 0.35 if bridge_bridge else 0.60
                rows.append(
                    {
                        "prev_edge_id": int(prev_eid),
                        "next_edge_id": int(next_eid),
                        "psi": float(psi),
                        "psi_bridge": float(psi),
                        "support_type": "synthetic_bridge_bridge" if bridge_bridge else "synthetic_bridge_local",
                        "prev_edge_type": prev_type,
                        "next_edge_type": next_type,
                    }
                )
                count += 1
                if count >= max_pairs_per_mid:
                    break
            if count >= max_pairs_per_mid:
                break
    return pd.DataFrame(rows)


def boundary_junction_metrics(junctions: pd.DataFrame, edge_exec: Optional[pd.DataFrame] = None) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"junction_count": int(len(junctions))}
    if len(junctions) == 0:
        metrics.update({"psi_AUROC_for_conditional_success": float("nan"), "supported_gap": float("nan"), "coverage": 0.0})
        return metrics
    psi = pd.to_numeric(junctions.get("psi_bridge", junctions.get("psi", 0.5)), errors="coerce").fillna(0.5)
    metrics["psi_mean"] = float(psi.mean())
    metrics["psi_q50"] = float(psi.quantile(0.5))
    supported = junctions.get("support_type", pd.Series([""] * len(junctions))).astype(str).isin(["overlap", "same_traj"])
    metrics["supported_pair_rate"] = float(supported.mean())
    if edge_exec is not None and len(edge_exec) and {"edge_id", "success"}.issubset(edge_exec.columns):
        success = edge_exec.set_index("edge_id")["success"].astype(float).to_dict()
        y = []
        p = []
        sup = []
        for r in junctions.itertuples(index=False):
            s1 = success.get(int(r.prev_edge_id))
            s2 = success.get(int(r.next_edge_id))
            if s1 is None or s2 is None:
                continue
            # Weak 2-edge proxy: conditional next-edge success after the first edge.
            y.append(float(s1 >= 0.5 and s2 >= 0.5))
            p.append(float(getattr(r, "psi_bridge", getattr(r, "psi", 0.5))))
            sup.append(str(getattr(r, "support_type", "")) in ["overlap", "same_traj"])
        if len(set(y)) > 1:
            metrics["psi_AUROC_for_conditional_success"] = float(roc_auc_score(y, p))
        else:
            metrics["psi_AUROC_for_conditional_success"] = float("nan")
        if y:
            y_arr = np.asarray(y)
            sup_arr = np.asarray(sup, dtype=bool)
            metrics["conditional_success_rate"] = float(y_arr.mean())
            metrics["supported_success_rate"] = float(y_arr[sup_arr].mean()) if sup_arr.any() else float("nan")
            metrics["unsupported_success_rate"] = float(y_arr[~sup_arr].mean()) if (~sup_arr).any() else float("nan")
            metrics["supported_gap"] = float(metrics["supported_success_rate"] - metrics["unsupported_success_rate"]) if np.isfinite(metrics["supported_success_rate"]) and np.isfinite(metrics["unsupported_success_rate"]) else float("nan")
            metrics["coverage"] = float(len(y) / len(junctions))
    else:
        metrics.update({"psi_AUROC_for_conditional_success": float("nan"), "supported_gap": float("nan"), "coverage": 0.0})
    return metrics


def save_bridge_junctions(junctions: pd.DataFrame, metrics: dict[str, float | int], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    junctions.to_csv(out / "bridge_boundary_junctions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out / "bridge_boundary_metrics.csv", index=False)
