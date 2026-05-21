#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.gas_bars.bridge_graph import BRIDGE_EDGE_TYPES, load_bridge_graph


CLASSES = [
    "local_local_edge_junctions",
    "bridge_entry_exit_junctions",
    "bridge_bridge_junctions",
    "virtual_start_goal_connector_pairs",
]

COLUMNS = [
    "env",
    "seed",
    "graph_id",
    "junction_class",
    "junction_count",
    "supported_count",
    "unsupported_count",
    "coverage",
    "supported_success_rate",
    "unsupported_success_rate",
    "supported_gap",
    "psi_AUROC",
    "psi_AUPRC",
]


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _junction_class(prev_type: str, next_type: str) -> str:
    bridge_prev = prev_type in BRIDGE_EDGE_TYPES
    bridge_next = next_type in BRIDGE_EDGE_TYPES
    virtual = "virtual" in prev_type or "virtual" in next_type
    if virtual:
        return "virtual_start_goal_connector_pairs"
    if bridge_prev and bridge_next:
        return "bridge_bridge_junctions"
    if bridge_prev or bridge_next:
        return "bridge_entry_exit_junctions"
    return "local_local_edge_junctions"


def _safe_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan, np.nan
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    except Exception:
        return np.nan, np.nan


def analyze_one(env: str, seed: int, graph_id: str, artifact_root: Path, oracle_reports_root: Path) -> list[dict[str, object]]:
    graph_path = artifact_root / env / f"seed{seed}" / "bridge_graphs" / f"{graph_id}.pkl"
    if not graph_path.exists():
        return []
    bundle = load_bridge_graph(graph_path).to_dict()
    edges = bundle["edges"].copy()
    edge_types = {int(r.edge_id): str(r.edge_type) for r in edges[["edge_id", "edge_type"]].itertuples(index=False)}
    edge_by_u: dict[int, list[int]] = {}
    edge_by_v: dict[int, list[int]] = {}
    for r in edges[["edge_id", "u", "v"]].itertuples(index=False):
        edge_by_u.setdefault(int(r.u), []).append(int(r.edge_id))
        edge_by_v.setdefault(int(r.v), []).append(int(r.edge_id))
    boundary = _read(artifact_root / env / f"seed{seed}" / "boundary_scores.csv")
    if len(boundary) == 0:
        boundary = _read(artifact_root / env / f"seed{seed}" / "graph" / "boundary_scores.csv")
    bmap = {}
    if len(boundary):
        for r in boundary.itertuples(index=False):
            bmap[(int(getattr(r, "prev_edge_id")), int(getattr(r, "next_edge_id")))] = {
                "psi": float(getattr(r, "psi", np.nan)),
                "supported": str(getattr(r, "support_type", "")) in {"overlap", "same_traj"},
            }
    class_rows: dict[str, list[tuple[int, int, float, bool]]] = {name: [] for name in CLASSES}
    for node in sorted(set(edge_by_v) & set(edge_by_u)):
        for prev_eid in edge_by_v[node]:
            for next_eid in edge_by_u[node]:
                if prev_eid == next_eid:
                    continue
                cls = _junction_class(edge_types.get(prev_eid, ""), edge_types.get(next_eid, ""))
                info = bmap.get((prev_eid, next_eid), {})
                class_rows[cls].append((prev_eid, next_eid, float(info.get("psi", np.nan)), bool(info.get("supported", False))))
    rows = []
    for cls, vals in class_rows.items():
        junction_count = len(vals)
        supported = np.asarray([v[3] for v in vals], dtype=bool)
        psi = np.asarray([v[2] for v in vals], dtype=float)
        finite = np.isfinite(psi)
        supported_count = int(supported.sum())
        unsupported_count = int((~supported).sum())
        coverage = supported_count / max(junction_count, 1)
        sup_rate = float(np.nanmean(psi[supported & finite])) if np.any(supported & finite) else np.nan
        unsup_rate = float(np.nanmean(psi[(~supported) & finite])) if np.any((~supported) & finite) else np.nan
        gap = sup_rate - unsup_rate if np.isfinite(sup_rate) and np.isfinite(unsup_rate) else np.nan
        auroc, auprc = _safe_auc(psi[finite], supported[finite].astype(int))
        rows.append(
            {
                "env": env,
                "seed": seed,
                "graph_id": graph_id,
                "junction_class": cls,
                "junction_count": junction_count,
                "supported_count": supported_count,
                "unsupported_count": unsupported_count,
                "coverage": coverage,
                "supported_success_rate": sup_rate,
                "unsupported_success_rate": unsup_rate,
                "supported_gap": gap,
                "psi_AUROC": auroc,
                "psi_AUPRC": auprc,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-artifact-root", default="artifacts/stage25")
    p.add_argument("--oracle-reports-root", default="reports/stage25_oracle_scan_tmp")
    p.add_argument("--out", default="reports/stage25_boundary_coverage.csv")
    p.add_argument("--summary-out", default="reports/stage25_boundary_coverage.md")
    p.add_argument("--min-junctions-per-class", type=int, default=100)
    args = p.parse_args()
    artifact_root = Path(args.oracle_artifact_root)
    reports_root = Path(args.oracle_reports_root)
    summary = _read(reports_root / "stage23_bridge_graph_summary.csv")
    rows = []
    if len(summary):
        for r in summary.itertuples(index=False):
            gid = str(getattr(r, "graph_id", ""))
            if gid == "G0":
                continue
            rows.extend(analyze_one(str(getattr(r, "env")), int(getattr(r, "seed")), gid, artifact_root, reports_root))
    df = pd.DataFrame(rows, columns=COLUMNS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    lines = ["# Stage25 Boundary Coverage", ""]
    if len(df):
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + df.to_csv(index=False).strip() + "\n```")
        best_cov = float(pd.to_numeric(df["coverage"], errors="coerce").fillna(0).max())
        best_gap = float(pd.to_numeric(df["supported_gap"], errors="coerce").fillna(-999).max())
        best_auc = float(pd.to_numeric(df["psi_AUROC"], errors="coerce").fillna(0).max())
        gate = "PASS_BOUNDARY_DIAGNOSTIC" if best_cov >= 0.05 and best_gap >= 0.10 and best_auc >= 0.65 else "HOLD_BOUNDARY_DIAGNOSTIC_ONLY"
    else:
        lines.append("No boundary junction rows were available.")
        gate = "HOLD_BOUNDARY_DIAGNOSTIC_ONLY"
    lines.extend(["", f"Gate: {gate}"])
    Path(args.summary_out).write_text("\n".join(lines) + "\n")
    print(f"[stage25_boundary] rows={len(df)} gate={gate}")


if __name__ == "__main__":
    main()
