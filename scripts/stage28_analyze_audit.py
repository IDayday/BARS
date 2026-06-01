#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_LABEL_TO_DIRECTION = {
    "NO_DATA_PATH_AFTER_NODE_PROJECTION": "Dataset/support limitation: evaluate model-based bridge generation or denser support graph before planner changes.",
    "BASE_LOST_SUPPORTED_PATH_OR_EDGE_PRUNING": "Bridge-preserving abstraction: preserve endpoints/bottlenecks and avoid pruning support-critical edges.",
    "BASE_USES_CROSS_TRAJ_SHORTCUT_FOR_SUPPORTED_PAIR": "Conservative connectedness: distinguish true bridges from cross-trajectory optimistic shortcuts.",
    "BASE_SINGLE_HOP_DOMINATED_PATH": "Path robustness: validate long hops and use k-diverse path ensemble/recovery instead of single shortest path.",
    "SINGLE_PATH_FRAGILITY_PROXY": "Path ensemble: construct alternative routes and execution-time path switching.",
    "GRAPH_PATH_EXISTS_EXECUTION_PROBE_NEEDED": "Execution diagnosis: run local edge rollouts/failure monitor before changing graph construction.",
    "NO_GRAPH_PATH_UNRESOLVED": "Connectivity audit: compare dense/projection graphs and inspect missing components.",
}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def collect(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for root in paths:
        if root.is_file():
            df = _read_csv(root)
            if len(df):
                df["audit_path"] = str(root)
                frames.append(df)
            continue
        if not root.exists():
            continue
        for p in root.rglob("stage28_graph_audit.csv"):
            df = _read_csv(p)
            if len(df):
                df["audit_path"] = str(p)
                frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _mean_ci(x: pd.Series) -> tuple[float, float, float, int]:
    vals = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    n = int(len(vals))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(vals.mean())
    if n == 1:
        return mean, float("nan"), float("nan"), n
    se = float(vals.std(ddof=1) / np.sqrt(n))
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def summarize_graphs(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df:
        return pd.DataFrame()
    g = df[df["phase"].astype(str).eq("stage28_graph_summary")].copy()
    if len(g) == 0:
        return pd.DataFrame()
    numeric_cols = [
        "num_nodes", "num_edges", "mean_out_degree", "weak_components", "strong_components",
        "largest_weak_component_rate", "largest_strong_component_rate", "node_cover_emb_dist_p50",
        "node_cover_emb_dist_p90", "endpoint_exact_retention_rate", "cross_edge_rate",
        "temporal_supported_edge_rate", "low_cost_high_conf_cross_edge_rate", "edge_p_exec_mean",
    ]
    keys = [c for c in ["env", "graph_id"] if c in g]
    rows: list[dict[str, Any]] = []
    for key, sub in g.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row["runs"] = int(len(sub))
        for col in numeric_cols:
            if col not in sub:
                continue
            mean, lo, hi, n = _mean_ci(sub[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
            row[f"{col}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_paths(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df:
        return pd.DataFrame()
    p = df[df["phase"].astype(str).eq("stage28_path_probe")].copy()
    if len(p) == 0:
        return pd.DataFrame()
    p["found_num"] = _num(p, "found")
    for col in ["num_edges", "objective", "path_cross_edge_rate", "path_largest_edge_cost_ratio", "alt_path_rate"]:
        if col not in p:
            p[col] = np.nan
    keys = [c for c in ["env", "graph_id", "planner_variant", "pair_type"] if c in p]
    agg = p.groupby(keys, dropna=False).agg(
        rows=("found_num", "count"),
        path_found_rate=("found_num", "mean"),
        num_edges_mean=("num_edges", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        objective_mean=("objective", lambda x: pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
        path_cross_edge_rate_mean=("path_cross_edge_rate", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        path_largest_edge_cost_ratio_mean=("path_largest_edge_cost_ratio", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        alt_path_rate_mean=("alt_path_rate", lambda x: pd.to_numeric(x, errors="coerce").mean()),
    ).reset_index()
    return agg


def summarize_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df:
        return pd.DataFrame()
    t = df[df["phase"].astype(str).eq("stage28_failure_taxonomy_proxy")].copy()
    if len(t) == 0 or "failure_label" not in t:
        return pd.DataFrame()
    keys = [c for c in ["env", "seed"] if c in t]
    rows = []
    for key, sub in t.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(keys, key))
        total = max(1, len(sub))
        for label, cnt in sub["failure_label"].astype(str).value_counts().items():
            rows.append({**base, "failure_label": label, "count": int(cnt), "rate": float(cnt / total), "num_pairs": int(total), "direction": KEY_LABEL_TO_DIRECTION.get(label, "Inspect manually.")})
    return pd.DataFrame(rows)


def build_recommendations(tax: pd.DataFrame, graph_summary: pd.DataFrame, path_summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Stage28 Graph Method Audit Recommendations")
    lines.append("")
    lines.append("This report converts diagnostic evidence into research directions. It should be read before implementing new graph algorithms.")
    lines.append("")
    lines.append("## Evidence contract")
    lines.append("")
    lines.append("All Stage28 audit CSV rows carry `gate`, `evidence_class`, and `report_file` fields. The recommendations below are derived from `PASS_STAGE28_DIAGNOSE_FIRST_TAXONOMY` rows, with graph and path context from `PASS_STAGE28_GRAPH_COUNTERFACTUALS` and `PASS_STAGE28_PATH_PROBE` rows.")
    lines.append("")
    if len(tax) == 0:
        lines.append("No taxonomy rows were found. Run `scripts/stage28_graph_audit.py` first.")
        return "\n".join(lines) + "\n"
    for env, sub in tax.groupby("env", dropna=False):
        lines.append(f"## {env}")
        lines.append("")
        agg = sub.groupby("failure_label", dropna=False).agg(rate=("rate", "mean"), count=("count", "sum"), num_pairs=("num_pairs", "sum"), direction=("direction", "first")).reset_index()
        agg = agg.sort_values(["rate", "count"], ascending=[False, False])
        dominant = agg.iloc[0]
        lines.append(f"Dominant proxy: `{dominant['failure_label']}` at mean rate {float(dominant['rate']):.3f}.")
        lines.append(f"Recommended next algorithm family: {dominant['direction']}")
        lines.append("")
        lines.append("| failure label | mean rate | count | direction |")
        lines.append("|---|---:|---:|---|")
        for _, r in agg.iterrows():
            lines.append(f"| `{r['failure_label']}` | {float(r['rate']):.3f} | {int(r['count'])} | {r['direction']} |")
        lines.append("")
        if len(graph_summary) and "env" in graph_summary:
            gs = graph_summary[graph_summary["env"].astype(str).eq(str(env))]
            if len(gs):
                lines.append("Graph-level signals:")
                for _, r in gs.iterrows():
                    graph_id = r.get("graph_id", "unknown")
                    lcc = r.get("largest_weak_component_rate_mean", np.nan)
                    cross = r.get("cross_edge_rate_mean", np.nan)
                    cover = r.get("node_cover_emb_dist_p90_mean", np.nan)
                    lines.append(f"- `{graph_id}`: LCC={float(lcc):.3f} cross_edge={float(cross):.3f} cover_p90={float(cover):.3f}")
                lines.append("")
        if len(path_summary) and "env" in path_summary:
            ps = path_summary[path_summary["env"].astype(str).eq(str(env))]
            base = ps[ps["graph_id"].astype(str).eq("base_cached")] if "graph_id" in ps else pd.DataFrame()
            if len(base):
                lines.append("Base path probes:")
                for _, r in base.iterrows():
                    lines.append(f"- `{r.get('planner_variant','')}` / `{r.get('pair_type','')}`: found={float(r.get('path_found_rate', np.nan)):.3f}, edges={float(r.get('num_edges_mean', np.nan)):.2f}, cross={float(r.get('path_cross_edge_rate_mean', np.nan)):.3f}")
                lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage28 graph-method audit CSVs.")
    parser.add_argument("--roots", nargs="+", required=True, help="Audit CSVs or directories containing stage28_graph_audit.csv")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect([Path(x) for x in args.roots])
    df.to_csv(out_dir / "stage28_audit_all.csv", index=False)
    graph_summary = summarize_graphs(df)
    path_summary = summarize_paths(df)
    taxonomy = summarize_taxonomy(df)
    graph_summary.to_csv(out_dir / "stage28_graph_summary.csv", index=False)
    path_summary.to_csv(out_dir / "stage28_path_summary.csv", index=False)
    taxonomy.to_csv(out_dir / "stage28_failure_taxonomy.csv", index=False)
    (out_dir / "stage28_recommendations.md").write_text(build_recommendations(taxonomy, graph_summary, path_summary), encoding="utf-8")
    print(str(out_dir))


if __name__ == "__main__":
    main()
