#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_LABELS = [
    "BASE_USES_CROSS_TRAJ_SHORTCUT_FOR_SUPPORTED_PAIR",
    "BASE_LOST_SUPPORTED_PATH_OR_EDGE_PRUNING",
    "GRAPH_PATH_EXISTS_EXECUTION_PROBE_NEEDED",
    "NO_GRAPH_PATH_UNRESOLVED",
]


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
        for p in root.rglob("stage29_support_calibrated_audit.csv"):
            df = _read_csv(p)
            if len(df):
                df["audit_path"] = str(p)
                frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _mean_ci(x: pd.Series) -> tuple[float, float, float, int]:
    vals = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    n = int(len(vals))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(vals.mean())
    if n == 1:
        return mean, float("nan"), float("nan"), n
    se = float(vals.std(ddof=1) / np.sqrt(n))
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def _summarize_numeric(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if len(df) == 0:
        return pd.DataFrame()
    for key, sub in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row["runs"] = int(len(sub))
        for col in cols:
            if col not in sub:
                continue
            mean, lo, hi, n = _mean_ci(sub[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
            row[f"{col}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_graphs(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df:
        return pd.DataFrame()
    g = df[df["phase"].astype(str).eq("stage29_graph_summary")].copy()
    cols = [
        "num_nodes",
        "num_edges",
        "protected_nodes",
        "endpoint_exact_retention_rate",
        "cross_edge_rate",
        "temporal_supported_edge_rate",
        "unsupported_shortcut_edge_rate",
        "support_count_positive_rate",
        "support_score_mean",
        "support_score_p50",
        "support_score_p90",
        "support_risk_mean",
    ]
    return _summarize_numeric(g, [c for c in ["env", "graph_id"] if c in g], cols)


def summarize_paths(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df:
        return pd.DataFrame()
    p = df[df["phase"].astype(str).eq("stage29_path_probe")].copy()
    if len(p) == 0:
        return pd.DataFrame()
    p["found_num"] = _num(p, "found", 0.0)
    cols = [
        "found_num",
        "num_edges",
        "objective",
        "path_cross_edge_rate",
        "path_largest_edge_cost_ratio",
        "unsupported_edges",
        "support_risk",
        "cross_support_risk",
        "path_unsupported_shortcut_rate",
        "path_min_support_score",
        "path_mean_support_score",
    ]
    out = _summarize_numeric(p, [c for c in ["env", "graph_id", "planner_id", "pair_type"] if c in p], cols)
    if len(out) and "found_num_mean" in out:
        out = out.rename(columns={"found_num_mean": "path_found_rate"})
    return out


def summarize_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df:
        return pd.DataFrame()
    t = df[df["phase"].astype(str).eq("stage29_path_probe")].copy()
    if len(t) == 0 or "failure_label" not in t:
        return pd.DataFrame()
    keys = [c for c in ["env", "seed", "graph_id", "planner_id"] if c in t]
    rows = []
    for key, sub in t.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(keys, key))
        total = max(1, len(sub))
        counts = sub["failure_label"].astype(str).value_counts()
        for label, cnt in counts.items():
            rows.append({**base, "failure_label": label, "count": int(cnt), "rate": float(cnt / total), "num_pairs": int(total)})
    return pd.DataFrame(rows)


def summarize_deltas(tax: pd.DataFrame, default_planner: str) -> pd.DataFrame:
    if len(tax) == 0:
        return pd.DataFrame()
    rows = []
    for env, sub in tax.groupby("env", dropna=False):
        base = sub[sub["graph_id"].astype(str).eq("base_cached")]
        stage = sub[sub["planner_id"].astype(str).eq(default_planner)]
        for label in sorted(set(TARGET_LABELS) | set(sub["failure_label"].astype(str))):
            b = base[base["failure_label"].astype(str).eq(label)]["rate"].mean()
            s = stage[stage["failure_label"].astype(str).eq(label)]["rate"].mean()
            b = 0.0 if pd.isna(b) else float(b)
            s = 0.0 if pd.isna(s) else float(s)
            rows.append({"env": env, "planner_id": default_planner, "failure_label": label, "base_rate": b, "stage29_rate": s, "delta": s - b})
    base_all = tax[tax["graph_id"].astype(str).eq("base_cached")]
    stage_all = tax[tax["planner_id"].astype(str).eq(default_planner)]
    for label in sorted(set(TARGET_LABELS) | set(tax["failure_label"].astype(str))):
        b_total = base_all.groupby(["env", "seed", "planner_id"], dropna=False)["num_pairs"].first().sum()
        s_total = stage_all.groupby(["env", "seed", "planner_id"], dropna=False)["num_pairs"].first().sum()
        b_rate = float(base_all[base_all["failure_label"].astype(str).eq(label)]["count"].sum() / max(1, b_total))
        s_rate = float(stage_all[stage_all["failure_label"].astype(str).eq(label)]["count"].sum() / max(1, s_total))
        rows.append({"env": "__overall__", "planner_id": default_planner, "failure_label": label, "base_rate": b_rate, "stage29_rate": s_rate, "delta": s_rate - b_rate})
    return pd.DataFrame(rows)


def build_report(graphs: pd.DataFrame, paths: pd.DataFrame, tax: pd.DataFrame, deltas: pd.DataFrame, default_planner: str) -> str:
    lines: list[str] = []
    lines.append("# Stage29 Support-Calibrated Graph Stitching Audit")
    lines.append("")
    lines.append("This report is offline evidence only. It evaluates support-calibrated graph semantics before any online evaluation.")
    lines.append("")
    overall = deltas[deltas["env"].astype(str).eq("__overall__")] if len(deltas) else pd.DataFrame()
    if len(overall):
        lines.append("## Overall Taxonomy Delta")
        lines.append("")
        lines.append("| label | base | stage29 | delta |")
        lines.append("|---|---:|---:|---:|")
        for _, r in overall.sort_values("base_rate", ascending=False).iterrows():
            lines.append(f"| `{r['failure_label']}` | {float(r['base_rate']):.3f} | {float(r['stage29_rate']):.3f} | {float(r['delta']):+.3f} |")
        lines.append("")
    if len(paths):
        lines.append("## Target Path Metrics")
        lines.append("")
        lines.append("| env | planner | pair | found | path_cross | unsupported_edges | support_risk |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        keep = paths[(paths["graph_id"].astype(str).eq("base_cached")) | (paths["planner_id"].astype(str).eq(default_planner))]
        for _, r in keep.sort_values(["env", "planner_id", "pair_type"]).iterrows():
            lines.append(
                f"| {r.get('env','')} | `{r.get('planner_id','')}` | `{r.get('pair_type','')}` | "
                f"{float(r.get('path_found_rate', np.nan)):.3f} | {float(r.get('path_cross_edge_rate_mean', np.nan)):.3f} | "
                f"{float(r.get('unsupported_edges_mean', np.nan)):.2f} | {float(r.get('support_risk_mean', np.nan)):.2f} |"
            )
        lines.append("")
    if len(graphs):
        lines.append("## Graph Metrics")
        lines.append("")
        lines.append("| env | nodes | edges | cross_edge | temporal_supported | unsupported_shortcut | endpoint_retention |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in graphs.sort_values("env").iterrows():
            lines.append(
                f"| {r.get('env','')} | {float(r.get('num_nodes_mean', np.nan)):.0f} | {float(r.get('num_edges_mean', np.nan)):.0f} | "
                f"{float(r.get('cross_edge_rate_mean', np.nan)):.3f} | {float(r.get('temporal_supported_edge_rate_mean', np.nan)):.3f} | "
                f"{float(r.get('unsupported_shortcut_edge_rate_mean', np.nan)):.3f} | {float(r.get('endpoint_exact_retention_rate_mean', np.nan)):.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage29 support-calibrated graph audit CSVs.")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--default-planner-id", default="stage29_lexicographic")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect([Path(x) for x in args.roots])
    df.to_csv(out_dir / "stage29_audit_all.csv", index=False)
    graphs = summarize_graphs(df)
    paths = summarize_paths(df)
    tax = summarize_taxonomy(df)
    deltas = summarize_deltas(tax, args.default_planner_id)
    graphs.to_csv(out_dir / "stage29_graph_summary.csv", index=False)
    paths.to_csv(out_dir / "stage29_path_summary.csv", index=False)
    tax.to_csv(out_dir / "stage29_failure_taxonomy.csv", index=False)
    deltas.to_csv(out_dir / "stage29_failure_taxonomy_delta.csv", index=False)
    (out_dir / "stage29_report.md").write_text(build_report(graphs, paths, tax, deltas, args.default_planner_id), encoding="utf-8")
    print(str(out_dir))


if __name__ == "__main__":
    main()
