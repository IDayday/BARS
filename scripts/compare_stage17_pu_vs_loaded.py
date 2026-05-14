#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return pd.DataFrame()


def mean_table(df: pd.DataFrame, phase: str, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty or "phase" not in df.columns:
        return pd.DataFrame()
    x = df[df["phase"].eq(phase)].copy()
    if x.empty:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in x.columns]
    metric_cols = [c for c in metric_cols if c in x.columns]
    if not group_cols or not metric_cols:
        return pd.DataFrame()
    return x.groupby(group_cols)[metric_cols].mean(numeric_only=True).reset_index()


def add_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.insert(0, "condition", label)
    return df


def safe_concat(items: list[pd.DataFrame]) -> pd.DataFrame:
    items = [x for x in items if x is not None and not x.empty]
    if not items:
        return pd.DataFrame()
    return pd.concat(items, ignore_index=True, sort=False)


def write_md_table(lines: list[str], title: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    lines.append(f"\n## {title}\n")
    if df.empty:
        lines.append("_No data._\n")
        return
    lines.append(df.head(max_rows).to_markdown(index=False))
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loaded-root", default="runs_stage16_full12")
    ap.add_argument("--pu-root", default="runs_stage17_pu_retrain4")
    ap.add_argument("--out", default="reports/stage17_pu_vs_loaded_compare.md")
    ap.add_argument("--csv-out-dir", default="reports/stage17_compare_csv")
    args = ap.parse_args()

    loaded = Path(args.loaded_root) / "_analysis"
    pu = Path(args.pu_root) / "_analysis"
    out = Path(args.out)
    csv_out = Path(args.csv_out_dir)
    csv_out.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    d_loaded = read_csv(loaded / "diagnostics_all.csv")
    d_pu = read_csv(pu / "diagnostics_all.csv")
    s_loaded = read_csv(loaded / "summary_all.csv")
    s_pu = read_csv(pu / "summary_all.csv")
    t_loaded = read_csv(loaded / "train_all.csv")
    t_pu = read_csv(pu / "train_all.csv")
    p_loaded = read_csv(loaded / "profile_all.csv")
    p_pu = read_csv(pu / "profile_all.csv")
    g_loaded = read_csv(loaded / "graph_all.csv")
    g_pu = read_csv(pu / "graph_all.csv")

    # Completion.
    def status_counts(s: pd.DataFrame, label: str) -> pd.DataFrame:
        if s.empty or "status" not in s.columns:
            return pd.DataFrame()
        group = [c for c in ["env", "status"] if c in s.columns]
        z = s.groupby(group).size().reset_index(name="count")
        z.insert(0, "condition", label)
        return z

    status = safe_concat([
        status_counts(s_loaded, "loaded_baseline"),
        status_counts(s_pu, "pu_retrain"),
    ])

    # Balanced edge diagnostic.
    bal_cols = [
        "edge_auc_balanced",
        "edge_auprc_balanced",
        "supported_edge_rate",
        "selected_supported_rate",
        "selected_hard_neg_proxy_rate",
        "selected_unlabeled_bridge_rate",
        "score_supported_mean",
        "score_hard_neg_proxy_mean",
        "score_unlabeled_bridge_mean",
    ]
    bal = safe_concat([
        add_label(mean_table(d_loaded, "balanced_edge_diag", ["env"], bal_cols), "loaded_baseline"),
        add_label(mean_table(d_pu, "balanced_edge_diag", ["env"], bal_cols), "pu_retrain"),
    ])
    if not bal.empty:
        bal["supported_minus_hard_selected"] = bal.get("selected_supported_rate", np.nan) - bal.get("selected_hard_neg_proxy_rate", np.nan)
        bal["supported_minus_hard_score"] = bal.get("score_supported_mean", np.nan) - bal.get("score_hard_neg_proxy_mean", np.nan)
        bal.to_csv(csv_out / "balanced_edge_compare.csv", index=False)

    # Edge rollout diagnostic.
    rollout_cols = [
        "edge_rollout_auc",
        "edge_rollout_auprc",
        "success_rate",
        "selected_edge_success_rate",
        "unselected_edge_success_rate",
        "success_rate_selected_supported",
        "success_rate_selected_hard_neg_proxy",
        "success_rate_unselected_supported",
        "success_rate_unselected_hard_neg_proxy",
        "p_exec_mean_selected_supported",
        "p_exec_mean_selected_hard_neg_proxy",
        "final_dist_mean_selected_supported",
        "final_dist_mean_selected_hard_neg_proxy",
        "reset_available",
        "reset_ok_count",
        "reset_unavailable_count",
        "num_edges_eval",
    ]
    roll = safe_concat([
        add_label(mean_table(d_loaded, "edge_rollout_diag", ["env"], rollout_cols), "loaded_baseline"),
        add_label(mean_table(d_pu, "edge_rollout_diag", ["env"], rollout_cols), "pu_retrain"),
    ])
    if not roll.empty:
        roll["selected_minus_unselected_success"] = roll.get("selected_edge_success_rate", np.nan) - roll.get("unselected_edge_success_rate", np.nan)
        roll["selected_supported_minus_hard_success"] = roll.get("success_rate_selected_supported", np.nan) - roll.get("success_rate_selected_hard_neg_proxy", np.nan)
        roll.to_csv(csv_out / "edge_rollout_compare.csv", index=False)

    # Path diagnostic.
    path_cols = [
        "found",
        "total_risk",
        "total_boundary",
        "total_cost",
        "objective",
        "num_edges",
        "num_subgoals",
        "trivial_pair_rate",
    ]
    path = safe_concat([
        add_label(mean_table(d_loaded, "path_diag", ["env", "variant", "lambda_risk"], path_cols), "loaded_baseline"),
        add_label(mean_table(d_pu, "path_diag", ["env", "variant", "lambda_risk"], path_cols), "pu_retrain"),
    ])
    if not path.empty:
        path.to_csv(csv_out / "path_compare.csv", index=False)

    # Edge exact proxy.
    edge_cols = [
        "reach_auc_proxy",
        "reach_auprc_proxy",
        "cross_traj_selected_rate",
        "reachable_edge_coverage_proxy",
        "selected_edges",
        "num_edges",
    ]
    edge = safe_concat([
        add_label(mean_table(d_loaded, "edge_diag", ["env"], edge_cols), "loaded_baseline"),
        add_label(mean_table(d_pu, "edge_diag", ["env"], edge_cols), "pu_retrain"),
    ])
    if not edge.empty:
        edge.to_csv(csv_out / "edge_proxy_compare.csv", index=False)

    # Training evidence.
    def train_summary(t: pd.DataFrame, label: str) -> pd.DataFrame:
        if t.empty:
            return pd.DataFrame()
        keep = [c for c in ["env", "seed", "module", "event", "phase", "step", "epoch", "loss", "message"] if c in t.columns]
        z = t[keep].copy() if keep else t.copy()
        z.insert(0, "condition", label)
        return z

    train = safe_concat([
        train_summary(t_loaded, "loaded_baseline"),
        train_summary(t_pu, "pu_retrain"),
    ])
    if not train.empty:
        train.to_csv(csv_out / "train_evidence.csv", index=False)

    # Profile summary.
    def profile_summary(p: pd.DataFrame, label: str) -> pd.DataFrame:
        if p.empty:
            return pd.DataFrame()
        dur_cols = [c for c in ["duration_sec", "elapsed_sec", "seconds", "duration"] if c in p.columns]
        if not dur_cols:
            return pd.DataFrame()
        dur = dur_cols[0]
        group = [c for c in ["env", "phase", "event"] if c in p.columns]
        if not group:
            return pd.DataFrame()
        z = p.groupby(group)[dur].mean(numeric_only=True).reset_index()
        z.insert(0, "condition", label)
        return z

    prof = safe_concat([
        profile_summary(p_loaded, "loaded_baseline"),
        profile_summary(p_pu, "pu_retrain"),
    ])
    if not prof.empty:
        prof.to_csv(csv_out / "profile_compare.csv", index=False)

    # Graph summary.
    graph_cols = ["num_nodes", "num_edges", "mean_out_degree", "p_exec_mean", "risk_mean", "cost_mean", "duration_sec", "spectral_seconds"]
    graph = safe_concat([
        add_label(mean_table(g_loaded.rename(columns={"event": "phase"}) if "event" in g_loaded.columns and "phase" not in g_loaded.columns else g_loaded, "completed", ["env"], graph_cols), "loaded_baseline"),
        add_label(mean_table(g_pu.rename(columns={"event": "phase"}) if "event" in g_pu.columns and "phase" not in g_pu.columns else g_pu, "completed", ["env"], graph_cols), "pu_retrain"),
    ])
    if not graph.empty:
        graph.to_csv(csv_out / "graph_compare.csv", index=False)

    lines = []
    lines.append("# Stage17 PU Retrain vs Loaded Baseline Comparison\n")
    lines.append(f"- loaded_root: `{args.loaded_root}`")
    lines.append(f"- pu_root: `{args.pu_root}`")
    lines.append("")
    write_md_table(lines, "Run Status", status)
    write_md_table(lines, "Balanced Edge Diagnostics", bal)
    write_md_table(lines, "Edge Rollout Diagnostics", roll)
    write_md_table(lines, "Path Diagnostics", path)
    write_md_table(lines, "Edge Proxy Diagnostics", edge)
    write_md_table(lines, "Graph Summary", graph)
    write_md_table(lines, "Profile Summary", prof.head(120) if not prof.empty else prof)
    lines.append("\n## Training Evidence\n")
    if train.empty:
        lines.append("_No train data._\n")
    else:
        # Show rows likely related to reachability.
        if "module" in train.columns:
            rt = train[train["module"].astype(str).str.contains("reach", case=False, na=False)]
        elif "message" in train.columns:
            rt = train[train["message"].astype(str).str.contains("reach", case=False, na=False)]
        else:
            rt = train
        lines.append(rt.head(120).to_markdown(index=False))
        lines.append("")

    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print(f"Wrote CSVs under {csv_out}")


if __name__ == "__main__":
    main()
