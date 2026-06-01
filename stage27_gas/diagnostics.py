from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .graph import GraphData
from .math_utils import bootstrap_mean_ci, wilson_interval


def compute_path_diagnostics(graph: GraphData, path: list[int], path_edge_ids: Optional[list[int]] = None) -> Dict[str, float]:
    if not path:
        return {
            "path_found": 0.0,
            "path_num_nodes": 0.0,
            "path_num_edges": 0.0,
            "path_total_cost": float("inf"),
        }
    if path_edge_ids is None:
        path_edge_ids = graph.path_edge_ids(path) if len(path) > 1 else []
    eids = np.asarray(path_edge_ids, dtype=np.int64)
    out: Dict[str, float] = {
        "path_found": 1.0,
        "path_num_nodes": float(len(path)),
        "path_num_edges": float(len(eids)),
        "path_total_cost": float(np.sum(graph.edge_costs[eids])) if len(eids) else 0.0,
        "path_max_edge_cost": float(np.max(graph.edge_costs[eids])) if len(eids) else 0.0,
        "path_mean_edge_cost": float(np.mean(graph.edge_costs[eids])) if len(eids) else 0.0,
    }
    if len(eids) > 0:
        ef = graph.edge_features
        for key in [
            "norm_d_tdr",
            "norm_d_tmd",
            "d_tdr",
            "d_tmd",
            "metric_disagreement",
            "is_cross_traj",
            "p_exec",
            "exec_uncertainty",
            "longhop_penalty",
            "tmd_shortcut_used",
            "support_score",
        ]:
            if key in ef:
                vals = np.asarray(ef[key])[eids].astype(np.float32)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                out[f"path_{key}_mean"] = float(np.mean(vals))
                out[f"path_{key}_max"] = float(np.max(vals))
                out[f"path_{key}_min"] = float(np.min(vals))
        if "norm_d_tdr" in ef:
            vals = np.asarray(ef["norm_d_tdr"])[eids].astype(np.float32)
            out["largest_hop_ratio"] = float(np.max(vals)) if len(vals) else 0.0
        elif len(eids):
            out["largest_hop_ratio"] = float(np.max(graph.edge_costs[eids]) / max(np.median(graph.edge_costs), 1e-8))
        if "is_cross_traj" in ef:
            out["cross_traj_edge_ratio"] = float(np.mean(np.asarray(ef["is_cross_traj"])[eids]))
    else:
        out["largest_hop_ratio"] = 0.0
        out["cross_traj_edge_ratio"] = 0.0
    return out


def summarize_graph(graph: GraphData) -> dict:
    out = {
        "variant": graph.metadata.get("variant", "unknown"),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "mean_edge_cost": float(np.mean(graph.edge_costs)) if graph.num_edges else float("nan"),
        "median_edge_cost": float(np.median(graph.edge_costs)) if graph.num_edges else float("nan"),
        "max_edge_cost": float(np.max(graph.edge_costs)) if graph.num_edges else float("nan"),
    }
    for key in ["is_cross_traj", "p_exec", "metric_disagreement", "longhop_penalty", "tmd_shortcut_used"]:
        if key in graph.edge_features:
            vals = np.asarray(graph.edge_features[key], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                out[f"edge_{key}_mean"] = float(vals.mean())
                out[f"edge_{key}_p95"] = float(np.percentile(vals, 95))
    return out


def summarize_eval_rows(df: pd.DataFrame, group_cols: list[str] | None = None, success_col: str = "success") -> pd.DataFrame:
    group_cols = group_cols or ["env", "variant"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        n = len(g)
        success_values = pd.to_numeric(g[success_col], errors="coerce").fillna(0.0) if success_col in g else pd.Series(dtype=float)
        successes = int(success_values.sum()) if len(success_values) else 0
        lo, hi = wilson_interval(successes, n)
        row.update({
            "episodes": n,
            "success_mean": successes / n if n else float("nan"),
            "success_wilson95_low": lo,
            "success_wilson95_high": hi,
        })
        for col in [
            "return",
            "path_total_cost",
            "largest_hop_ratio",
            "cross_traj_edge_ratio",
            "subgoal_reach_rate",
            "first_failed_edge_idx",
            "path_p_exec_mean",
            "path_metric_disagreement_mean",
            "steps",
            "no_path_rate",
            "goal_distance_improvement",
            "subgoal_switch_count",
            "final_goal_mode_steps",
            "stage27_path_found",
            "stage27_path_len",
            "stage27_path_cost",
            "stage27_largest_hop_ratio",
            "stage27_cross_traj_edge_ratio",
            "stage27_path_p_exec_mean",
            "stage27_path_metric_disagreement_mean",
            "stage27_path_longhop_penalty_mean",
            "stage27_path_tmd_shortcut_rate",
            "stage27_subgoal_reach_rate",
            "stage27_first_failed_edge_id",
        ]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
                if len(vals):
                    ci_lo, ci_hi = bootstrap_mean_ci(vals)
                    row[f"{col}_mean"] = float(np.mean(vals))
                    row[f"{col}_ci95_low"] = ci_lo
                    row[f"{col}_ci95_high"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compare_to_baseline(summary: pd.DataFrame, baseline_variant: str = "B0_GAS") -> pd.DataFrame:
    if not {"env", "variant", "success_mean"}.issubset(summary.columns):
        return summary
    out = []
    group_cols = ["env"]
    if "run_episodes" in summary.columns:
        group_cols.append("run_episodes")
    for _, g in summary.groupby(group_cols, dropna=False):
        base = g[g["variant"] == baseline_variant]
        if base.empty:
            out.extend(g.to_dict("records"))
            continue
        base_success = float(base.iloc[0]["success_mean"])
        for _, row in g.iterrows():
            r = row.to_dict()
            r["delta_success_vs_baseline_pp"] = 100.0 * (float(row["success_mean"]) - base_success)
            out.append(r)
    return pd.DataFrame(out)


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = ["" if pd.isna(row[c]) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown_report(summary: pd.DataFrame, graph_summary: Optional[pd.DataFrame], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Stage27 GAS Evaluation Report\n")
    lines.append("## Success summary\n")
    if summary.empty:
        lines.append("No eval rows found.\n")
    else:
        cols = [
            c
            for c in [
                "env",
                "run_episodes",
                "variant",
                "episodes",
                "success_mean",
                "success_wilson95_low",
                "success_wilson95_high",
                "delta_success_vs_baseline_pp",
                "stage27_subgoal_reach_rate_mean",
                "stage27_largest_hop_ratio_mean",
                "stage27_cross_traj_edge_ratio_mean",
                "stage27_path_p_exec_mean_mean",
            ]
            if c in summary.columns
        ]
        lines.append(_to_markdown_table(summary[cols]))
        lines.append("\n")
    if graph_summary is not None and not graph_summary.empty:
        lines.append("## Graph summary\n")
        lines.append(_to_markdown_table(graph_summary))
        lines.append("\n")
    # Promotion guardrails.
    if not summary.empty and "delta_success_vs_baseline_pp" in summary.columns:
        lines.append("## Promotion guardrail checklist\n")
        lines.append("- Promote only if target env improves and non-target envs are not materially worse.\n")
        lines.append("- Inspect path diagnostics: largest_hop_ratio, cross_traj_edge_ratio, p_exec, and first_failed_edge_idx.\n")
        lines.append("- Treat TMD-gated variants as shortcut proposals, not a global GAS replacement.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_eval_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            csv_paths = sorted(p.rglob("eval.csv"))
        else:
            csv_paths = [p]
        for q in csv_paths:
            frame = pd.read_csv(q)
            eval_args = {}
            args_path = q.parent / "eval_args.json"
            if args_path.exists():
                try:
                    eval_args = json.loads(args_path.read_text(encoding="utf-8"))
                except Exception:
                    eval_args = {}
            mode = str(eval_args.get("mode", ""))
            if "mode" in frame.columns and len(frame):
                mode = str(frame["mode"].iloc[0] or mode)
            if "variant" not in frame.columns:
                if "stage27_variant" in frame.columns and frame["stage27_variant"].fillna("").astype(str).str.len().max() > 0:
                    frame["variant"] = frame["stage27_variant"].fillna("").replace("", pd.NA).fillna("UNKNOWN_STAGE27")
                elif mode == "gas_graph_policy":
                    frame["variant"] = "GAS_BASE"
                elif mode == "gas_graph_tmd_cost_policy":
                    weight = eval_args.get("tmd_cost_weight", "")
                    run_name = q.parent.name.lower()
                    frame["variant"] = "GNAV_TMD_POSCTRL" if "gnav_tmd_posctrl" in run_name else f"TMD_COST_W{weight}"
                else:
                    frame["variant"] = mode or "UNKNOWN"
            if "run_episodes" not in frame.columns:
                frame["run_episodes"] = str(eval_args.get("episodes", ""))
            frame["eval_path"] = str(q)
            frame["run_name"] = q.parent.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
