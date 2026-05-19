from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stage22r_common import (
    add_path_metrics,
    edge_lookup,
    filter_eval,
    load_edge_scores,
    md_table,
    parse_csv_list,
    parse_edge_ids,
    read_all_eval,
)


def _bucket(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True)


def build_selected_edges(df: pd.DataFrame, artifact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path_df = add_path_metrics(df, artifact_root)
    edge_rows = []
    cache: dict[tuple[str, int], dict[int, dict[str, object]]] = {}
    for row in path_df.itertuples(index=False):
        env = str(row.env)
        seed = int(row.seed)
        key = (env, seed)
        if key not in cache:
            cache[key] = edge_lookup(load_edge_scores(artifact_root, env, seed))
        lookup = cache[key]
        for pos, edge_id in enumerate(parse_edge_ids(getattr(row, "path_edge_ids", ""))):
            e = lookup.get(edge_id, {})
            edge_rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "variant": str(row.variant),
                    "budget": float(row.budget),
                    "fallback_mode": str(row.fallback_mode),
                    "task_id": int(row.task_id),
                    "episode_id": int(row.episode_id),
                    "success": int(row.success),
                    "edge_position": pos,
                    "edge_id": edge_id,
                    "known_edge": int(bool(e)),
                    "p_exec": float(e.get("p_exec", np.nan)) if e else np.nan,
                    "r_exec": float(e.get("r_exec", np.nan)) if e else np.nan,
                    "local_support": float(e.get("local_support", 0.0)) if e else np.nan,
                    "same_traj_support": float(e.get("same_traj_support", 0.0)) if e else np.nan,
                    "edge_source": str(e.get("edge_source", "virtual_or_unknown")) if e else "virtual_or_unknown",
                }
            )
    edge_df = pd.DataFrame(edge_rows)
    if len(edge_df):
        edge_df["p_exec_bucket"] = _bucket(edge_df["p_exec"], [-0.01, 0.5, 0.8, 0.95, 0.99, 1.0], ["<=0.5", "0.5-0.8", "0.8-0.95", "0.95-0.99", "0.99-1"])
        edge_df["r_exec_bucket"] = _bucket(edge_df["r_exec"], [-0.01, 0.05, 0.15, 0.3, 1, 999], ["0-0.05", "0.05-0.15", "0.15-0.3", "0.3-1", ">1"])
        edge_df["local_support_bucket"] = np.where(edge_df["local_support"].fillna(0) > 0, "supported", "unsupported")
    overlap_rows = []
    keys = ["env", "seed", "budget", "fallback_mode", "task_id", "episode_id"]
    shortest = path_df[path_df["variant"].astype(str) == "gas_shortest"].copy()
    shortest_map = {
        tuple(getattr(r, k) for k in keys): set(parse_edge_ids(getattr(r, "path_edge_ids", "")))
        for r in shortest.itertuples(index=False)
    }
    for row in path_df[path_df["variant"].astype(str) != "gas_shortest"].itertuples(index=False):
        key = tuple(getattr(row, k) for k in keys)
        base = shortest_map.get(key)
        if base is None:
            continue
        cur = set(parse_edge_ids(getattr(row, "path_edge_ids", "")))
        union = base | cur
        inter = base & cur
        avoided = base - cur
        new = cur - base
        overlap_rows.append(
            {
                "env": row.env,
                "seed": int(row.seed),
                "variant": row.variant,
                "budget": float(row.budget),
                "fallback_mode": row.fallback_mode,
                "task_id": int(row.task_id),
                "episode_id": int(row.episode_id),
                "success": int(row.success),
                "shortest_edges": len(base),
                "variant_edges": len(cur),
                "edge_overlap_jaccard": len(inter) / max(len(union), 1),
                "edge_overlap_vs_shortest": len(inter) / max(len(base), 1),
                "avoided_edges": len(avoided),
                "new_edges": len(new),
                "avoided_edge_ids": "|".join(map(str, sorted(avoided))),
                "new_edge_ids": "|".join(map(str, sorted(new))),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    return edge_df, path_df, overlap_df


def write_outputs(out: Path, edge_df: pd.DataFrame, path_df: pd.DataFrame, overlap_df: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(out / "stage22r_selected_edge_diagnostics.csv", index=False)
    overlap_df.to_csv(out / "stage22r_path_edge_overlap.csv", index=False)
    summaries = []
    if len(edge_df):
        summaries.append(
            edge_df.groupby(["env", "seed", "variant", "budget", "fallback_mode", "local_support_bucket"], dropna=False)
            .agg(
                selected_edges=("edge_id", "count"),
                episode_success=("success", "mean"),
                p_exec_median=("p_exec", "median"),
                r_exec_median=("r_exec", "median"),
                local_support_rate=("local_support", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean())),
            )
            .reset_index()
        )
    edge_summary = summaries[0] if summaries else pd.DataFrame()
    edge_summary.to_csv(out / "stage22r_selected_edge_summary.csv", index=False)
    lines = ["# Stage22R Selected Edge Diagnostics", ""]
    lines.append("## Selected Edge Summary")
    lines.append(md_table(edge_summary, max_rows=80))
    lines.append("")
    lines.append("## Path Edge Overlap")
    if len(overlap_df):
        ov = (
            overlap_df.groupby(["env", "seed", "variant", "budget", "fallback_mode"], dropna=False)
            .agg(
                episodes=("success", "count"),
                success=("success", "mean"),
                edge_overlap_vs_shortest=("edge_overlap_vs_shortest", "mean"),
                edge_overlap_jaccard=("edge_overlap_jaccard", "mean"),
                avoided_edges=("avoided_edges", "mean"),
                new_edges=("new_edges", "mean"),
            )
            .reset_index()
        )
        lines.append(md_table(ov))
        if (ov["edge_overlap_vs_shortest"] > 0.9).any():
            lines.append("")
            lines.append("- REPAIR_SCORING signal: at least one reachability/boundary setting overlaps GAS shortest by >90%.")
    else:
        lines.append("_No paired overlap rows._")
    lines.append("")
    lines.append("## Path Risk Summary")
    cols = [c for c in ["env", "seed", "variant", "budget", "fallback_mode", "path_edges", "local_support_rate", "unsupported_edge_rate", "success"] if c in path_df.columns]
    if cols:
        ps = path_df.groupby(["env", "seed", "variant", "budget", "fallback_mode"], dropna=False).agg(
            episodes=("success", "count"),
            success=("success", "mean"),
            path_edges=("path_edges", "mean"),
            local_support_rate=("local_support_rate", "mean"),
            unsupported_edge_rate=("unsupported_edge_rate", "mean"),
        ).reset_index()
        lines.append(md_table(ps))
    path = out / "stage22r_selected_edge_diagnostics.md"
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--artifact-root", default="artifacts/stage22")
    p.add_argument("--eval-root", default="runs_stage22_eval")
    p.add_argument("--out", default="reports")
    args = p.parse_args(argv)
    envs = parse_csv_list(args.envs)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    df = filter_eval(read_all_eval(args.eval_root), envs, seeds)
    edge_df, path_df, overlap_df = build_selected_edges(df, Path(args.artifact_root))
    write_outputs(Path(args.out), edge_df, path_df, overlap_df)
    print(json.dumps({"edge_rows": int(len(edge_df)), "overlap_rows": int(len(overlap_df)), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
