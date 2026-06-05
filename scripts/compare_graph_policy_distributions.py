#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cage_gp0_common import (
    group_rows,
    iter_jsonl,
    js_divergence,
    nearest_pair_distance,
    pair_arrays,
    summarize_numeric,
    write_csv,
    write_json,
    write_jsonl,
    write_markdown_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare graph-planned q_G pairs against policy-training q_train pairs.")
    parser.add_argument("--graph_jsonl", nargs="+", required=True)
    parser.add_argument("--train_jsonl", nargs="+", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_pair_support_jsonl", default=None)
    parser.add_argument("--max_graph_pairs_per_env", type=int, default=10000)
    parser.add_argument("--max_train_pairs_per_env", type=int, default=30000)
    parser.add_argument("--support_tau", type=float, default=None)
    parser.add_argument("--coverage_radius", type=float, default=None)
    parser.add_argument("--hist_bins", type=int, default=30)
    parser.add_argument("--audit_seed", type=int, default=0)
    return parser.parse_args()


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(iter_jsonl(path))
    return rows


def deterministic_subsample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(rows), size=limit, replace=False)
    return [rows[int(i)] for i in idxs]


def support_for_env(
    env_name: str,
    graph_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    graph_rows = deterministic_subsample(graph_rows, args.max_graph_pairs_per_env, args.audit_seed + len(env_name))
    train_rows = deterministic_subsample(train_rows, args.max_train_pairs_per_env, args.audit_seed + len(env_name) + 17)
    qg_s, qg_g, graph_valid = pair_arrays(graph_rows)
    qt_s, qt_g, _ = pair_arrays(train_rows)
    if len(qg_s) == 0 or len(qt_s) == 0:
        support_rows: list[dict[str, Any]] = []
        summary = {
            "env_name": env_name,
            "num_graph_pairs": len(graph_rows),
            "num_train_pairs": len(train_rows),
            "num_graph_pairs_with_vectors": len(qg_s),
            "num_train_pairs_with_vectors": len(qt_s),
            "support_available": False,
        }
        return summary, [], support_rows

    nn_dist = nearest_pair_distance(qg_s, qg_g, qt_s, qt_g)
    train_d = np.asarray([float(r.get("d_phi", 0.0)) for r in train_rows if r.get("d_phi") is not None], dtype=np.float64)
    tau = float(args.support_tau) if args.support_tau is not None else float(np.median(train_d) if len(train_d) else 1.0)
    tau = max(tau, 1e-6)
    coverage_radius = float(args.coverage_radius) if args.coverage_radius is not None else tau
    support_score = np.exp(-nn_dist / tau)
    covered = nn_dist <= coverage_radius

    support_rows = []
    for local_i, row_idx in enumerate(graph_valid):
        row = dict(graph_rows[row_idx])
        row.pop("s_ref", None)
        row.pop("g_ref", None)
        row["nn_pair_distance_to_q_train"] = float(nn_dist[local_i])
        row["q_train_support_score"] = float(support_score[local_i])
        row["q_train_covered"] = bool(covered[local_i])
        row["support_tau"] = tau
        row["coverage_radius"] = coverage_radius
        support_rows.append(row)

    graph_temporal_available = any("temporal_gap" in r for r in graph_rows)
    summary = {
        "env_name": env_name,
        "num_graph_pairs": len(graph_rows),
        "num_train_pairs": len(train_rows),
        "num_graph_pairs_with_vectors": len(qg_s),
        "num_train_pairs_with_vectors": len(qt_s),
        "support_available": True,
        "support_tau": tau,
        "coverage_radius": coverage_radius,
        "d_phi_js_divergence": js_divergence([r.get("d_phi") for r in graph_rows], [r.get("d_phi") for r in train_rows], bins=args.hist_bins),
        "temporal_gap_js_divergence": js_divergence([r.get("temporal_gap") for r in graph_rows], [r.get("temporal_gap") for r in train_rows], bins=args.hist_bins) if graph_temporal_available else None,
        "graph_temporal_gap_available": graph_temporal_available,
        "graph_d_phi_mean": summarize_numeric(r.get("d_phi") for r in graph_rows)["mean"],
        "train_d_phi_mean": summarize_numeric(r.get("d_phi") for r in train_rows)["mean"],
        "train_temporal_gap_mean": summarize_numeric(r.get("temporal_gap") for r in train_rows)["mean"],
        "nn_pair_distance_mean": float(np.mean(nn_dist)),
        "nn_pair_distance_p90": float(np.quantile(nn_dist, 0.9)),
        "support_score_mean": float(np.mean(support_score)),
        "support_score_p10": float(np.quantile(support_score, 0.1)),
        "coverage_rate": float(np.mean(covered)),
        "final_phase_coverage_rate": coverage_rate_for(support_rows, lambda r: bool(r.get("final_phase"))),
        "recovery_target_coverage_rate": coverage_rate_for(support_rows, lambda r: "recovery" in str(r.get("pair_role", "")).lower() or bool(r.get("recovery_target", False))),
    }

    bucket_rows = []
    for (bucket,), records in group_rows(support_rows, ["path_position_bucket"]).items():
        bucket_rows.append(
            {
                "env_name": env_name,
                "path_position_bucket": bucket,
                "num_pairs": len(records),
                "coverage_rate": mean_bool(r.get("q_train_covered") for r in records),
                "support_score_mean": summarize_numeric(r.get("q_train_support_score") for r in records)["mean"],
                "nn_pair_distance_mean": summarize_numeric(r.get("nn_pair_distance_to_q_train") for r in records)["mean"],
                "d_phi_mean": summarize_numeric(r.get("d_phi") for r in records)["mean"],
            }
        )
    for (role,), records in group_rows(support_rows, ["pair_role"]).items():
        bucket_rows.append(
            {
                "env_name": env_name,
                "path_position_bucket": f"role:{role}",
                "num_pairs": len(records),
                "coverage_rate": mean_bool(r.get("q_train_covered") for r in records),
                "support_score_mean": summarize_numeric(r.get("q_train_support_score") for r in records)["mean"],
                "nn_pair_distance_mean": summarize_numeric(r.get("nn_pair_distance_to_q_train") for r in records)["mean"],
                "d_phi_mean": summarize_numeric(r.get("d_phi") for r in records)["mean"],
            }
        )
    return summary, sorted(bucket_rows, key=lambda r: (str(r["env_name"]), str(r["path_position_bucket"]))), support_rows


def mean_bool(values) -> float | None:
    vals = [bool(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def coverage_rate_for(rows: list[dict[str, Any]], predicate) -> float | None:
    selected = [r for r in rows if predicate(r)]
    if not selected:
        return None
    return mean_bool(r.get("q_train_covered") for r in selected)


def main() -> None:
    args = parse_args()
    graph_rows = load_rows(args.graph_jsonl)
    train_rows = load_rows(args.train_jsonl)
    graph_by_env = group_rows(graph_rows, ["env_name"])
    train_by_env = group_rows(train_rows, ["env_name"])
    summaries = []
    bucket_rows_all = []
    support_rows_all = []
    for (env_name,), env_graph_rows in sorted(graph_by_env.items()):
        env_train_rows = train_by_env.get((env_name,), [])
        summary, bucket_rows, support_rows = support_for_env(str(env_name), env_graph_rows, env_train_rows, args)
        summaries.append(summary)
        bucket_rows_all.extend(bucket_rows)
        support_rows_all.extend(support_rows)

    output = {
        "summary": summaries,
        "bucket_coverage": bucket_rows_all,
        "notes": [
            "support_score = exp(-nearest_pair_distance(q_G, q_train) / support_tau)",
            "coverage_rate uses nearest_pair_distance <= coverage_radius",
            "temporal_gap_js_divergence is null unless q_G rows include temporal_gap",
            "recovery_target_coverage_rate requires recovery-labeled q_G or debug step trace target rows",
        ],
    }
    write_json(args.out_json, output)
    if args.out_csv:
        write_csv(args.out_csv, summaries)
    if args.out_pair_support_jsonl:
        write_jsonl(args.out_pair_support_jsonl, support_rows_all)

    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_cols = [
        "env_name",
        "num_graph_pairs",
        "num_train_pairs",
        "d_phi_js_divergence",
        "graph_d_phi_mean",
        "train_d_phi_mean",
        "train_temporal_gap_mean",
        "nn_pair_distance_mean",
        "support_score_mean",
        "coverage_rate",
        "final_phase_coverage_rate",
        "recovery_target_coverage_rate",
    ]
    bucket_cols = ["env_name", "path_position_bucket", "num_pairs", "coverage_rate", "support_score_mean", "nn_pair_distance_mean", "d_phi_mean"]
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# GP0 Graph-Policy Distribution Comparison\n\n")
    write_markdown_table(md_path, "GP0 Graph-Policy Distribution Comparison", summaries, summary_cols)
    with md_path.open("a", encoding="utf-8") as fh:
        fh.write("\n## Coverage By Path Position / Role\n\n")
        if bucket_rows_all:
            fh.write("| " + " | ".join(bucket_cols) + " |\n")
            fh.write("| " + " | ".join("---" for _ in bucket_cols) + " |\n")
            for row in bucket_rows_all:
                vals = []
                for col in bucket_cols:
                    val = row.get(col)
                    vals.append("NA" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val)))
                fh.write("| " + " | ".join(vals) + " |\n")
        else:
            fh.write("No bucket rows.\n")
        fh.write("\n## Notes\n\n")
        for note in output["notes"]:
            fh.write(f"- {note}\n")
    print({"out_json": args.out_json, "out_md": args.out_md, "groups": len(summaries), "support_rows": len(support_rows_all)})


if __name__ == "__main__":
    main()
