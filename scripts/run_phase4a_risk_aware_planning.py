#!/usr/bin/env python
"""Run Phase 4A reset-free risk-aware planning over support option edges."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.risk_aware_planning import (  # noqa: E402
    RiskPlannerConfig,
    evaluate_planning_methods,
    load_edge_table,
    planning_summary_dict,
    summarize_planning_results,
)


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _run_name(path: str | Path) -> str:
    return Path(path).expanduser().name


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _parse_methods(value: Any) -> list[str]:
    if value is None:
        return ["support_shortest_path", "certified_only", "proxy_threshold", "proxy_penalized"]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "output_dir": "results/phase4a",
        "path_queries_csv": None,
        "methods": ["support_shortest_path", "certified_only", "proxy_threshold", "proxy_penalized"],
        "seed": 0,
        "max_queries": None,
        "risk_weight": 2.0,
        "ood_weight": 1.0,
        "incompat_weight": 1.0,
        "uncertified_weight": 1.0,
        "min_proxy_score": 0.25,
        "min_heldout_support_lcb": 0.01,
        "high_ood_threshold": 0.5,
        "high_incompat_threshold": 0.5,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    for required in ["dataset_name", "phase2_run_dir", "edge_certification_csv"]:
        if not merged.get(required):
            raise ValueError(f"--{required} is required")
    merged["methods"] = _parse_methods(merged.get("methods"))
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--edge_certification_csv", default=None)
    parser.add_argument("--path_queries_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--risk_weight", type=float, default=None)
    parser.add_argument("--ood_weight", type=float, default=None)
    parser.add_argument("--incompat_weight", type=float, default=None)
    parser.add_argument("--uncertified_weight", type=float, default=None)
    parser.add_argument("--min_proxy_score", type=float, default=None)
    parser.add_argument("--min_heldout_support_lcb", type=float, default=None)
    parser.add_argument("--high_ood_threshold", type=float, default=None)
    parser.add_argument("--high_incompat_threshold", type=float, default=None)
    return _merge_args(parser.parse_args())


def _default_path_queries(phase2_run_dir: Path) -> Path:
    candidates = [
        phase2_run_dir / "path_queries.csv",
        phase2_run_dir.parent / "path_queries.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer path_queries.csv from {phase2_run_dir}; pass --path_queries_csv"
    )


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("method_metrics", [])
    lines = [
        "# Phase 4A Risk-Aware Offline Planning Summary",
        "",
        "This is a reset-free offline planning result. It does not run rollout",
        "and does not claim policy execution success.",
        "",
        f"Best path coverage method: `{summary.get('best_path_coverage_method')}`",
        f"Best mean minimum edge proxy method: `{summary.get('best_mean_min_edge_proxy_method')}`",
        f"Lowest uncertified edge fraction method: `{summary.get('lowest_uncertified_edge_fraction_method')}`",
        "",
        "| method | coverage | mean min proxy | uncertified frac | base cost | graph edges |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {path_coverage:.3f} | {mean_min_edge_proxy_score:.3f} | "
            "{mean_uncertified_edge_fraction:.3f} | {mean_base_path_cost:.3f} | {num_graph_edges} |".format(
                method=row.get("method"),
                path_coverage=float(row.get("path_coverage") or 0.0),
                mean_min_edge_proxy_score=float(row.get("mean_min_edge_proxy_score") or 0.0),
                mean_uncertified_edge_fraction=float(row.get("mean_uncertified_edge_fraction") or 0.0),
                mean_base_path_cost=float(row.get("mean_base_path_cost") or 0.0),
                num_graph_edges=int(row.get("num_graph_edges") or 0),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: higher proxy scores and lower uncertified fractions are",
            "offline risk indicators only. They are not calibrated rollout success",
            "probabilities.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    x = np.arange(summary.shape[0])
    labels = summary["method"].astype(str).tolist()

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(x - 0.2, summary["path_coverage"], width=0.4, label="coverage", color="#4c78a8")
    ax1.set_ylabel("path coverage")
    ax1.set_ylim(0, max(1.0, float(summary["path_coverage"].max()) * 1.15))
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.2,
        summary["mean_min_edge_proxy_score"],
        width=0.4,
        label="mean min proxy",
        color="#f58518",
    )
    ax2.set_ylabel("mean min edge proxy")
    ax2.set_ylim(0, max(1.0, float(summary["mean_min_edge_proxy_score"].max()) * 1.15))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Coverage vs offline edge proxy")
    fig.tight_layout()
    fig.savefig(plots / "coverage_vs_proxy_by_method.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        summary["path_coverage"],
        summary["mean_uncertified_edge_fraction"],
        s=90,
        color="#54a24b",
    )
    for row in summary.itertuples(index=False):
        ax.annotate(str(row.method), (row.path_coverage, row.mean_uncertified_edge_fraction), fontsize=8)
    ax.set_xlabel("path coverage")
    ax.set_ylabel("mean uncertified edge fraction")
    ax.set_title("Offline risk trade-off")
    fig.tight_layout()
    fig.savefig(plots / "coverage_vs_uncertified_fraction.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    option_edges_csv = phase2_run_dir / "option_edges.csv"
    path_queries_csv = Path(args.path_queries_csv).expanduser() if args.path_queries_csv else _default_path_queries(phase2_run_dir)
    cert_csv = Path(args.edge_certification_csv).expanduser()

    dataset_key = _dataset_key(args.dataset_name)
    run_name = _run_name(phase2_run_dir)
    out_dir = Path(args.output_dir).expanduser() / dataset_key / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    option_edges = pd.read_csv(option_edges_csv)
    certification = pd.read_csv(cert_csv)
    path_queries = pd.read_csv(path_queries_csv)
    edge_table = load_edge_table(option_edges, certification)

    planner_config = RiskPlannerConfig(
        risk_weight=float(args.risk_weight),
        ood_weight=float(args.ood_weight),
        incompat_weight=float(args.incompat_weight),
        uncertified_weight=float(args.uncertified_weight),
        min_proxy_score=float(args.min_proxy_score),
        min_heldout_support_lcb=float(args.min_heldout_support_lcb),
        high_ood_threshold=float(args.high_ood_threshold),
        high_incompat_threshold=float(args.high_incompat_threshold),
    )
    paths, graph_metrics = evaluate_planning_methods(
        edge_table,
        path_queries,
        methods=list(args.methods),
        config=planner_config,
        max_queries=args.max_queries,
        seed=int(args.seed),
    )
    summary = summarize_planning_results(paths, graph_metrics)
    payload = planning_summary_dict(summary)
    payload["config"] = vars(args)
    payload["phase2_run_dir"] = str(phase2_run_dir)
    payload["edge_certification_csv"] = str(cert_csv)
    payload["path_queries_csv"] = str(path_queries_csv)

    paths.to_csv(out_dir / "risk_aware_paths.csv", index=False)
    graph_metrics.to_csv(out_dir / "risk_aware_graphs.csv", index=False)
    summary.to_csv(out_dir / "risk_aware_planning_summary.csv", index=False)
    _write_json(out_dir / "risk_aware_planning_summary.json", payload)
    _write_markdown(out_dir / "risk_aware_planning_summary.md", payload)
    _plot_summary(summary, out_dir)

    print(f"Wrote Phase 4A risk-aware planning outputs to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
