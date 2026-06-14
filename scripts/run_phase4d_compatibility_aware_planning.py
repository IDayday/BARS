#!/usr/bin/env python
"""Run Phase 4D reset-free compatibility-aware support planning."""

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

from phase3e.compatibility_aware_planning import (  # noqa: E402
    CompatibilityPlannerConfig,
    compatibility_summary_dict,
    compute_pair_compatibility_from_segments,
    evaluate_compatibility_planning_methods,
    make_compatibility_edge_table,
    summarize_compatibility_planning,
)
from phase3e.risk_aware_planning import RiskPlannerConfig  # noqa: E402


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


def _list_arg(value: Any, cast=str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [cast(x) for x in value]
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def _default_path_queries(phase2_run_dir: Path) -> Path:
    for candidate in [phase2_run_dir / "path_queries.csv", phase2_run_dir.parent / "path_queries.csv"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer path_queries.csv from {phase2_run_dir}; pass --path_queries_csv"
    )


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "output_dir": "results/phase4d",
        "path_queries_csv": None,
        "edge_certification_csv": None,
        "pair_compatibility_csv": None,
        "H_intra": 10,
        "max_queries": None,
        "seed": 0,
        "methods": [
            "support_shortest_path",
            "calibrated_edge_penalized",
            "compat_penalized",
            "calibrated_compat_penalized",
            "compat_threshold",
            "calibrated_compat_threshold",
        ],
        "pair_weight": 10.0,
        "min_pair_coverage": 0.05,
        "missing_pair_coverage": 0.0,
        "high_pair_risk_threshold": 0.5,
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
    for required in ["dataset_name", "phase2_run_dir"]:
        if not merged.get(required):
            raise ValueError(f"--{required} is required")
    merged["methods"] = [str(x) for x in _list_arg(merged["methods"], str)]
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--edge_certification_csv", default=None)
    parser.add_argument("--pair_compatibility_csv", default=None)
    parser.add_argument("--path_queries_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--H_intra", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--pair_weight", type=float, default=None)
    parser.add_argument("--min_pair_coverage", type=float, default=None)
    parser.add_argument("--missing_pair_coverage", type=float, default=None)
    parser.add_argument("--high_pair_risk_threshold", type=float, default=None)
    parser.add_argument("--risk_weight", type=float, default=None)
    parser.add_argument("--ood_weight", type=float, default=None)
    parser.add_argument("--incompat_weight", type=float, default=None)
    parser.add_argument("--uncertified_weight", type=float, default=None)
    parser.add_argument("--min_proxy_score", type=float, default=None)
    parser.add_argument("--min_heldout_support_lcb", type=float, default=None)
    parser.add_argument("--high_ood_threshold", type=float, default=None)
    parser.add_argument("--high_incompat_threshold", type=float, default=None)
    return _merge_args(parser.parse_args())


def _load_edge_segments(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _plot_outputs(summary: pd.DataFrame, pair_df: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            summary["path_coverage"],
            summary["mean_min_pair_termination_bridge_coverage"],
            s=70,
        )
        for row in summary.itertuples(index=False):
            ax.annotate(str(row.method), (row.path_coverage, row.mean_min_pair_termination_bridge_coverage), fontsize=8)
        ax.set_xlabel("path coverage")
        ax.set_ylabel("mean minimum pair bridge coverage")
        ax.set_title("Coverage vs adjacent-option compatibility")
        fig.tight_layout()
        fig.savefig(plots / "coverage_vs_pair_compatibility.png", dpi=160)
        plt.close(fig)

    if not pair_df.empty and "termination_bridge_coverage" in pair_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pair_df["termination_bridge_coverage"], bins=30)
        ax.set_xlabel("termination bridge coverage")
        ax.set_ylabel("adjacent edge pairs")
        ax.set_title("Adjacent-edge compatibility distribution")
        fig.tight_layout()
        fig.savefig(plots / "pair_bridge_coverage_hist.png", dpi=160)
        plt.close(fig)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 4D Compatibility-Aware Planning Summary",
        "",
        "This is reset-free offline planning. It does not use environment rollout",
        "and does not claim option execution success.",
        "",
        "## Methods",
        "",
        "- `support_shortest_path`: Phase 2 support graph shortest path.",
        "- `calibrated_edge_penalized`: Phase 4C calibrated single-edge risk cost.",
        "- `compat_penalized`: support edges with adjacent-edge bridge penalty.",
        "- `calibrated_compat_penalized`: calibrated edge risk plus bridge penalty.",
        "- `compat_threshold`: rejects adjacent edge transitions below the bridge floor.",
        "- `calibrated_compat_threshold`: calibrated edge risk with bridge floor.",
        "",
        "## Key Metrics",
        "",
    ]
    for row in payload.get("method_metrics", []):
        lines.extend(
            [
                f"### {row.get('method')}",
                "",
                f"- `path_coverage`: `{row.get('path_coverage')}`",
                f"- `mean_min_pair_termination_bridge_coverage`: `{row.get('mean_min_pair_termination_bridge_coverage')}`",
                f"- `mean_pair_incompatible_fraction`: `{row.get('mean_pair_incompatible_fraction')}`",
                f"- `mean_min_edge_proxy_score`: `{row.get('mean_min_edge_proxy_score')}`",
                f"- `mean_original_uncertified_edge_fraction`: `{row.get('mean_original_uncertified_edge_fraction')}`",
                f"- `mean_base_path_cost`: `{row.get('mean_base_path_cost')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Phase 4D evaluates whether path selection improves when option",
            "composition is treated as a transition-dependent cost. A better",
            "offline planner should preserve useful coverage while raising",
            "path-level adjacent-edge bridge coverage and lowering incompatible",
            "pair exposure. These are graph-layer proxies, not rollout labels.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    path_queries_csv = Path(args.path_queries_csv).expanduser() if args.path_queries_csv else _default_path_queries(phase2_run_dir)
    dataset_key = _dataset_key(args.dataset_name)
    run_name = _run_name(phase2_run_dir)
    out_dir = Path(args.output_dir).expanduser() / dataset_key / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    option_edges = pd.read_csv(phase2_run_dir / "option_edges.csv")
    certification = None
    if args.edge_certification_csv:
        certification = pd.read_csv(Path(args.edge_certification_csv).expanduser())
    edge_table = make_compatibility_edge_table(option_edges, certification)
    path_queries = pd.read_csv(path_queries_csv)

    if args.pair_compatibility_csv and Path(args.pair_compatibility_csv).expanduser().exists():
        pair_summary = pd.DataFrame()
        pair_df = pd.read_csv(Path(args.pair_compatibility_csv).expanduser())
    else:
        edge_segments = _load_edge_segments(phase2_run_dir / "edge_segments.npz")
        pair_summary, pair_df = compute_pair_compatibility_from_segments(
            option_edges,
            edge_segments,
            H_intra=int(args.H_intra),
        )

    risk_config = RiskPlannerConfig(
        risk_weight=float(args.risk_weight),
        ood_weight=float(args.ood_weight),
        incompat_weight=float(args.incompat_weight),
        uncertified_weight=float(args.uncertified_weight),
        min_proxy_score=float(args.min_proxy_score),
        min_heldout_support_lcb=float(args.min_heldout_support_lcb),
        high_ood_threshold=float(args.high_ood_threshold),
        high_incompat_threshold=float(args.high_incompat_threshold),
    )
    planner_config = CompatibilityPlannerConfig(
        pair_weight=float(args.pair_weight),
        min_pair_coverage=float(args.min_pair_coverage),
        missing_pair_coverage=float(args.missing_pair_coverage),
        high_pair_risk_threshold=float(args.high_pair_risk_threshold),
        risk_config=risk_config,
    )

    path_metrics, graph_metrics = evaluate_compatibility_planning_methods(
        edge_table=edge_table,
        pair_compatibility=pair_df,
        path_queries=path_queries,
        methods=args.methods,
        config=planner_config,
        max_queries=args.max_queries,
        seed=int(args.seed),
    )
    summary = summarize_compatibility_planning(path_metrics, graph_metrics)
    payload = compatibility_summary_dict(summary, pair_summary)
    payload.update(
        {
            "config": vars(args),
            "risk_config": risk_config.__dict__,
            "planner_config": {
                "pair_weight": planner_config.pair_weight,
                "min_pair_coverage": planner_config.min_pair_coverage,
                "missing_pair_coverage": planner_config.missing_pair_coverage,
                "high_pair_risk_threshold": planner_config.high_pair_risk_threshold,
            },
            "phase2_run_dir": str(phase2_run_dir),
            "path_queries_csv": str(path_queries_csv),
            "num_pair_rows": int(pair_df.shape[0]),
            "note": "Reset-free offline compatibility-aware planning; not rollout success.",
        }
    )

    pair_summary.to_csv(out_dir / "pair_compatibility_summary.csv", index=False)
    pair_df.to_csv(out_dir / "pair_compatibility.csv", index=False)
    path_metrics.to_csv(out_dir / "compatibility_planning_paths.csv", index=False)
    summary.to_csv(out_dir / "compatibility_planning_summary.csv", index=False)
    _write_json(out_dir / "phase4d_compatibility_aware_summary.json", payload)
    _write_markdown(out_dir / "phase4d_compatibility_aware_summary.md", payload)
    _plot_outputs(summary, pair_df, out_dir)

    print(f"Wrote Phase 4D compatibility-aware outputs to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

