#!/usr/bin/env python
"""Run Phase 4B calibrated risk-aware planner sweep."""

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
from phase3e.risk_aware_sweep import (  # noqa: E402
    RiskSweepConfig,
    make_sweep_configs,
    run_planner_sweep,
    select_recommended_config,
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


def _list_arg(value: Any, cast=float) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [cast(x) for x in value]
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "output_dir": "results/phase4b",
        "path_queries_csv": None,
        "seed": 0,
        "max_queries": None,
        "planner_methods": ["floor_proxy_penalized"],
        "risk_weights": [0.0, 1.0, 2.0, 4.0, 8.0],
        "ood_weights": [0.0, 1.0],
        "incompat_weights": [0.0, 1.0],
        "uncertified_weights": [0.0, 1.0],
        "proxy_floors": [0.0, 0.05, 0.1, 0.2],
        "heldout_support_lcb_floors": [0.0, 0.01, 0.05],
        "high_ood_threshold": 0.5,
        "high_incompat_threshold": 0.5,
        "min_coverage_ratio": 0.95,
        "max_base_cost_increase": 0.2,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    for required in ["dataset_name", "phase2_run_dir", "edge_certification_csv"]:
        if not merged.get(required):
            raise ValueError(f"--{required} is required")
    merged["planner_methods"] = [str(x) for x in _list_arg(merged["planner_methods"], str)]
    for key in [
        "risk_weights",
        "ood_weights",
        "incompat_weights",
        "uncertified_weights",
        "proxy_floors",
        "heldout_support_lcb_floors",
    ]:
        merged[key] = _list_arg(merged[key], float)
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--edge_certification_csv", default=None)
    parser.add_argument("--path_queries_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--planner_methods", default=None)
    parser.add_argument("--risk_weights", default=None)
    parser.add_argument("--ood_weights", default=None)
    parser.add_argument("--incompat_weights", default=None)
    parser.add_argument("--uncertified_weights", default=None)
    parser.add_argument("--proxy_floors", default=None)
    parser.add_argument("--heldout_support_lcb_floors", default=None)
    parser.add_argument("--high_ood_threshold", type=float, default=None)
    parser.add_argument("--high_incompat_threshold", type=float, default=None)
    parser.add_argument("--min_coverage_ratio", type=float, default=None)
    parser.add_argument("--max_base_cost_increase", type=float, default=None)
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


def _plot_sweep(sweep: pd.DataFrame, pareto: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if sweep.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sweep["path_coverage"],
        sweep["mean_min_edge_proxy_score"],
        c=sweep["mean_uncertified_edge_fraction"],
        cmap="viridis_r",
        s=30,
        alpha=0.55,
    )
    if not pareto.empty:
        ax.scatter(
            pareto["path_coverage"],
            pareto["mean_min_edge_proxy_score"],
            facecolors="none",
            edgecolors="#d62728",
            s=90,
            linewidths=1.5,
            label="Pareto",
        )
        ax.legend()
    ax.set_xlabel("path coverage")
    ax.set_ylabel("mean minimum edge proxy")
    ax.set_title("Phase 4B coverage/proxy trade-off")
    fig.tight_layout()
    fig.savefig(plots / "pareto_coverage_vs_min_proxy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sweep["path_coverage"],
        sweep["mean_uncertified_edge_fraction"],
        c=sweep["mean_base_path_cost"],
        cmap="magma",
        s=30,
        alpha=0.6,
    )
    ax.set_xlabel("path coverage")
    ax.set_ylabel("mean uncertified edge fraction")
    ax.set_title("Coverage vs uncertified edge exposure")
    fig.tight_layout()
    fig.savefig(plots / "coverage_vs_uncertified_sweep.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sweep["mean_base_path_cost"],
        sweep["mean_min_edge_proxy_score"],
        c=sweep["path_coverage"],
        cmap="plasma",
        s=30,
        alpha=0.6,
    )
    ax.set_xlabel("mean base path cost")
    ax.set_ylabel("mean minimum edge proxy")
    ax.set_title("Cost vs proxy trade-off")
    fig.tight_layout()
    fig.savefig(plots / "cost_vs_min_proxy_sweep.png", dpi=160)
    plt.close(fig)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    baseline = pd.DataFrame(payload.get("baseline_metrics", []))
    recommended = payload.get("recommended_config", {})
    lines = [
        "# Phase 4B Calibrated Risk-Aware Planner Sweep",
        "",
        "This is reset-free and offline-only. It sweeps support-only risk-aware",
        "planner parameters and does not claim rollout success.",
        "",
        f"Total sweep configs: `{payload.get('num_sweep_configs')}`",
        f"Pareto configs: `{payload.get('num_pareto_configs')}`",
        "",
        "## Baselines",
        "",
    ]
    if not baseline.empty:
        lines.extend(
            [
                "| method | coverage | mean min proxy | uncertified frac | base cost |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in baseline.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.path_coverage:.3f} | "
                f"{row.mean_min_edge_proxy_score:.3f} | "
                f"{row.mean_uncertified_edge_fraction:.3f} | "
                f"{row.mean_base_path_cost:.3f} |"
            )
    lines.extend(["", "## Recommended Config", ""])
    if recommended:
        for key in [
            "method",
            "planner_method",
            "path_coverage",
            "mean_min_edge_proxy_score",
            "mean_uncertified_edge_fraction",
            "mean_base_path_cost",
            "risk_weight",
            "ood_weight",
            "incompat_weight",
            "uncertified_weight",
            "min_proxy_score",
            "min_heldout_support_lcb",
            "is_pareto",
        ]:
            if key in recommended:
                lines.append(f"- `{key}`: `{recommended[key]}`")
    lines.extend(
        [
            "",
            "Interpretation: the recommended config is selected by an offline",
            "coverage/risk heuristic constrained against the support-shortest-path",
            "baseline. It is not a calibrated execution policy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    cert_csv = Path(args.edge_certification_csv).expanduser()
    path_queries_csv = Path(args.path_queries_csv).expanduser() if args.path_queries_csv else _default_path_queries(phase2_run_dir)

    dataset_key = _dataset_key(args.dataset_name)
    run_name = _run_name(phase2_run_dir)
    out_dir = Path(args.output_dir).expanduser() / dataset_key / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    option_edges = pd.read_csv(phase2_run_dir / "option_edges.csv")
    certification = pd.read_csv(cert_csv)
    path_queries = pd.read_csv(path_queries_csv)
    edge_table = load_edge_table(option_edges, certification)

    baseline_paths, baseline_graphs = evaluate_planning_methods(
        edge_table,
        path_queries,
        methods=["support_shortest_path", "proxy_penalized", "proxy_threshold", "certified_only"],
        config=RiskPlannerConfig(),
        max_queries=args.max_queries,
        seed=int(args.seed),
    )
    baseline_summary = summarize_planning_results(baseline_paths, baseline_graphs)

    sweep_configs = make_sweep_configs(
        planner_methods=args.planner_methods,
        risk_weights=args.risk_weights,
        ood_weights=args.ood_weights,
        incompat_weights=args.incompat_weights,
        uncertified_weights=args.uncertified_weights,
        proxy_floors=args.proxy_floors,
        heldout_support_lcb_floors=args.heldout_support_lcb_floors,
        high_ood_threshold=float(args.high_ood_threshold),
        high_incompat_threshold=float(args.high_incompat_threshold),
    )
    sweep_summary = run_planner_sweep(
        edge_table,
        path_queries,
        sweep_configs,
        max_queries=args.max_queries,
        seed=int(args.seed),
    )
    pareto = sweep_summary[sweep_summary["is_pareto"]].copy() if not sweep_summary.empty else pd.DataFrame()
    recommended = select_recommended_config(
        sweep_summary,
        baseline_summary,
        min_coverage_ratio=float(args.min_coverage_ratio),
        max_base_cost_increase=float(args.max_base_cost_increase),
    )

    best_paths = pd.DataFrame()
    if recommended:
        best_config = RiskSweepConfig(
            planner_method=str(recommended["planner_method"]),
            risk_weight=float(recommended["risk_weight"]),
            ood_weight=float(recommended["ood_weight"]),
            incompat_weight=float(recommended["incompat_weight"]),
            uncertified_weight=float(recommended["uncertified_weight"]),
            min_proxy_score=float(recommended["min_proxy_score"]),
            min_heldout_support_lcb=float(recommended["min_heldout_support_lcb"]),
            high_ood_threshold=float(args.high_ood_threshold),
            high_incompat_threshold=float(args.high_incompat_threshold),
        )
        best_paths, _ = evaluate_planning_methods(
            edge_table,
            path_queries,
            methods=[best_config.planner_method],
            config=best_config.planner_config(),
            max_queries=args.max_queries,
            seed=int(args.seed),
        )
        best_paths["sweep_id"] = int(recommended["sweep_id"])
        best_paths["recommended_method"] = str(recommended["method"])

    payload = planning_summary_dict(baseline_summary)
    payload.update(
        {
            "config": vars(args),
            "phase2_run_dir": str(phase2_run_dir),
            "edge_certification_csv": str(cert_csv),
            "path_queries_csv": str(path_queries_csv),
            "num_sweep_configs": int(sweep_summary.shape[0]),
            "num_pareto_configs": int(pareto.shape[0]),
            "baseline_metrics": baseline_summary.to_dict("records"),
            "recommended_config": recommended,
            "note": "Reset-free offline planner sweep; not rollout success.",
        }
    )

    baseline_summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    sweep_summary.to_csv(out_dir / "risk_sweep_summary.csv", index=False)
    pareto.to_csv(out_dir / "pareto_front.csv", index=False)
    best_paths.to_csv(out_dir / "recommended_paths.csv", index=False)
    _write_json(out_dir / "phase4b_risk_sweep_summary.json", payload)
    _write_markdown(out_dir / "phase4b_risk_sweep_summary.md", payload)
    _plot_sweep(sweep_summary, pareto, out_dir)

    print(f"Wrote Phase 4B risk sweep outputs to {out_dir}")
    print("Baselines:")
    print(baseline_summary.to_string(index=False))
    print("\nRecommended:")
    print(pd.Series(recommended).to_string() if recommended else "none")


if __name__ == "__main__":
    main()
