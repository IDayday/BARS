#!/usr/bin/env python
"""Run Phase 4E support-only compatibility graph repair."""

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

from phase3e.compatibility_aware_planning import CompatibilityPlannerConfig  # noqa: E402
from phase3e.compatibility_graph_repair import (  # noqa: E402
    GraphRepairConfig,
    run_repair_evaluation,
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
        "output_dir": "results/phase4e",
        "path_queries_csv": None,
        "edge_certification_csv": None,
        "H_intra": 10,
        "max_queries": None,
        "seed": 0,
        "methods": [
            "support_shortest_path",
            "compat_penalized",
            "calibrated_compat_penalized",
            "compat_threshold",
            "calibrated_compat_threshold",
        ],
        "max_repair_edges": 500,
        "min_repair_support": 3,
        "min_repair_episodes": 2,
        "min_pair_coverage": 0.05,
        "bad_junction_weight": 3.0,
        "bad_endpoint_weight": 1.0,
        "support_weight": 0.25,
        "diversity_weight": 0.25,
        "short_horizon_weight": 0.25,
        "pair_weight": 10.0,
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
    for required in ["dataset_name", "base_phase2_run_dir", "repair_bank_phase2_run_dir"]:
        if not merged.get(required):
            raise ValueError(f"--{required} is required")
    merged["methods"] = [str(x) for x in _list_arg(merged["methods"], str)]
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--base_phase2_run_dir", default=None)
    parser.add_argument("--repair_bank_phase2_run_dir", default=None)
    parser.add_argument("--edge_certification_csv", default=None)
    parser.add_argument("--path_queries_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--H_intra", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--max_repair_edges", type=int, default=None)
    parser.add_argument("--min_repair_support", type=int, default=None)
    parser.add_argument("--min_repair_episodes", type=int, default=None)
    parser.add_argument("--min_pair_coverage", type=float, default=None)
    parser.add_argument("--bad_junction_weight", type=float, default=None)
    parser.add_argument("--bad_endpoint_weight", type=float, default=None)
    parser.add_argument("--support_weight", type=float, default=None)
    parser.add_argument("--diversity_weight", type=float, default=None)
    parser.add_argument("--short_horizon_weight", type=float, default=None)
    parser.add_argument("--pair_weight", type=float, default=None)
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


def _plot_outputs(summary: pd.DataFrame, repair_edges: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, group in summary.groupby("graph_variant", sort=False):
            ax.scatter(
                group["path_coverage"],
                group["mean_min_pair_termination_bridge_coverage"],
                label=str(variant),
                s=70,
            )
            for row in group.itertuples(index=False):
                ax.annotate(str(row.method), (row.path_coverage, row.mean_min_pair_termination_bridge_coverage), fontsize=8)
        ax.set_xlabel("path coverage")
        ax.set_ylabel("mean minimum pair bridge coverage")
        ax.set_title("Base vs repaired support graph")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "repair_coverage_vs_pair_compatibility.png", dpi=160)
        plt.close(fig)

    if not repair_edges.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(repair_edges["repair_score"], bins=30)
        ax.set_xlabel("repair score")
        ax.set_ylabel("repair edges")
        ax.set_title("Selected support-bank repair edge scores")
        fig.tight_layout()
        fig.savefig(plots / "repair_edge_score_hist.png", dpi=160)
        plt.close(fig)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 4E Compatibility Graph Repair Summary",
        "",
        "This is reset-free offline graph repair. Repair edges are selected only",
        "from a Phase 2 support-certified edge bank; no kNN/proximity/latent",
        "unsupported edges are introduced.",
        "",
        "## Repair Summary",
        "",
    ]
    for row in payload.get("repair_summary", []):
        for key, value in row.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Planner Metrics", ""])
    for row in payload.get("method_metrics", []):
        lines.extend(
            [
                f"### {row.get('graph_variant')} / {row.get('method')}",
                "",
                f"- `path_coverage`: `{row.get('path_coverage')}`",
                f"- `mean_min_pair_termination_bridge_coverage`: `{row.get('mean_min_pair_termination_bridge_coverage')}`",
                f"- `mean_pair_incompatible_fraction`: `{row.get('mean_pair_incompatible_fraction')}`",
                f"- `mean_base_path_cost`: `{row.get('mean_base_path_cost')}`",
                f"- `coverage_delta_vs_base_graph`: `{row.get('coverage_delta_vs_base_graph')}`",
                f"- `min_pair_coverage_delta_vs_base_graph`: `{row.get('min_pair_coverage_delta_vs_base_graph')}`",
                f"- `pair_incompatible_delta_vs_base_graph`: `{row.get('pair_incompatible_delta_vs_base_graph')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "A positive result is improved compatibility or coverage after adding",
            "only support-certified repair edges. These metrics remain offline",
            "graph evidence and do not prove policy execution success.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_phase2_run_dir).expanduser()
    bank_dir = Path(args.repair_bank_phase2_run_dir).expanduser()
    path_queries_csv = Path(args.path_queries_csv).expanduser() if args.path_queries_csv else _default_path_queries(base_dir)
    dataset_key = _dataset_key(args.dataset_name)
    run_name = f"{_run_name(base_dir)}__repair_{_run_name(bank_dir)}"
    out_dir = Path(args.output_dir).expanduser() / dataset_key / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_edges = pd.read_csv(base_dir / "option_edges.csv")
    bank_edges = pd.read_csv(bank_dir / "option_edges.csv")
    base_segments = _load_edge_segments(base_dir / "edge_segments.npz")
    bank_segments = _load_edge_segments(bank_dir / "edge_segments.npz")
    path_queries = pd.read_csv(path_queries_csv)
    certification = None
    if args.edge_certification_csv:
        certification = pd.read_csv(Path(args.edge_certification_csv).expanduser())

    repair_config = GraphRepairConfig(
        max_repair_edges=int(args.max_repair_edges),
        min_repair_support=int(args.min_repair_support),
        min_repair_episodes=int(args.min_repair_episodes),
        min_pair_coverage=float(args.min_pair_coverage),
        bad_junction_weight=float(args.bad_junction_weight),
        bad_endpoint_weight=float(args.bad_endpoint_weight),
        support_weight=float(args.support_weight),
        diversity_weight=float(args.diversity_weight),
        short_horizon_weight=float(args.short_horizon_weight),
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

    outputs = run_repair_evaluation(
        base_edges=base_edges,
        base_segments=base_segments,
        bank_edges=bank_edges,
        bank_segments=bank_segments,
        path_queries=path_queries,
        certification=certification,
        repair_config=repair_config,
        planner_config=planner_config,
        methods=args.methods,
        H_intra=int(args.H_intra),
        max_queries=args.max_queries,
        seed=int(args.seed),
    )
    payload = {
        "config": vars(args),
        "repair_config": repair_config.__dict__,
        "risk_config": risk_config.__dict__,
        "planner_config": {
            "pair_weight": planner_config.pair_weight,
            "min_pair_coverage": planner_config.min_pair_coverage,
            "missing_pair_coverage": planner_config.missing_pair_coverage,
            "high_pair_risk_threshold": planner_config.high_pair_risk_threshold,
        },
        "base_phase2_run_dir": str(base_dir),
        "repair_bank_phase2_run_dir": str(bank_dir),
        "path_queries_csv": str(path_queries_csv),
        "repair_summary": outputs["repair_summary"].to_dict("records"),
        "pair_summary": outputs["pair_summary"].to_dict("records"),
        "method_metrics": outputs["summary"].to_dict("records"),
        "note": "Reset-free support-only graph repair; not rollout success.",
    }

    outputs["repair_edges"].to_csv(out_dir / "selected_repair_edges.csv", index=False)
    outputs["repair_edge_map"].to_csv(out_dir / "repair_edge_map.csv", index=False)
    outputs["augmented_edges"].to_csv(out_dir / "augmented_option_edges.csv", index=False)
    outputs["pair_summary"].to_csv(out_dir / "repair_pair_summary.csv", index=False)
    outputs["base_pair_compatibility"].to_csv(out_dir / "base_pair_compatibility.csv", index=False)
    outputs["augmented_pair_compatibility"].to_csv(out_dir / "augmented_pair_compatibility.csv", index=False)
    outputs["path_metrics"].to_csv(out_dir / "repair_planning_paths.csv", index=False)
    outputs["summary"].to_csv(out_dir / "repair_planning_summary.csv", index=False)
    outputs["repair_summary"].to_csv(out_dir / "repair_edge_summary.csv", index=False)
    _write_json(out_dir / "phase4e_compatibility_graph_repair_summary.json", payload)
    _write_markdown(out_dir / "phase4e_compatibility_graph_repair_summary.md", payload)
    _plot_outputs(outputs["summary"], outputs["repair_edges"], out_dir)

    print(f"Wrote Phase 4E compatibility graph repair outputs to {out_dir}")
    print(outputs["repair_summary"].to_string(index=False))
    print(outputs["summary"].to_string(index=False))


if __name__ == "__main__":
    main()

