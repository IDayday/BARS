#!/usr/bin/env python
"""Run Phase 4F repair-edge certification and joint planning."""

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
from phase3e.edge_risk_calibration import EdgeRiskCalibrationConfig  # noqa: E402
from phase3e.repair_edge_certification import (  # noqa: E402
    RepairCertificationConfig,
    build_repair_edge_certification,
    evaluate_repair_certified_planning,
)
from phase3e.risk_aware_planning import RiskPlannerConfig  # noqa: E402


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


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


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "output_dir": "results/phase4f",
        "max_queries": None,
        "seed": 0,
        "methods": [
            "support_shortest_path",
            "calibrated_compat_penalized",
            "compat_threshold",
            "calibrated_compat_threshold",
        ],
        "support_floor": 0.02,
        "support_scale": 0.85,
        "endpoint_fallback_quantile": 0.25,
        "policy_transfer_scale": 0.85,
        "compatibility_floor": 0.02,
        "support_weight": 0.30,
        "policy_weight": 0.20,
        "behavior_weight": 0.20,
        "compatibility_weight": 0.20,
        "diversity_weight": 0.10,
        "min_component_score": 0.02,
        "certification_threshold": 0.18,
        "min_support_lcb": 0.02,
        "min_compatibility": 0.02,
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
    for required in ["dataset_name", "phase4e_run_dir", "base_edge_certification_csv", "path_queries_csv"]:
        if not merged.get(required):
            raise ValueError(f"--{required} is required")
    merged["methods"] = [str(x) for x in _list_arg(merged["methods"], str)]
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--phase4e_run_dir", default=None)
    parser.add_argument("--base_edge_certification_csv", default=None)
    parser.add_argument("--path_queries_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--support_floor", type=float, default=None)
    parser.add_argument("--support_scale", type=float, default=None)
    parser.add_argument("--endpoint_fallback_quantile", type=float, default=None)
    parser.add_argument("--policy_transfer_scale", type=float, default=None)
    parser.add_argument("--compatibility_floor", type=float, default=None)
    parser.add_argument("--support_weight", type=float, default=None)
    parser.add_argument("--policy_weight", type=float, default=None)
    parser.add_argument("--behavior_weight", type=float, default=None)
    parser.add_argument("--compatibility_weight", type=float, default=None)
    parser.add_argument("--diversity_weight", type=float, default=None)
    parser.add_argument("--min_component_score", type=float, default=None)
    parser.add_argument("--certification_threshold", type=float, default=None)
    parser.add_argument("--min_support_lcb", type=float, default=None)
    parser.add_argument("--min_compatibility", type=float, default=None)
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


def _plot_outputs(summary: pd.DataFrame, repair_cert: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            summary["path_coverage"],
            summary["mean_min_edge_proxy_score"],
            c=summary["mean_repair_edge_fraction"].fillna(0.0),
            cmap="viridis",
            s=70,
        )
        for row in summary.itertuples(index=False):
            ax.annotate(str(row.method), (row.path_coverage, row.mean_min_edge_proxy_score), fontsize=8)
        ax.set_xlabel("path coverage")
        ax.set_ylabel("mean minimum edge proxy")
        ax.set_title("Repair-certified planning trade-off")
        fig.tight_layout()
        fig.savefig(plots / "coverage_vs_proxy_with_repair_fraction.png", dpi=160)
        plt.close(fig)

    repair = repair_cert[repair_cert.get("is_repair_edge", False).astype(bool)].copy()
    if not repair.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(repair["edge_proxy_score"], bins=30)
        ax.set_xlabel("repair edge planner proxy")
        ax.set_ylabel("repair edges")
        ax.set_title("Repair transfer certification scores")
        fig.tight_layout()
        fig.savefig(plots / "repair_edge_proxy_hist.png", dpi=160)
        plt.close(fig)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 4F Repair-Edge Certification Summary",
        "",
        "This is reset-free offline certification. Repair-edge scores are",
        "conservative transfer proxies from support scale, endpoint-neighbor",
        "policy fitting, behavior support, and compatibility context. They are",
        "not rollout success probabilities.",
        "",
        "## Diagnostics",
        "",
    ]
    for key, value in payload.get("repair_diagnostics", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Planner Metrics", ""])
    for row in payload.get("method_metrics", []):
        lines.extend(
            [
                f"### {row.get('method')}",
                "",
                f"- `path_coverage`: `{row.get('path_coverage')}`",
                f"- `mean_min_edge_proxy_score`: `{row.get('mean_min_edge_proxy_score')}`",
                f"- `mean_uncertified_edge_fraction`: `{row.get('mean_uncertified_edge_fraction')}`",
                f"- `mean_original_uncertified_edge_fraction`: `{row.get('mean_original_uncertified_edge_fraction')}`",
                f"- `mean_pair_incompatible_fraction`: `{row.get('mean_pair_incompatible_fraction')}`",
                f"- `mean_repair_edge_fraction`: `{row.get('mean_repair_edge_fraction')}`",
                f"- `mean_repair_certified_fraction`: `{row.get('mean_repair_certified_fraction')}`",
                f"- `mean_base_path_cost`: `{row.get('mean_base_path_cost')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Phase 4F should be read as a planner-facing repair-edge risk estimate.",
            "It narrows the Phase 4E gap where repair edges improved coverage but",
            "were treated as uncertified defaults. It still does not replace",
            "GCBC edge rollout validation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase4e_dir = Path(args.phase4e_run_dir).expanduser()
    dataset_key = _dataset_key(args.dataset_name)
    out_dir = Path(args.output_dir).expanduser() / dataset_key / phase4e_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    augmented_edges = pd.read_csv(phase4e_dir / "augmented_option_edges.csv")
    pair_compatibility = pd.read_csv(phase4e_dir / "augmented_pair_compatibility.csv")
    base_certification = pd.read_csv(Path(args.base_edge_certification_csv).expanduser())
    path_queries = pd.read_csv(Path(args.path_queries_csv).expanduser())

    repair_config = RepairCertificationConfig(
        support_floor=float(args.support_floor),
        support_scale=float(args.support_scale),
        endpoint_fallback_quantile=float(args.endpoint_fallback_quantile),
        policy_transfer_scale=float(args.policy_transfer_scale),
        compatibility_floor=float(args.compatibility_floor),
    )
    calibration_config = EdgeRiskCalibrationConfig(
        support_weight=float(args.support_weight),
        policy_weight=float(args.policy_weight),
        behavior_weight=float(args.behavior_weight),
        compatibility_weight=float(args.compatibility_weight),
        diversity_weight=float(args.diversity_weight),
        min_component_score=float(args.min_component_score),
        certification_threshold=float(args.certification_threshold),
        min_support_lcb=float(args.min_support_lcb),
        min_compatibility=float(args.min_compatibility),
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

    calibrated, planner_cert, diagnostics = build_repair_edge_certification(
        augmented_edges=augmented_edges,
        base_certification=base_certification,
        pair_compatibility=pair_compatibility,
        repair_config=repair_config,
        calibration_config=calibration_config,
    )
    path_metrics, summary = evaluate_repair_certified_planning(
        augmented_edges=augmented_edges,
        pair_compatibility=pair_compatibility,
        path_queries=path_queries,
        planner_certification=planner_cert,
        planner_config=planner_config,
        methods=args.methods,
        max_queries=args.max_queries,
        seed=int(args.seed),
    )
    payload = {
        "config": vars(args),
        "repair_config": repair_config.__dict__,
        "calibration_config": calibration_config.__dict__,
        "risk_config": risk_config.__dict__,
        "planner_config": {
            "pair_weight": planner_config.pair_weight,
            "min_pair_coverage": planner_config.min_pair_coverage,
            "missing_pair_coverage": planner_config.missing_pair_coverage,
            "high_pair_risk_threshold": planner_config.high_pair_risk_threshold,
        },
        "phase4e_run_dir": str(phase4e_dir),
        "base_edge_certification_csv": str(Path(args.base_edge_certification_csv).expanduser()),
        "repair_diagnostics": diagnostics,
        "method_metrics": summary.to_dict("records"),
        "note": "Repair-edge certification is a reset-free transfer proxy, not rollout success.",
    }

    calibrated.to_csv(out_dir / "repair_edge_certification.csv", index=False)
    planner_cert.to_csv(out_dir / "planner_repair_edge_certification.csv", index=False)
    path_metrics.to_csv(out_dir / "repair_certified_planning_paths.csv", index=False)
    summary.to_csv(out_dir / "repair_certified_planning_summary.csv", index=False)
    _write_json(out_dir / "phase4f_repair_edge_certification_summary.json", payload)
    _write_markdown(out_dir / "phase4f_repair_edge_certification_summary.md", payload)
    _plot_outputs(summary, planner_cert, out_dir)

    print(f"Wrote Phase 4F repair-edge certification outputs to {out_dir}")
    print(pd.Series(diagnostics).to_string())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

