#!/usr/bin/env python
"""Run Phase 4G direct GCBC policy evidence for repair edges."""

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

from phase1.data import load_ogbench_dataset  # noqa: E402
from phase3e.compatibility_aware_planning import CompatibilityPlannerConfig  # noqa: E402
from phase3e.direct_repair_policy import (  # noqa: E402
    DirectRepairPolicyConfig,
    apply_direct_policy_scores,
    direct_repair_edge_policy_scores,
    direct_vs_transfer_diagnostics,
    evaluate_direct_repair_policy_planning,
)
from phase3e.edge_risk_calibration import EdgeRiskCalibrationConfig  # noqa: E402
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
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "max_transitions": 200000,
        "output_dir": "results/phase4g",
        "max_queries": None,
        "seed": 0,
        "batch_size": 2048,
        "max_examples": 50000,
        "temperature": 0.05,
        "device": None,
        "methods": [
            "support_shortest_path",
            "calibrated_compat_penalized",
            "compat_threshold",
            "calibrated_compat_threshold",
        ],
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
    for required in [
        "dataset_name",
        "phase4e_run_dir",
        "phase4f_run_dir",
        "repair_bank_phase2_run_dir",
        "model_path",
        "path_queries_csv",
    ]:
        if not merged.get(required):
            raise ValueError(f"--{required} is required")
    merged["methods"] = [str(x) for x in _list_arg(merged["methods"], str)]
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--phase4e_run_dir", default=None)
    parser.add_argument("--phase4f_run_dir", default=None)
    parser.add_argument("--repair_bank_phase2_run_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--path_queries_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--methods", default=None)
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _plot_outputs(direct_scores: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if not direct_scores.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(direct_scores["edge_action_mse"].dropna(), bins=30)
        ax.set_xlabel("direct repair edge action MSE")
        ax.set_ylabel("repair edges")
        ax.set_title("Direct GCBC repair-edge action fitting")
        fig.tight_layout()
        fig.savefig(plots / "direct_repair_edge_mse_hist.png", dpi=160)
        plt.close(fig)

        if "transfer_edge_policy_support_score" in direct_scores.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                direct_scores["transfer_edge_policy_support_score"],
                direct_scores["direct_edge_policy_support_score"],
                alpha=0.65,
                s=20,
            )
            ax.set_xlabel("transfer policy support score")
            ax.set_ylabel("direct policy support score")
            ax.set_title("Transfer vs direct repair policy support")
            fig.tight_layout()
            fig.savefig(plots / "transfer_vs_direct_policy_support.png", dpi=160)
            plt.close(fig)

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
        ax.set_title("Direct repair-policy planning trade-off")
        fig.tight_layout()
        fig.savefig(plots / "direct_policy_coverage_vs_proxy.png", dpi=160)
        plt.close(fig)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 4G Direct Repair-Edge Policy Evidence Summary",
        "",
        "This is reset-free offline supervised policy evidence. Direct repair",
        "edge MSE evaluates a trained GCBC model on real repair-edge segments;",
        "it is not rollout success.",
        "",
        "## Diagnostics",
        "",
    ]
    for key, value in payload.get("diagnostics", {}).items():
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
            "Phase 4G replaces the repair-edge policy component with direct GCBC",
            "action-fitting evidence where possible. The result is stronger than",
            "Phase 4F's transfer-only proxy, but still does not prove closed-loop",
            "edge execution.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase4e_dir = Path(args.phase4e_run_dir).expanduser()
    phase4f_dir = Path(args.phase4f_run_dir).expanduser()
    bank_dir = Path(args.repair_bank_phase2_run_dir).expanduser()
    dataset_key = _dataset_key(args.dataset_name)
    out_dir = Path(args.output_dir).expanduser() / dataset_key / phase4e_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    repair_map = pd.read_csv(phase4e_dir / "repair_edge_map.csv")
    bank_edges = pd.read_csv(bank_dir / "option_edges.csv")
    bank_segments = _load_npz(bank_dir / "edge_segments.npz")
    transfer_cert = pd.read_csv(phase4f_dir / "repair_edge_certification.csv")
    augmented_edges = pd.read_csv(phase4e_dir / "augmented_option_edges.csv")
    pair_compatibility = pd.read_csv(phase4e_dir / "augmented_pair_compatibility.csv")
    path_queries = pd.read_csv(Path(args.path_queries_csv).expanduser())

    direct_config = DirectRepairPolicyConfig(
        batch_size=int(args.batch_size),
        max_examples=int(args.max_examples),
        temperature=float(args.temperature),
        seed=int(args.seed),
        device=args.device,
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
    direct_scores = direct_repair_edge_policy_scores(
        dataset=dataset,
        repair_edge_map=repair_map,
        bank_option_edges=bank_edges,
        bank_edge_segments=bank_segments,
        model_path=args.model_path,
        config=direct_config,
    )
    transfer_repair = transfer_cert[transfer_cert.get("is_repair_edge", False).astype(bool)].copy()
    transfer_cols = transfer_repair[["edge_id", "edge_policy_support_score", "edge_proxy_score"]].rename(
        columns={
            "edge_policy_support_score": "transfer_edge_policy_support_score",
            "edge_proxy_score": "transfer_edge_proxy_score",
        }
    )
    direct_scores = direct_scores.merge(transfer_cols, on="edge_id", how="left")

    direct_cert, direct_planner_cert = apply_direct_policy_scores(
        transfer_cert,
        direct_scores,
        calibration_config=calibration_config,
        certification_source=direct_config.certification_source,
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
    path_metrics, summary = evaluate_direct_repair_policy_planning(
        augmented_edges=augmented_edges,
        pair_compatibility=pair_compatibility,
        path_queries=path_queries,
        direct_planner_certification=direct_planner_cert,
        planner_config=planner_config,
        methods=args.methods,
        max_queries=args.max_queries,
        seed=int(args.seed),
    )
    diagnostics = direct_vs_transfer_diagnostics(transfer_cert, direct_cert, direct_scores)
    payload = {
        "config": vars(args),
        "direct_config": direct_config.__dict__,
        "calibration_config": calibration_config.__dict__,
        "risk_config": risk_config.__dict__,
        "phase4e_run_dir": str(phase4e_dir),
        "phase4f_run_dir": str(phase4f_dir),
        "repair_bank_phase2_run_dir": str(bank_dir),
        "model_path": str(Path(args.model_path).expanduser()),
        "diagnostics": diagnostics,
        "method_metrics": summary.to_dict("records"),
        "note": "Direct repair policy MSE is offline supervised evidence, not rollout success.",
    }

    direct_scores.to_csv(out_dir / "direct_repair_edge_policy_scores.csv", index=False)
    direct_cert.to_csv(out_dir / "direct_repair_edge_certification.csv", index=False)
    direct_planner_cert.to_csv(out_dir / "planner_direct_repair_edge_certification.csv", index=False)
    path_metrics.to_csv(out_dir / "direct_repair_policy_planning_paths.csv", index=False)
    summary.to_csv(out_dir / "direct_repair_policy_planning_summary.csv", index=False)
    _write_json(out_dir / "phase4g_direct_repair_policy_summary.json", payload)
    _write_markdown(out_dir / "phase4g_direct_repair_policy_summary.md", payload)
    _plot_outputs(direct_scores, summary, out_dir)

    print(f"Wrote Phase 4G direct repair policy outputs to {out_dir}")
    print(pd.Series(diagnostics).to_string())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

