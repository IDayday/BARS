#!/usr/bin/env python
"""Run Phase 4J loss-weighted GCBC study."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data import load_ogbench_dataset  # noqa: E402
from phase3.train_gcbc import train_gcbc  # noqa: E402
from phase3e.phase4j_loss_weighting import (  # noqa: E402
    aggregate_loss_weighting_rows,
    build_phase4j_payload,
    compare_loss_weighting_to_baseline,
    recommend_loss_weighting_methods,
    write_phase4j_outputs,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _phase2_run_name(path: str | Path) -> str:
    return Path(path).expanduser().name


def _read_completed(train_dir: Path) -> dict[str, float] | None:
    val_path = train_dir / "val_metrics.csv"
    edge_path = train_dir / "edge_val_metrics.csv"
    if not val_path.exists() or not edge_path.exists():
        return None
    val = pd.read_csv(val_path)
    if val.empty:
        return None
    final = val.iloc[-1]
    return {
        "final_val_action_mse": float(final.get("val_action_mse", np.nan)),
        "best_val_action_mse": float(val["val_action_mse"].min()),
        "bottleneck_edge_val_mse": float(final.get("bottleneck_edge_val_mse", np.nan)),
        "non_bottleneck_edge_val_mse": float(final.get("non_bottleneck_edge_val_mse", np.nan)),
        "high_support_edge_val_mse": float(final.get("high_support_edge_val_mse", np.nan)),
        "low_support_edge_val_mse": float(final.get("low_support_edge_val_mse", np.nan)),
        "short_horizon_edge_val_mse": float(final.get("short_horizon_edge_val_mse", np.nan)),
        "long_horizon_edge_val_mse": float(final.get("long_horizon_edge_val_mse", np.nan)),
    }


def _metrics_from_result(result: dict[str, Any]) -> dict[str, float]:
    val = result["val_metrics"]
    final = val.iloc[-1]
    return {
        "final_val_action_mse": float(final.get("val_action_mse", np.nan)),
        "best_val_action_mse": float(val["val_action_mse"].min()),
        "bottleneck_edge_val_mse": float(final.get("bottleneck_edge_val_mse", np.nan)),
        "non_bottleneck_edge_val_mse": float(final.get("non_bottleneck_edge_val_mse", np.nan)),
        "high_support_edge_val_mse": float(final.get("high_support_edge_val_mse", np.nan)),
        "low_support_edge_val_mse": float(final.get("low_support_edge_val_mse", np.nan)),
        "short_horizon_edge_val_mse": float(final.get("short_horizon_edge_val_mse", np.nan)),
        "long_horizon_edge_val_mse": float(final.get("long_horizon_edge_val_mse", np.nan)),
    }


def _load_baseline_rows(config: dict[str, Any], phase2_run_name: str) -> list[dict[str, Any]]:
    baseline_csv = config.get("baseline_reference_csv")
    if not baseline_csv:
        return []
    df = pd.read_csv(Path(str(baseline_csv)).expanduser())
    sampling_mode = str(config.get("baseline_sampling_mode", "uniform_transition"))
    rows = df[(df["phase2_run"] == phase2_run_name) & (df["sampling_mode"] == sampling_mode)].copy()
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        out.append(
            {
                **row,
                "method": str(config.get("baseline_method", "uniform_transition_none")),
                "sampling_mode": sampling_mode,
                "loss_weight_mode": "none",
                "loss_weight_strength": 0.0,
                "loss_weight_min": 1.0,
                "loss_weight_max": 1.0,
                "source": "baseline_reference",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path)
    dataset_name = str(config["dataset_name"])
    phase2_run_dir = Path(str(config["phase2_run_dir"])).expanduser()
    phase2_run_name = _phase2_run_name(phase2_run_dir)
    output_root = Path(str(config.get("output_dir", "results/phase4j_weighted"))).expanduser()
    run_root = output_root / _dataset_key(dataset_name) / phase2_run_name
    summary_output_dir = Path(str(config.get("summary_output_dir", run_root))).expanduser()
    cuda_visible_devices = args.cuda_visible_devices
    if cuda_visible_devices is None:
        cuda_visible_devices = config.get("cuda_visible_devices")
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()

    dataset = load_ogbench_dataset(
        dataset_name,
        config.get("dataset_dir", "/mnt/project/offlinerl_datasets/ogbench"),
        split="train",
        max_transitions=config.get("max_transitions"),
    )
    option_edges = phase2_run_dir / "option_edges.csv"
    edge_segments = phase2_run_dir / "edge_segments.npz"
    if not option_edges.exists() or not edge_segments.exists():
        raise FileNotFoundError(f"Missing Phase 2 edge artifacts under {phase2_run_dir}")

    rows: list[dict[str, Any]] = _load_baseline_rows(config, phase2_run_name)
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("config must contain non-empty experiments list")
    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError("each experiment must be a mapping")
        method = str(exp["method"])
        sampling_mode = str(exp.get("sampling_mode", config.get("sampling_mode", "uniform_transition")))
        loss_weight_mode = str(exp.get("loss_weight_mode", "none"))
        loss_weight_strength = float(exp.get("loss_weight_strength", config.get("loss_weight_strength", 1.0)))
        loss_weight_min = float(exp.get("loss_weight_min", config.get("loss_weight_min", 0.25)))
        loss_weight_max = float(exp.get("loss_weight_max", config.get("loss_weight_max", 3.0)))
        for seed in [int(x) for x in config.get("seeds", [0])]:
            train_dir = run_root / f"{method}_seed{seed}"
            metrics = None if args.force_train else _read_completed(train_dir)
            if metrics is None:
                print(f"[phase4j] training method={method} seed={seed}")
                result = train_gcbc(
                    dataset=dataset,
                    option_edges_csv=option_edges,
                    edge_segments_npz=edge_segments,
                    output_dir=train_dir,
                    sampling_mode=sampling_mode,
                    batch_size=int(config.get("batch_size", 1024)),
                    num_steps=int(config.get("num_steps", 3000)),
                    lr=float(config.get("lr", 3e-4)),
                    hidden_dims=config.get("hidden_dims", [512, 512, 512]),
                    seed=seed,
                    val_fraction=float(config.get("val_fraction", 0.05)),
                    val_examples=int(config.get("val_examples", 8192)),
                    log_interval=config.get("log_interval"),
                    device=config.get("device"),
                    edge_embedding_dim=int(config.get("edge_embedding_dim", 0)),
                    loss_weight_mode=loss_weight_mode,
                    loss_weight_strength=loss_weight_strength,
                    loss_weight_min=loss_weight_min,
                    loss_weight_max=loss_weight_max,
                    config={**config, "phase2_run_dir": str(phase2_run_dir), "experiment": exp},
                )
                metrics = _metrics_from_result(result)
            rows.append(
                {
                    "dataset": dataset_name,
                    "phase2_run": phase2_run_name,
                    "phase2_run_dir": str(phase2_run_dir),
                    "method": method,
                    "sampling_mode": sampling_mode,
                    "loss_weight_mode": loss_weight_mode,
                    "loss_weight_strength": loss_weight_strength,
                    "loss_weight_min": loss_weight_min,
                    "loss_weight_max": loss_weight_max,
                    "seed": seed,
                    "source": "trained",
                    **metrics,
                }
            )

    all_rows = pd.DataFrame(rows)
    aggregate = aggregate_loss_weighting_rows(all_rows)
    comparisons = compare_loss_weighting_to_baseline(
        aggregate,
        baseline_method=str(config.get("baseline_method", "uniform_transition_none")),
    )
    recommendations = recommend_loss_weighting_methods(
        comparisons,
        max_overall_regret=float(config.get("max_overall_regret", 0.05)),
    )
    payload = build_phase4j_payload(
        config={**config, "config_path": str(config_path)},
        aggregate=aggregate,
        comparisons=comparisons,
        recommendations=recommendations,
    )
    write_phase4j_outputs(summary_output_dir, payload, all_rows, aggregate, comparisons, recommendations)
    print(f"[phase4j] wrote loss-weighting study summary under {summary_output_dir}")
    if not recommendations.empty:
        print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
