#!/usr/bin/env python
"""Run Phase 4M planner-relevant repair loss-weighting study."""

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
from phase3.train_gcbc import edge_loss_weight_values, train_gcbc  # noqa: E402
from phase3e.compatibility_graph_repair import build_augmented_graph_inputs  # noqa: E402
from phase3e.direct_repair_policy import DirectRepairPolicyConfig, direct_repair_edge_policy_scores  # noqa: E402
from phase3e.phase4m_planner_relevant_weighting import (  # noqa: E402
    aggregate_phase4m_training_rows,
    build_planner_relevant_loss_weights,
    compare_phase4m_to_baseline,
    grouped_direct_repair_metrics,
    summarize_direct_repair_scores,
    summarize_loss_weights,
    write_phase4m_outputs,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser()) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _read_completed(train_dir: Path) -> dict[str, float] | None:
    val_path = train_dir / "val_metrics.csv"
    edge_path = train_dir / "edge_val_metrics.csv"
    model_path = train_dir / "model.pt"
    if not val_path.exists() or not edge_path.exists() or not model_path.exists():
        return None
    val = pd.read_csv(val_path)
    if val.empty:
        return None
    final = val.iloc[-1]
    return _metrics_from_val_df(val, final)


def _metrics_from_val_df(val: pd.DataFrame, final: pd.Series | None = None) -> dict[str, float]:
    final = val.iloc[-1] if final is None else final
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
    return _metrics_from_val_df(result["val_metrics"])


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _prepare_augmented_inputs(config: dict[str, Any], artifact_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    phase4e_dir = Path(str(config["phase4e_run_dir"])).expanduser()
    base_dir = Path(str(config["base_phase2_run_dir"])).expanduser()
    bank_dir = Path(str(config["repair_bank_phase2_run_dir"])).expanduser()
    base_edges = pd.read_csv(base_dir / "option_edges.csv")
    bank_edges = pd.read_csv(bank_dir / "option_edges.csv")
    selected_repair_edges = pd.read_csv(phase4e_dir / "selected_repair_edges.csv")
    base_segments = _load_npz(base_dir / "edge_segments.npz")
    bank_segments = _load_npz(bank_dir / "edge_segments.npz")
    augmented_edges, augmented_segments, repair_map = build_augmented_graph_inputs(
        base_edges,
        base_segments,
        selected_repair_edges,
        bank_segments,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    augmented_edges.to_csv(artifact_dir / "augmented_option_edges.csv", index=False)
    repair_map.to_csv(artifact_dir / "repair_edge_map.csv", index=False)
    _save_npz(artifact_dir / "augmented_edge_segments.npz", augmented_segments)
    return augmented_edges, augmented_segments, repair_map


def _experiment_loss_weights(
    exp: dict[str, Any],
    augmented_edges: pd.DataFrame,
    path_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame | None:
    mode = str(exp.get("loss_weight_mode", "none"))
    if mode == "planner_relevant":
        return build_planner_relevant_loss_weights(
            augmented_edges,
            path_metrics=path_metrics,
            planner_method=str(config.get("planner_method", "calibrated_compat_threshold")),
            graph_variant=str(config.get("graph_variant", "repaired")),
            base_loss_weight_mode=str(exp.get("base_loss_weight_mode", "support_bottleneck")),
            base_loss_weight_strength=float(exp.get("base_loss_weight_strength", 0.3)),
            base_loss_weight_min=float(exp.get("base_loss_weight_min", 0.7)),
            base_loss_weight_max=float(exp.get("base_loss_weight_max", 1.8)),
            planner_relevance_strength=float(exp.get("planner_relevance_strength", 0.5)),
            hard_repair_strength=float(exp.get("hard_repair_strength", 0.25)),
            repair_only_planner_relevance=bool(exp.get("repair_only_planner_relevance", True)),
            min_weight=float(exp.get("loss_weight_min", config.get("loss_weight_min", 0.7))),
            max_weight=float(exp.get("loss_weight_max", config.get("loss_weight_max", 2.2))),
        )
    if mode in {"support", "bottleneck", "support_bottleneck", "external"}:
        weights = edge_loss_weight_values(
            augmented_edges,
            mode="none" if mode == "external" else mode,
            strength=float(exp.get("loss_weight_strength", config.get("loss_weight_strength", 1.0))),
            min_weight=float(exp.get("loss_weight_min", config.get("loss_weight_min", 0.25))),
            max_weight=float(exp.get("loss_weight_max", config.get("loss_weight_max", 3.0))),
        )
        weights["base_loss_weight"] = weights["loss_weight"]
        weights["external_multiplier"] = 1.0
        weights["planner_usage_count"] = 0
        weights["planner_relevance_score"] = 0.0
        weights["hard_repair_score"] = 0.0
        repair_lookup = augmented_edges.copy()
        if "is_repair_edge" not in repair_lookup.columns:
            repair_lookup["is_repair_edge"] = False
        weights["is_repair_edge"] = (
            repair_lookup.set_index("edge_id").reindex(weights["edge_id"])["is_repair_edge"].fillna(False).to_numpy()
        )
        weights["loss_weight_reason"] = mode
        return weights
    return None


def _scalar_direct_metrics(group_metrics: pd.DataFrame) -> dict[str, float]:
    out = {
        "direct_repair_edge_mse": np.nan,
        "planner_used_repair_edge_mse": np.nan,
        "not_planner_used_repair_edge_mse": np.nan,
        "direct_repair_policy_support_score": np.nan,
    }
    if group_metrics.empty:
        return out

    def _row(group_type: str, group_value: str) -> pd.Series | None:
        sub = group_metrics[
            (group_metrics["group_type"] == group_type) & (group_metrics["group_value"] == group_value)
        ]
        return None if sub.empty else sub.iloc[0]

    all_row = _row("all_repair_edges", "all_repair_edges")
    used_row = _row("planner_usage_group", "planner_used")
    unused_row = _row("planner_usage_group", "not_planner_used")
    if all_row is not None:
        out["direct_repair_edge_mse"] = float(all_row.get("mean_edge_action_mse", np.nan))
        out["direct_repair_policy_support_score"] = float(all_row.get("mean_direct_policy_support_score", np.nan))
    if used_row is not None:
        out["planner_used_repair_edge_mse"] = float(used_row.get("mean_edge_action_mse", np.nan))
    if unused_row is not None:
        out["not_planner_used_repair_edge_mse"] = float(unused_row.get("mean_edge_action_mse", np.nan))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = _load_yaml(config_path)
    dataset_name = str(config["dataset_name"])
    dataset_key = _dataset_key(dataset_name)
    phase2_run = str(config.get("phase2_run", Path(str(config["base_phase2_run_dir"])).name))
    run_name = str(config.get("run_name", f"{phase2_run}_planner_relevant"))
    output_root = Path(str(config.get("output_dir", "results/phase4m_training"))).expanduser()
    summary_output_dir = Path(str(config.get("summary_output_dir", "results/phase4m"))).expanduser()
    train_root = output_root / dataset_key / run_name
    summary_dir = summary_output_dir / dataset_key / run_name
    artifact_dir = summary_dir / "artifacts"
    cuda_visible_devices = args.cuda_visible_devices if args.cuda_visible_devices is not None else config.get("cuda_visible_devices")
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()

    augmented_edges, augmented_segments, repair_map = _prepare_augmented_inputs(config, artifact_dir)
    path_metrics_csv = Path(str(config.get("path_metrics_csv", Path(str(config["phase4e_run_dir"])) / "repair_planning_paths.csv"))).expanduser()
    path_metrics = pd.read_csv(path_metrics_csv)
    bank_dir = Path(str(config["repair_bank_phase2_run_dir"])).expanduser()
    bank_edges = pd.read_csv(bank_dir / "option_edges.csv")
    bank_segments = _load_npz(bank_dir / "edge_segments.npz")
    dataset = load_ogbench_dataset(
        dataset_name,
        config.get("dataset_dir", "/mnt/project/offlinerl_datasets/ogbench"),
        split="train",
        max_transitions=config.get("max_transitions"),
    )

    experiments = config.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("config must contain a non-empty experiments list")
    seeds = [int(x) for x in config.get("seeds", [0])]
    per_seed_rows: list[dict[str, Any]] = []
    all_group_rows: list[pd.DataFrame] = []
    weight_summaries: list[pd.DataFrame] = []

    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError("each experiment must be a mapping")
        method = str(exp["method"])
        sampling_mode = str(exp.get("sampling_mode", config.get("sampling_mode", "uniform_transition")))
        exp_mode = str(exp.get("loss_weight_mode", "none"))
        train_loss_mode = "external" if exp_mode == "planner_relevant" else exp_mode
        weights = _experiment_loss_weights(exp, augmented_edges, path_metrics, config)
        if weights is not None:
            weight_path = artifact_dir / f"{method}_loss_weights.csv"
            weights.to_csv(weight_path, index=False)
            ws = summarize_loss_weights(augmented_edges, weights)
            ws["method"] = method
            weight_summaries.append(ws)
        else:
            weight_path = None

        for seed in seeds:
            train_dir = train_root / f"{method}_seed{seed}"
            metrics = None if args.force_train else _read_completed(train_dir)
            if metrics is None:
                print(f"[phase4m] training method={method} seed={seed}")
                result = train_gcbc(
                    dataset=dataset,
                    option_edges_csv=augmented_edges,
                    edge_segments_npz=augmented_segments,
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
                    loss_weight_mode=train_loss_mode,
                    loss_weight_strength=float(exp.get("loss_weight_strength", config.get("loss_weight_strength", 1.0))),
                    loss_weight_min=float(exp.get("loss_weight_min", config.get("loss_weight_min", 0.25))),
                    loss_weight_max=float(exp.get("loss_weight_max", config.get("loss_weight_max", 3.0))),
                    external_loss_weights=weights if train_loss_mode == "external" else None,
                    external_loss_weight_column="loss_weight",
                    external_loss_weight_combine="replace",
                    config={**config, "config_path": str(config_path), "experiment": exp},
                )
                metrics = _metrics_from_result(result)

            direct_scores_path = train_dir / "phase4m_direct_repair_edge_policy_scores.csv"
            if args.force_eval or not direct_scores_path.exists():
                print(f"[phase4m] direct repair scoring method={method} seed={seed}")
                direct_scores = direct_repair_edge_policy_scores(
                    dataset=dataset,
                    repair_edge_map=repair_map,
                    bank_option_edges=bank_edges,
                    bank_edge_segments=bank_segments,
                    model_path=str(train_dir / "model.pt"),
                    config=DirectRepairPolicyConfig(
                        batch_size=int(config.get("direct_batch_size", 2048)),
                        max_examples=int(config.get("direct_max_examples", 50000)),
                        temperature=float(config.get("direct_temperature", 0.05)),
                        seed=seed,
                        device=config.get("device"),
                    ),
                )
                direct_scores.to_csv(direct_scores_path, index=False)
            else:
                direct_scores = pd.read_csv(direct_scores_path)
            grouped_scores = summarize_direct_repair_scores(
                direct_scores,
                augmented_edges,
                path_metrics,
                planner_method=str(config.get("planner_method", "calibrated_compat_threshold")),
                graph_variant=str(config.get("graph_variant", "repaired")),
            )
            grouped_scores.to_csv(train_dir / "phase4m_direct_repair_edge_scores_grouped.csv", index=False)
            group_metrics = grouped_direct_repair_metrics(grouped_scores)
            group_metrics["dataset"] = dataset_name
            group_metrics["phase2_run"] = phase2_run
            group_metrics["method"] = method
            group_metrics["seed"] = seed
            group_metrics.to_csv(train_dir / "phase4m_direct_repair_group_metrics.csv", index=False)
            all_group_rows.append(group_metrics)
            per_seed_rows.append(
                {
                    "dataset": dataset_name,
                    "phase2_run": phase2_run,
                    "method": method,
                    "seed": seed,
                    "sampling_mode": sampling_mode,
                    "loss_weight_mode": exp_mode,
                    "loss_weight_strength": float(exp.get("loss_weight_strength", config.get("loss_weight_strength", 0.0))),
                    "loss_weight_min": float(exp.get("loss_weight_min", config.get("loss_weight_min", 1.0))),
                    "loss_weight_max": float(exp.get("loss_weight_max", config.get("loss_weight_max", 1.0))),
                    "training_run_dir": str(train_dir),
                    "model_path": str(train_dir / "model.pt"),
                    **metrics,
                    **_scalar_direct_metrics(group_metrics),
                }
            )

    per_seed = pd.DataFrame(per_seed_rows)
    aggregate = aggregate_phase4m_training_rows(per_seed)
    comparisons = compare_phase4m_to_baseline(
        aggregate,
        baseline_method=str(config.get("baseline_method", "augmented_loss_support_bottleneck_s03")),
    )
    weight_summary = pd.concat(weight_summaries, ignore_index=True) if weight_summaries else pd.DataFrame()
    direct_group_metrics = pd.concat(all_group_rows, ignore_index=True) if all_group_rows else pd.DataFrame()
    payload = {
        "phase": "Phase 4M",
        "title": "Planner-Relevant Repair Loss Weighting",
        "config": {**config, "config_path": str(config_path)},
        "phase4e_run_dir": str(Path(str(config["phase4e_run_dir"])).expanduser()),
        "path_metrics_csv": str(path_metrics_csv),
        "per_seed_metrics": per_seed.to_dict("records"),
        "aggregate_metrics": aggregate.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
        "weight_summary": weight_summary.to_dict("records"),
        "related_work_reviewed": [
            {
                "name": "Goal-Conditioned Supervised Learning",
                "url": "https://arxiv.org/abs/1912.06088",
                "role": "Goal-conditioned supervised policy training framing.",
            },
            {
                "name": "RvS: What is Essential for Offline RL via Supervised Learning?",
                "url": "https://arxiv.org/abs/2112.10751",
                "role": "Supervised offline RL claim-boundary reference.",
            },
            {
                "name": "Prioritized Experience Replay",
                "url": "https://arxiv.org/abs/1511.05952",
                "role": "Priority signal inspiration; Phase 4M uses supervised loss weights, not TD-error replay.",
            },
            {
                "name": "Class-Balanced Loss Based on Effective Number of Samples",
                "url": "https://arxiv.org/abs/1901.05555",
                "role": "Long-tail weighting motivation.",
            },
        ],
        "note": "Offline supervised loss weighting only; no environment rollout and no online success claim.",
    }
    write_phase4m_outputs(summary_dir, payload, per_seed, aggregate, comparisons, weight_summary, direct_group_metrics)
    print(f"[phase4m] wrote outputs under {summary_dir}")
    if not comparisons.empty:
        print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
