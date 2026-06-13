#!/usr/bin/env python
"""Run Phase 3D offline GCBC sampling ablations.

This script trains state-based GCBC policies on Phase 2 option-edge segments
with different supervised sampling schemes. It does not construct environments
or run closed-loop rollout.
"""

from __future__ import annotations

import argparse
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
from phase3.train_gcbc import train_gcbc, write_json  # noqa: E402


SUMMARY_COLUMNS = [
    "dataset",
    "phase2_run",
    "sampling_mode",
    "seed",
    "final_val_action_mse",
    "best_val_action_mse",
    "bottleneck_edge_val_mse",
    "non_bottleneck_edge_val_mse",
    "high_support_edge_val_mse",
    "low_support_edge_val_mse",
    "short_horizon_edge_val_mse",
    "long_horizon_edge_val_mse",
]


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _run_name(path: str | Path) -> str:
    return Path(path).expanduser().name


def _parse_list(value: Any, cast=str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [cast(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _parse_hidden_dims(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.replace("[", "").replace("]", "").split(",") if x.strip()]


def _merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "output_dir": "results/phase3_sampling",
        "max_transitions": None,
        "phase2_run_dirs": [],
        "sampling_modes": ["uniform_transition", "uniform_edge", "bottleneck_weighted"],
        "seeds": [0],
        "num_steps": 1000,
        "batch_size": 1024,
        "lr": 3e-4,
        "hidden_dims": [512, 512, 512],
        "val_fraction": 0.05,
        "val_examples": 8192,
        "log_interval": None,
        "device": None,
        "edge_embedding_dim": 0,
        "no_resume": False,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    if not merged.get("dataset_name"):
        raise ValueError("--dataset_name is required")
    merged["phase2_run_dirs"] = _parse_list(merged.get("phase2_run_dirs"), str)
    if not merged["phase2_run_dirs"]:
        raise ValueError("phase2_run_dirs must contain at least one Phase 2 run directory")
    merged["sampling_modes"] = _parse_list(merged.get("sampling_modes"), str)
    merged["seeds"] = _parse_list(merged.get("seeds"), int)
    merged["hidden_dims"] = _parse_hidden_dims(merged.get("hidden_dims"))
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--phase2_run_dirs", default=None)
    parser.add_argument("--sampling_modes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dims", default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--val_examples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--edge_embedding_dim", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true", default=None)
    return _merge_args(parser.parse_args())


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
    if val.empty:
        return {column: np.nan for column in SUMMARY_COLUMNS if column.endswith("_mse")}
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


def _aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in SUMMARY_COLUMNS if col.endswith("_mse")]
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(["dataset", "phase2_run", "sampling_mode"], sort=True)[metric_cols]
    mean = grouped.mean().reset_index()
    std = grouped.std(ddof=0).reset_index()
    out = mean.copy()
    for col in metric_cols:
        out[f"{col}_std"] = std[col]
    out["num_seeds"] = grouped.size().to_numpy(dtype=np.int64)
    return out


def _grouped_by_sampling(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in SUMMARY_COLUMNS if col.endswith("_mse")]
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby(["sampling_mode"], sort=True)[metric_cols].mean().reset_index()


def _plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["sampling_mode"], summary["final_val_action_mse"])
    ax.set_ylabel("final val action MSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "final_val_mse_by_sampling.png", dpi=160)
    plt.close(fig)

    cols = [
        "bottleneck_edge_val_mse",
        "low_support_edge_val_mse",
        "long_horizon_edge_val_mse",
    ]
    x = np.arange(summary.shape[0])
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, col in enumerate(cols):
        ax.bar(x + (offset - 1) * width, summary[col], width=width, label=col.replace("_edge_val_mse", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(summary["sampling_mode"], rotation=20)
    ax.set_ylabel("edge val action MSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "edge_group_mse_by_sampling.png", dpi=160)
    plt.close(fig)


def _write_markdown(dataset_dir: Path, dataset_name: str, all_rows: pd.DataFrame) -> None:
    lines = [
        "# Phase 3D Sampling Ablation Summary",
        "",
        "This is an offline supervised GCBC sampling ablation. It compares action-prediction",
        "MSE under different edge-sampling schemes and does not measure closed-loop rollout",
        "success.",
        "",
        "Phase 3C closed-loop execution remains gated on a reliable reset-to-state probe.",
        "Current `env_unavailable` reset probes are environment dependency blockers, not",
        "evidence that the benchmark environments themselves lack reset support.",
        "",
        f"Dataset: `{dataset_name}`",
        "",
    ]
    if all_rows.empty:
        lines.append("No ablation rows were generated.")
    else:
        best = all_rows.sort_values("final_val_action_mse", kind="mergesort").head(5)
        lines.append("Top rows by final validation action MSE:")
        lines.append("")
        cols = SUMMARY_COLUMNS
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in best[cols].to_dict("records"):
            values = []
            for col in cols:
                value = row.get(col)
                if isinstance(value, float):
                    values.append("" if not np.isfinite(value) else f"{value:.6g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    path = dataset_dir / "phase3_sampling_ablation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    dataset_dir = Path(args.output_dir) / _dataset_key(args.dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for phase2_run_dir in args.phase2_run_dirs:
        phase2_path = Path(phase2_run_dir).expanduser()
        run_name = _run_name(phase2_path)
        run_dir = dataset_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        option_edges = phase2_path / "option_edges.csv"
        edge_segments = phase2_path / "edge_segments.npz"
        if not option_edges.exists() or not edge_segments.exists():
            raise FileNotFoundError(f"Missing Phase 2 edge artifacts under {phase2_path}")

        run_rows: list[dict[str, Any]] = []
        for sampling_mode in args.sampling_modes:
            if sampling_mode not in {"uniform_transition", "uniform_edge", "bottleneck_weighted"}:
                raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")
            for seed in args.seeds:
                train_dir = run_dir / f"{sampling_mode}_seed{int(seed)}"
                metrics = None if args.no_resume else _read_completed(train_dir)
                if metrics is None:
                    print(
                        f"[phase3D] training dataset={args.dataset_name} run={run_name} "
                        f"sampling={sampling_mode} seed={seed}"
                    )
                    result = train_gcbc(
                        dataset=dataset,
                        option_edges_csv=option_edges,
                        edge_segments_npz=edge_segments,
                        output_dir=train_dir,
                        sampling_mode=sampling_mode,
                        batch_size=args.batch_size,
                        num_steps=args.num_steps,
                        lr=args.lr,
                        hidden_dims=args.hidden_dims,
                        seed=int(seed),
                        val_fraction=args.val_fraction,
                        val_examples=args.val_examples,
                        log_interval=args.log_interval,
                        device=args.device,
                        edge_embedding_dim=args.edge_embedding_dim,
                        config={**vars(args), "phase2_run_dir": str(phase2_path)},
                    )
                    metrics = _metrics_from_result(result)
                row = {
                    "dataset": args.dataset_name,
                    "phase2_run": run_name,
                    "phase2_run_dir": str(phase2_path),
                    "sampling_mode": sampling_mode,
                    "seed": int(seed),
                    **metrics,
                }
                run_rows.append(row)
                all_rows.append(row)

        per_seed = pd.DataFrame(run_rows)
        per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
        summary = _aggregate(per_seed)
        summary.to_csv(run_dir / "sampling_ablation_summary.csv", index=False)
        grouped = _grouped_by_sampling(per_seed)
        grouped.to_csv(run_dir / "grouped_val_metrics_by_sampling.csv", index=False)
        _plot_summary(grouped, run_dir)
        write_json(
            run_dir / "sampling_ablation_config.json",
            {
                "dataset_name": args.dataset_name,
                "phase2_run_dir": str(phase2_path),
                "sampling_modes": args.sampling_modes,
                "seeds": args.seeds,
                "num_steps": args.num_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dims": args.hidden_dims,
            },
        )

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(dataset_dir / "all_per_seed_metrics.csv", index=False)
    _aggregate(all_df).to_csv(dataset_dir / "all_sampling_ablation_summary.csv", index=False)
    _grouped_by_sampling(all_df).to_csv(dataset_dir / "all_grouped_val_metrics_by_sampling.csv", index=False)
    _write_markdown(dataset_dir, args.dataset_name, all_df)
    print(f"[phase3D] wrote sampling ablation outputs under {dataset_dir}")


if __name__ == "__main__":
    main()
