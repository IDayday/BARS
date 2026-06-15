#!/usr/bin/env python
"""Calibrate Phase 5I state-conditioned outcome risk on heldout attempts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3f.state_outcome_calibration import (  # noqa: E402
    calibrate_state_outcome_model,
    load_attempt_examples_with_sources,
)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _parse_list(value: Any, cast=str) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [cast(x) for x in value]
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--trace_dirs", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--min_examples", type=int, default=None)
    parser.add_argument("--l2", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--penalty_weights", default=None)
    parser.add_argument("--max_mean_penalty", type=float, default=None)
    parser.add_argument("--max_completed_mean_penalty", type=float, default=None)
    return parser.parse_args()


def merge_args(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "dataset_name": "antmaze-large-stitch-v0",
        "trace_dirs": [],
        "output_dir": "results/phase3f/state_outcome_calibration",
        "seed": 0,
        "val_fraction": 0.3,
        "min_examples": 8,
        "l2": 1.0,
        "learning_rate": 0.05,
        "num_steps": 800,
        "feature_columns": None,
        "penalty_weights": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        "max_mean_penalty": 0.5,
        "max_completed_mean_penalty": 0.5,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    merged["trace_dirs"] = _parse_list(merged["trace_dirs"], str)
    merged["penalty_weights"] = _parse_list(merged["penalty_weights"], float)
    if merged.get("feature_columns") is not None:
        merged["feature_columns"] = _parse_list(merged["feature_columns"], str)
    if not merged["trace_dirs"]:
        raise ValueError("trace_dirs must contain at least one rollout result dir or episode_traces.jsonl")
    return argparse.Namespace(**merged)


def _write_markdown(path: Path, *, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Phase 5J State-Outcome Calibration",
                "",
                "This is a heldout attempt-level calibration check for the Phase 5I",
                "state-conditioned outcome model. It does not claim task success.",
                "",
                "## Summary",
                "",
                f"- dataset: `{args.dataset_name}`",
                f"- num examples: `{summary['num_examples']}`",
                f"- train examples: `{summary['num_train_examples']}`",
                f"- val examples: `{summary['num_val_examples']}`",
                f"- selected penalty weight: `{summary['selected_penalty_weight']}`",
                f"- val Brier: `{summary['val_brier']}`",
                f"- val AUC: `{summary['val_auc']}`",
                f"- val risk separation: `{summary['val_risk_separation']}`",
                "",
                "The selected weight is constrained by validation mean-penalty budgets;",
                "it is a conservative planner-cost scale, not a calibrated success",
                "probability threshold.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = merge_args(parse_args())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_attempt_examples_with_sources(args.trace_dirs)
    if examples.empty:
        raise RuntimeError("No attempt examples could be extracted from trace_dirs")
    result = calibrate_state_outcome_model(
        examples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_examples=args.min_examples,
        feature_columns=args.feature_columns,
        l2=args.l2,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        penalty_weights=args.penalty_weights,
        max_mean_penalty=args.max_mean_penalty,
        max_completed_mean_penalty=args.max_completed_mean_penalty,
    )

    examples.to_csv(out_dir / "attempt_examples.csv", index=False)
    result["train_examples"].to_csv(out_dir / "train_attempt_examples.csv", index=False)
    result["val_examples"].to_csv(out_dir / "val_attempt_examples.csv", index=False)
    result["predictions"].to_csv(out_dir / "calibration_predictions.csv", index=False)
    result["metrics"].to_csv(out_dir / "calibration_metrics.csv", index=False)
    result["selection"].to_csv(out_dir / "penalty_weight_selection.csv", index=False)
    (out_dir / "state_conditioned_outcome_model_summary.json").write_text(
        json.dumps(result["model"].to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    val_metrics = result["metrics"].loc[result["metrics"]["split"] == "val"].iloc[0].to_dict()
    summary = {
        "dataset_name": args.dataset_name,
        "trace_dirs": args.trace_dirs,
        "num_examples": int(len(examples)),
        "num_train_examples": int(len(result["train_examples"])),
        "num_val_examples": int(len(result["val_examples"])),
        "num_trace_groups": int(examples["trace_group"].nunique()) if "trace_group" in examples.columns else 0,
        "selected_penalty_weight": float(result["selected_weight"]),
        "feature_columns": result["model"].feature_columns,
        "val_brier": float(val_metrics.get("brier", float("nan"))),
        "val_log_loss": float(val_metrics.get("log_loss", float("nan"))),
        "val_auc": float(val_metrics.get("auc", float("nan"))),
        "val_risk_separation": float(val_metrics.get("risk_separation", float("nan"))),
        "max_mean_penalty": float(args.max_mean_penalty),
        "max_completed_mean_penalty": float(args.max_completed_mean_penalty),
        "note": "Heldout attempt calibration is not task success.",
    }
    (out_dir / "phase5j_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(out_dir / "phase5j_calibration_summary.md", args=args, summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
