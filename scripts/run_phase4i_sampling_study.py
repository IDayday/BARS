#!/usr/bin/env python
"""Run Phase 4I stronger GCBC sampling study."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4i_sampling_study import (  # noqa: E402
    aggregate_sampling_tradeoffs,
    build_phase4i_summary_payload,
    compare_to_baseline,
    load_sampling_rows,
    recommend_sampling_modes,
    write_phase4i_outputs,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=None)
    return parser.parse_args()


def _run_ablation(config_path: Path, cuda_visible_devices: str | None, force_train: bool) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_phase3_sampling_ablation.py"), "--config", str(config_path)]
    if force_train:
        cmd.append("--no_resume")
    env = os.environ.copy()
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path)
    dataset_name = str(config["dataset_name"])
    output_root = Path(str(config.get("output_dir", "results/phase4i_sampling"))).expanduser()
    dataset_metrics_path = output_root / _dataset_key(dataset_name) / "all_per_seed_metrics.csv"
    summary_output_dir = Path(
        str(config.get("summary_output_dir", Path("results/phase4i") / _dataset_key(dataset_name)))
    ).expanduser()

    cuda_visible_devices = args.cuda_visible_devices
    if cuda_visible_devices is None:
        cuda_visible_devices = config.get("cuda_visible_devices")
    if not args.summarize_only:
        _run_ablation(config_path, cuda_visible_devices, force_train=args.force_train)
    if not dataset_metrics_path.exists():
        raise FileNotFoundError(f"Expected sampling metrics at {dataset_metrics_path}")

    rows = load_sampling_rows(dataset_metrics_path)
    aggregate = aggregate_sampling_tradeoffs(rows)
    comparisons = compare_to_baseline(
        aggregate,
        baseline_sampling_mode=str(config.get("baseline_sampling_mode", "uniform_transition")),
    )
    recommendations = recommend_sampling_modes(
        comparisons,
        max_overall_regret=float(config.get("max_overall_regret", 0.05)),
    )
    payload = build_phase4i_summary_payload(
        config={**config, "config_path": str(config_path)},
        aggregate=aggregate,
        comparisons=comparisons,
        recommendations=recommendations,
    )
    write_phase4i_outputs(summary_output_dir, payload, aggregate, comparisons, recommendations)
    print(f"[phase4i] wrote sampling study summary under {summary_output_dir}")
    if not recommendations.empty:
        print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
