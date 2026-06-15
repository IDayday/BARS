#!/usr/bin/env python
"""Audit whether BARS graph targets are compatible with GAS target space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase5m.target_distribution import run_target_distribution_audit  # noqa: E402


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--option_edges_csv", default=None)
    parser.add_argument("--edge_segments_npz", default=None)
    parser.add_argument("--gas_dataset_embeddings_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--max_reference", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "option_edges_csv": "results/phase2/antmaze_large_stitch/core_plus_bottleneck_budget120_H10/option_edges.csv",
        "edge_segments_npz": "results/phase2/antmaze_large_stitch/core_plus_bottleneck_budget120_H10/edge_segments.npz",
        "gas_dataset_embeddings_path": (
            "artifacts/gas_ogbench_offline_full_20260522_165138/"
            "antmaze-large-stitch-v0/seed44/features/dataset_embeddings.npy"
        ),
        "output_dir": "results/phase5m/target_distribution_audit/antmaze_large_stitch_seed44",
        "max_targets": 50000,
        "max_reference": 200000,
        "seed": 0,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    return argparse.Namespace(**merged)


def main() -> None:
    args = parse_args()
    result = run_target_distribution_audit(
        option_edges_csv=args.option_edges_csv,
        edge_segments_npz=args.edge_segments_npz,
        gas_dataset_embeddings_path=args.gas_dataset_embeddings_path,
        output_dir=args.output_dir,
        max_targets=args.max_targets,
        max_reference=args.max_reference,
        seed=args.seed,
    )
    print(
        f"[phase5m] wrote target-distribution audit under {result['output_dir']} "
        f"status={result['summary']['status']}"
    )


if __name__ == "__main__":
    main()

