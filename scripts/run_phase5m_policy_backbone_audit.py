#!/usr/bin/env python
"""Audit reusable policy backbones for policy-grounded BARS validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase5m.backbone_audit import run_backbone_audit  # noqa: E402


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
    parser.add_argument("--env_name", default=None)
    parser.add_argument("--dataset_key", default=None)
    parser.add_argument("--gas_inventory_csv", default=None)
    parser.add_argument("--gas_success_csv", default=None)
    parser.add_argument("--phase3f_root", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--bars_model_path", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "env_name": "antmaze-large-stitch-v0",
        "dataset_key": "antmaze_large_stitch",
        "gas_inventory_csv": "runs_stage31_official_gas/wide_artifact_inventory_active/official_gas_ready_matrix.csv",
        "gas_success_csv": "runs_stage31_official_gas/wide_atlas_active/global/stage31_success_by_env.csv",
        "phase3f_root": "results/phase3f",
        "phase2_run_dir": "results/phase2/antmaze_large_stitch/core_plus_bottleneck_budget120_H10",
        "bars_model_path": "results/phase3/antmaze_large_stitch/core_plus_bottleneck/model.pt",
        "output_dir": "results/phase5m/policy_backbone_audit/antmaze_large_stitch",
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    return argparse.Namespace(**merged)


def main() -> None:
    args = parse_args()
    result = run_backbone_audit(
        gas_inventory_csv=args.gas_inventory_csv,
        gas_success_csv=args.gas_success_csv,
        phase3f_root=args.phase3f_root,
        dataset_key=args.dataset_key,
        env_name=args.env_name,
        output_dir=args.output_dir,
        phase2_run_dir=args.phase2_run_dir,
        bars_model_path=args.bars_model_path,
    )
    summary = result["summary"]
    print(
        "[phase5m] wrote policy-backbone audit under "
        f"{result['output_dir']} ready_gas_backbones={summary['num_ready_gas_backbones']} "
        f"official_gas_success_rate={summary['official_gas_success_rate']}"
    )


if __name__ == "__main__":
    main()

