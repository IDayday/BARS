#!/usr/bin/env python
"""Run Phase 4H Scene GCBC training and direct repair validation comparison."""

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

from phase3e.phase4h_scene_validation import (  # noqa: E402
    build_phase4h_summary,
    diagnostics_delta_frame,
    load_json,
    load_training_summary,
    method_delta_frame,
    write_phase4h_outputs,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _phase4g_summary_from_config(config_path: str | Path) -> Path:
    config = _load_yaml(config_path)
    dataset_name = str(config["dataset_name"])
    out_root = Path(config.get("output_dir", "results/phase4g")).expanduser()
    phase4e_name = Path(str(config["phase4e_run_dir"])).name
    return out_root / _dataset_key(dataset_name) / phase4e_name / "phase4g_direct_repair_policy_summary.json"


def _training_output_from_config(config_path: str | Path) -> Path:
    config = _load_yaml(config_path)
    if "output_dir" not in config:
        raise ValueError(f"{config_path} must set output_dir for Phase 4H")
    return Path(str(config["output_dir"])).expanduser()


def _run(cmd: list[str], cuda_visible_devices: str | None = None) -> None:
    env = os.environ.copy()
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_direct_repair", action="store_true")
    parser.add_argument("--force_training", action="store_true")
    parser.add_argument("--force_direct_repair", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = _load_yaml(config_path)
    train_config = Path(str(config["train_config"])).expanduser()
    direct_config = Path(str(config["direct_repair_config"])).expanduser()
    output_dir = Path(str(config["output_dir"])).expanduser()
    baseline_summary_path = Path(str(config["baseline_phase4g_summary"])).expanduser()
    candidate_summary_path = Path(
        str(config.get("candidate_phase4g_summary") or _phase4g_summary_from_config(direct_config))
    ).expanduser()
    train_output_dir = _training_output_from_config(train_config)
    model_path = Path(str(config.get("model_path") or (train_output_dir / "model.pt"))).expanduser()
    cuda_visible_devices = args.cuda_visible_devices
    if cuda_visible_devices is None:
        cuda_visible_devices = config.get("cuda_visible_devices")

    run_training = bool(config.get("run_training", True)) and not args.skip_training and not args.summarize_only
    run_direct = bool(config.get("run_direct_repair", True)) and not args.skip_direct_repair and not args.summarize_only
    skip_training_if_model_exists = bool(config.get("skip_training_if_model_exists", True)) and not args.force_training
    skip_direct_if_summary_exists = bool(config.get("skip_direct_if_summary_exists", True)) and not args.force_direct_repair

    if run_training:
        if skip_training_if_model_exists and model_path.exists():
            print(f"[phase4h] skip training; model already exists: {model_path}")
        else:
            _run(
                [sys.executable, str(ROOT / "scripts" / "train_phase3_gcbc.py"), "--config", str(train_config)],
                cuda_visible_devices=cuda_visible_devices,
            )
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model at {model_path}")

    if run_direct:
        if skip_direct_if_summary_exists and candidate_summary_path.exists():
            print(f"[phase4h] skip direct repair; summary already exists: {candidate_summary_path}")
        else:
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_phase4g_direct_repair_policy.py"),
                    "--config",
                    str(direct_config),
                ],
                cuda_visible_devices=cuda_visible_devices,
            )
    if not candidate_summary_path.exists():
        raise FileNotFoundError(f"Expected direct repair summary at {candidate_summary_path}")

    baseline_summary = load_json(baseline_summary_path)
    candidate_summary = load_json(candidate_summary_path)
    training_summary = load_training_summary(train_output_dir)
    baseline_label = str(config.get("baseline_label", "phase4g_baseline"))
    candidate_label = str(config.get("candidate_label", "phase4h_candidate"))
    diag_delta = diagnostics_delta_frame(baseline_summary, candidate_summary, baseline_label, candidate_label)
    method_delta = method_delta_frame(baseline_summary, candidate_summary, baseline_label, candidate_label)
    payload = build_phase4h_summary(
        config={**config, "config_path": str(config_path)},
        training_summary=training_summary,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        diagnostic_deltas=diag_delta,
        method_deltas=method_delta,
    )
    write_phase4h_outputs(output_dir, payload, diag_delta, method_delta)
    print(f"[phase4h] wrote summary outputs under {output_dir}")
    if not diag_delta.empty:
        print(diag_delta.to_string(index=False))
    if not method_delta.empty:
        selected = method_delta[method_delta["metric"].isin(["path_coverage", "mean_uncertified_edge_fraction"])]
        print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
