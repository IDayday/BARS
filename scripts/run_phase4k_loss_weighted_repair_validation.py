#!/usr/bin/env python
"""Run Phase 4K direct repair validation for loss-weighted GCBC checkpoints."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3e.phase4h_scene_validation import load_training_summary  # noqa: E402
from phase3e.phase4k_loss_weighted_repair_validation import (  # noqa: E402
    aggregate_phase4k_rows,
    build_phase4k_payload,
    compare_phase4k_to_baseline,
    flatten_direct_repair_summary,
    load_json,
    recommend_phase4k_methods,
    write_phase4k_outputs,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def _dataset_key(dataset_name: str) -> str:
    return dataset_name.replace("-v0", "").replace("-", "_")


def _phase4g_summary_path(output_root: Path, dataset_name: str, phase4e_run_dir: str | Path) -> Path:
    return (
        output_root
        / _dataset_key(dataset_name)
        / Path(str(phase4e_run_dir)).expanduser().name
        / "phase4g_direct_repair_policy_summary.json"
    )


def _run(cmd: list[str], cuda_visible_devices: str | None = None) -> None:
    env = os.environ.copy()
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _model_records_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = str(config["dataset_name"])
    phase2_run = str(config["phase2_run"])
    records = config.get("models", [])
    if not isinstance(records, list) or not records:
        raise ValueError("config must contain a non-empty models list")
    out: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("each model entry must be a mapping")
        row = dict(raw)
        row.setdefault("dataset", dataset)
        row.setdefault("phase2_run", phase2_run)
        if "method" not in row or "seed" not in row or "model_path" not in row:
            raise ValueError("each model entry must contain method, seed, and model_path")
        row["seed"] = int(row["seed"])
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = _load_yaml(config_path)
    dataset_name = str(config["dataset_name"])
    base_direct_config_path = Path(str(config["base_direct_repair_config"])).expanduser()
    base_direct_config = _load_yaml(base_direct_config_path)
    phase4e_run_dir = str(base_direct_config["phase4e_run_dir"])
    output_root = Path(str(config.get("output_dir", "results/phase4k"))).expanduser()
    summary_output_dir = Path(str(config.get("summary_output_dir", output_root))).expanduser()
    eval_root = output_root / _dataset_key(dataset_name) / str(config["phase2_run"])
    cuda_visible_devices = args.cuda_visible_devices
    if cuda_visible_devices is None:
        cuda_visible_devices = config.get("cuda_visible_devices")

    rows: list[dict[str, Any]] = []
    for model in _model_records_from_config(config):
        run_name = str(model.get("run_name") or f"{model['method']}_seed{model['seed']}")
        run_output_root = eval_root / run_name / "direct_repair_policy"
        summary_path = _phase4g_summary_path(run_output_root, dataset_name, phase4e_run_dir)
        model_path = Path(str(model["model_path"])).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint for {run_name}: {model_path}")
        resolved_config = {
            **base_direct_config,
            "dataset_name": dataset_name,
            "model_path": str(model_path),
            "output_dir": str(run_output_root),
            "seed": int(model["seed"]),
        }
        if config.get("device") is not None:
            resolved_config["device"] = config["device"]
        if config.get("max_examples") is not None:
            resolved_config["max_examples"] = int(config["max_examples"])
        if config.get("batch_size") is not None:
            resolved_config["batch_size"] = int(config["batch_size"])
        resolved_config_path = run_output_root / "phase4g_config_resolved.yaml"
        _write_yaml(resolved_config_path, resolved_config)
        if not args.summarize_only and (args.force_eval or not summary_path.exists()):
            print(f"[phase4k] evaluating {run_name}")
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_phase4g_direct_repair_policy.py"),
                    "--config",
                    str(resolved_config_path),
                ],
                cuda_visible_devices=cuda_visible_devices,
            )
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected direct repair summary for {run_name}: {summary_path}")
        training_summary = load_training_summary(Path(str(model.get("training_run_dir", ""))).expanduser())
        flat_model = {
            **model,
            "model_path": str(model_path),
            "direct_summary_path": str(summary_path),
        }
        rows.append(
            flatten_direct_repair_summary(
                summary=load_json(summary_path),
                model_record=flat_model,
                training_summary=training_summary,
                planner_method=str(config.get("planner_method", "calibrated_compat_threshold")),
            )
        )

    per_seed = pd.DataFrame(rows)
    aggregate = aggregate_phase4k_rows(per_seed)
    comparisons = compare_phase4k_to_baseline(
        aggregate,
        baseline_method=str(config.get("baseline_method", "uniform_transition_none")),
    )
    recommendations = recommend_phase4k_methods(
        comparisons,
        max_overall_regret=float(config.get("max_overall_regret", 0.05)),
        require_direct_mse_improvement=bool(config.get("require_direct_mse_improvement", True)),
    )
    payload = build_phase4k_payload(
        config={
            **config,
            "config_path": str(config_path),
            "base_direct_repair_config": str(base_direct_config_path),
            "note": "No environment rollout is used.",
        },
        per_seed=per_seed,
        aggregate=aggregate,
        comparisons=comparisons,
        recommendations=recommendations,
    )
    write_phase4k_outputs(summary_output_dir, payload, per_seed, aggregate, comparisons, recommendations)
    print(f"[phase4k] wrote summary outputs under {summary_output_dir}")
    if not recommendations.empty:
        print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
