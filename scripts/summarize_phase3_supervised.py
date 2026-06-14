#!/usr/bin/env python
"""Summarize Phase 3A offline supervised GCBC metrics."""

from __future__ import annotations

import argparse
import json
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
from phase2.compatibility import compute_edge_compatibility  # noqa: E402
from phase3.edge_bc_dataset import build_edge_bc_examples  # noqa: E402
from phase3.evaluation import (  # noqa: E402
    OFFLINE_MSE_NOTE,
    load_phase2_artifacts,
    normalize_edge_val_metrics,
    summarize_supervised_metrics,
)
from phase3.models import GCBCMLP  # noqa: E402
from phase3.train_gcbc import evaluate_policy_mse  # noqa: E402


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


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _load_model_edge_metrics(
    run_dir: Path,
    phase2_run_dir: Path,
    dataset_name: str,
    dataset_dir: str | None,
    max_transitions: int | None,
    batch_size: int,
    val_examples: int,
    val_fraction: float,
    seed: int,
    device: str | None,
) -> pd.DataFrame:
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing edge_val_metrics.csv and model.pt under {run_dir}")
    import torch

    dataset = load_ogbench_dataset(
        dataset_name,
        dataset_dir,
        split="train",
        max_transitions=max_transitions,
    )
    full = build_edge_bc_examples(
        dataset,
        phase2_run_dir / "option_edges.csv",
        phase2_run_dir / "edge_segments.npz",
        sampling_mode="uniform_edge",
        seed=seed,
    )
    _, val_ds = full.split_by_segments(val_fraction, seed=seed)
    ckpt = torch.load(model_path, map_location="cpu")
    model_config = ckpt["model_config"]
    model = GCBCMLP(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    _, edge_metrics = evaluate_policy_mse(
        model,
        val_ds,
        batch_size=batch_size,
        max_examples=val_examples,
        device=dev,
    )
    return edge_metrics


def _load_or_compute_edge_metrics(args: argparse.Namespace, config: dict[str, Any]) -> pd.DataFrame:
    run_dir = Path(args.run_dir).expanduser()
    edge_path = run_dir / "edge_val_metrics.csv"
    if edge_path.exists():
        return pd.read_csv(edge_path)
    dataset_name = args.dataset_name or config.get("dataset_name")
    if not dataset_name:
        raise ValueError("--dataset_name is required when edge_val_metrics.csv is missing")
    return _load_model_edge_metrics(
        run_dir=run_dir,
        phase2_run_dir=Path(args.phase2_run_dir).expanduser(),
        dataset_name=str(dataset_name),
        dataset_dir=args.dataset_dir or config.get("dataset_dir"),
        max_transitions=args.max_transitions if args.max_transitions is not None else config.get("max_transitions"),
        batch_size=int(args.batch_size or config.get("batch_size", 1024)),
        val_examples=int(args.val_examples or config.get("val_examples", 8192)),
        val_fraction=float(args.val_fraction if args.val_fraction is not None else config.get("val_fraction", 0.05)),
        seed=int(args.seed if args.seed is not None else config.get("seed", 0)),
        device=args.device,
    )


def _format_float(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    return "" if not np.isfinite(value) else f"{value:.7g}"


def _write_markdown(
    path: Path,
    run_dir: Path,
    phase2_run_dir: Path,
    summary: dict[str, Any],
    grouped: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 3A Supervised GCBC Summary",
        "",
        "This summary is reset-free and offline-only. It reports action-prediction",
        "MSE on held-out edge BC samples from Phase 2 support-certified option",
        "segments.",
        "",
        f"Run dir: `{run_dir}`",
        f"Phase 2 run dir: `{phase2_run_dir}`",
        "",
        f"Final train action MSE: `{_format_float(summary.get('final_train_action_mse'))}`",
        f"Final val action MSE: `{_format_float(summary.get('final_val_action_mse'))}`",
        f"Best val action MSE: `{_format_float(summary.get('best_val_action_mse'))}`",
        "",
        "The existing 100000-step GCBC run has final `val_action_mse = 0.0426389`.",
        "This shows that the state-based GCBC model can fit held-out edge BC",
        "samples. It does not prove option-edge executability or online rollout",
        "success.",
        "",
        "Rollout remains skipped while Phase 3 preflight reports `env_unavailable`",
        "because this Python environment lacks env-construction dependencies. That",
        "is an environment dependency blocker, not evidence that AntMaze or Scene",
        "lack reset-to-state support.",
        "",
        f"Note: {OFFLINE_MSE_NOTE}",
        "",
    ]
    if not grouped.empty:
        lines += [
            "## Grouped Metrics",
            "",
            "| group | num_edges | num_val_samples | val_action_mse |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in grouped.to_dict("records"):
            lines.append(
                "| {group} | {num_edges} | {num_val_samples} | {mse} |".format(
                    group=row["group"],
                    num_edges=int(row["num_edges"]),
                    num_val_samples=int(row["num_val_samples"]),
                    mse=_format_float(row["val_action_mse"]),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--phase2_run_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_examples", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compat_H_intra", type=int, default=10)
    parser.add_argument("--skip_compatibility_context", action="store_true")
    parser.add_argument("--docs_path", default="docs/phase3_supervised_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    config = _read_yaml(run_dir / "config_resolved.yaml")
    option_edges, edge_segments, _ = load_phase2_artifacts(phase2_run_dir)
    train_metrics = pd.read_csv(run_dir / "train_metrics.csv")
    val_metrics = pd.read_csv(run_dir / "val_metrics.csv")
    raw_edge_metrics = _load_or_compute_edge_metrics(args, config)

    pair_compatibility = None
    if not args.skip_compatibility_context:
        _, pair_compatibility = compute_edge_compatibility(
            option_edges,
            edge_segments,
            labels=np.empty(0, dtype=np.int64),
            pair_records={},
            H_intra=int(args.compat_H_intra),
        )

    edge_metrics = normalize_edge_val_metrics(raw_edge_metrics, option_edges, pair_compatibility)
    grouped, summary = summarize_supervised_metrics(train_metrics, val_metrics, edge_metrics)
    edge_metrics.to_csv(run_dir / "edge_val_metrics.csv", index=False)
    grouped.to_csv(run_dir / "grouped_val_metrics.csv", index=False)
    _write_json(run_dir / "phase3a_supervised_summary.json", summary)
    _write_markdown(run_dir / "phase3a_supervised_summary.md", run_dir, phase2_run_dir, summary, grouped)
    _write_markdown(Path(args.docs_path), run_dir, phase2_run_dir, summary, grouped)
    print(f"[phase3A] wrote supervised summary under {run_dir}")
    print(json.dumps(_json_safe(summary), sort_keys=True))


if __name__ == "__main__":
    main()
