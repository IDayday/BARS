#!/usr/bin/env python
"""Train a state-based GCBC policy on Phase 2 option-edge segments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data import load_ogbench_dataset  # noqa: E402
from phase3.edge_bc_dataset import SUPPORTED_SAMPLING_MODES  # noqa: E402
from phase3.evaluation import default_phase3_output_dir  # noqa: E402
from phase3.train_gcbc import train_gcbc  # noqa: E402


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _parse_hidden_dims(value: str | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.replace("[", "").replace("]", "").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--phase2_run_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--sampling_mode", default=None, choices=sorted(SUPPORTED_SAMPLING_MODES))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dims", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--val_examples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--edge_embedding_dim", type=int, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    merged = vars(args).copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    defaults = {
        "dataset_dir": "/mnt/project/offlinerl_datasets/ogbench",
        "max_transitions": None,
        "sampling_mode": "uniform_edge",
        "batch_size": 1024,
        "num_steps": 100000,
        "lr": 3e-4,
        "hidden_dims": [512, 512, 512],
        "seed": 0,
        "val_fraction": 0.05,
        "val_examples": 8192,
        "log_interval": None,
        "edge_embedding_dim": 0,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    if not merged.get("dataset_name"):
        raise ValueError("--dataset_name is required")
    if not merged.get("phase2_run_dir"):
        raise ValueError("--phase2_run_dir is required")
    merged["hidden_dims"] = _parse_hidden_dims(merged.get("hidden_dims"))
    if merged.get("output_dir") is None:
        merged["output_dir"] = str(default_phase3_output_dir(merged["dataset_name"], merged["phase2_run_dir"]))
    return argparse.Namespace(**merged)


def main() -> None:
    args = parse_args()
    phase2_run_dir = Path(args.phase2_run_dir).expanduser()
    dataset = load_ogbench_dataset(
        args.dataset_name,
        args.dataset_dir,
        split="train",
        max_transitions=args.max_transitions,
    )
    train_gcbc(
        dataset=dataset,
        option_edges_csv=phase2_run_dir / "option_edges.csv",
        edge_segments_npz=phase2_run_dir / "edge_segments.npz",
        output_dir=args.output_dir,
        sampling_mode=args.sampling_mode,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        seed=args.seed,
        val_fraction=args.val_fraction,
        val_examples=args.val_examples,
        log_interval=args.log_interval,
        device=args.device,
        edge_embedding_dim=args.edge_embedding_dim,
        config=vars(args),
    )
    print(f"[phase3] wrote GCBC training outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
