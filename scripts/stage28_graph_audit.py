#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.common.config import apply_dotlist, load_json
from bars.common.device import get_torch_device
from bars.common.logging import CSVLogger
from bars.experiments.pipeline import _apply_routeb_backbone_config, _load_data, _load_reachability_if_available
from bars.graph.audit import run_graph_method_audit
from bars.graph.boundary import BoundaryIndex
from bars.graph.types import BARSGraph


def _default_fields(cfg: dict, run_dir: Path) -> dict:
    return {
        "run_id": cfg.get("run_id", run_dir.name),
        "env": cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown")),
        "seed": cfg.get("seed", 0),
        "variant": cfg.get("planner", {}).get("variant", "full_bars"),
        "stage": "stage28_graph_audit",
    }


def _load_cached_artifacts(cfg: dict, run_dir: Path, device):
    emb_path = run_dir / "cache" / "embeddings.npy"
    graph_path = run_dir / "cache" / "graph.npz"
    boundary_path = run_dir / "cache" / "boundary.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing cached embeddings: {emb_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing cached graph: {graph_path}")
    embeddings = np.load(emb_path).astype(np.float32)
    graph = BARSGraph.load_npz(str(graph_path))
    boundary = None
    if bool(cfg.get("stage28_audit", {}).get("load_boundary", cfg.get("boundary", {}).get("enabled", True))) and boundary_path.exists():
        boundary = BoundaryIndex.load_npz(str(boundary_path))
    reach_model = None
    if bool(cfg.get("stage28_audit", {}).get("load_reachability", True)):
        reach_model = _load_reachability_if_available(cfg, str(run_dir), embeddings.shape[1], device)
    return embeddings, graph, boundary, reach_model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage28 graph-method audit over cached BARS/GAS artifacts.")
    parser.add_argument("--config", required=True, help="BARS config used to load the dataset and audit knobs.")
    parser.add_argument("--run-dir", required=True, help="Existing run dir containing cache/embeddings.npy and cache/graph.npz.")
    parser.add_argument("--out", default=None, help="CSV path. Defaults to <run-dir>/logs/stage28_graph_audit.csv.")
    parser.add_argument("--env", dest="env_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-pairs", type=int, default=None, help="Override stage28_audit.num_future_pairs.")
    parser.add_argument("--num-cross-pairs", type=int, default=None, help="Override stage28_audit.num_cross_pairs.")
    parser.add_argument("--graph-variants", default=None, help="Comma-separated audit graph variants.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--clear", action="store_true", help="Replace existing audit CSV instead of appending.")
    parser.add_argument("--set", action="append", default=[], help="Dotlist config override, e.g. --set stage28_audit.edge_knn=64")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = load_json(args.config)
    cfg = _apply_routeb_backbone_config(cfg)
    cfg.setdefault("run_id", run_dir.name)
    if args.env_name is not None:
        cfg.setdefault("data", {})["env_name"] = args.env_name
        cfg["env_name"] = args.env_name
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = args.device
    cfg = apply_dotlist(cfg, args.set or [])
    cfg.setdefault("stage28_audit", {})
    if args.num_pairs is not None:
        cfg["stage28_audit"]["num_future_pairs"] = int(args.num_pairs)
    if args.num_cross_pairs is not None:
        cfg["stage28_audit"]["num_cross_pairs"] = int(args.num_cross_pairs)
    if args.graph_variants:
        cfg["stage28_audit"]["graph_variants"] = [x.strip() for x in args.graph_variants.split(",") if x.strip()]

    out_path = Path(args.out) if args.out else run_dir / "logs" / "stage28_graph_audit.csv"
    if args.clear and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_torch_device(str(cfg.get("device", "cuda")))
    _, dataset = _load_data(cfg)
    embeddings, graph, boundary, reach_model = _load_cached_artifacts(cfg, run_dir, device)
    logger = CSVLogger(str(out_path), _default_fields(cfg, run_dir))
    run_graph_method_audit(dataset, embeddings, graph, cfg, logger, reach_model=reach_model, device=device, boundary=boundary)
    print(str(out_path))


if __name__ == "__main__":
    main()
