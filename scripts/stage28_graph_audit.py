#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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


def _default_fields(cfg: dict, run_dir: Path, out_path: Path) -> dict:
    return {
        "run_id": cfg.get("run_id", run_dir.name),
        "env": cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown")),
        "seed": cfg.get("seed", 0),
        "variant": cfg.get("planner", {}).get("variant", "full_bars"),
        "stage": "stage28_graph_audit",
        "report_file": str(out_path),
        "baseline_graph_role": "sota_study_baseline_cached_bars_gas_aligned",
    }


def _merge_stage28_config(audit_cfg: dict, run_dir: Path) -> dict:
    """Prefer the original run config so dataset/checkpoint shapes match cached artifacts.

    The Stage28 config is intentionally audit-only.  Existing BARS/GAS runs often
    carry source-specific dataset paths, Route-B switches, hidden dimensions, and
    boundary settings in <run-dir>/config.json.  Reusing that config prevents the
    audit from accidentally loading a different dataset or mismatched reachability
    checkpoint.  The audit config contributes only stage28_audit knobs by default.
    """
    run_cfg_path = run_dir / "config.json"
    use_run_config = bool(audit_cfg.get("stage28_audit", {}).get("use_run_config", True))
    if not use_run_config or not run_cfg_path.exists():
        return copy.deepcopy(audit_cfg)
    cfg = load_json(str(run_cfg_path))
    if "stage28_audit" in audit_cfg:
        cfg.setdefault("stage28_audit", {}).update(copy.deepcopy(audit_cfg["stage28_audit"]))
    if "ann" in audit_cfg and bool(audit_cfg.get("stage28_audit", {}).get("override_ann", False)):
        cfg["ann"] = copy.deepcopy(audit_cfg["ann"])
    if "experiment" in audit_cfg:
        cfg.setdefault("experiment", {}).update(copy.deepcopy(audit_cfg["experiment"]))
    return cfg


def _load_cached_artifacts(cfg: dict, run_dir: Path, device):
    emb_path = run_dir / "cache" / "embeddings.npy"
    graph_path = run_dir / "cache" / "graph.npz"
    boundary_path = run_dir / "cache" / "boundary.npz"
    reachability_path = run_dir / "checkpoints" / "reachability.pt"
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
    artifact_meta = {
        "phase": "stage28_cache_artifacts",
        "event": "completed",
        "gate": "PASS_STAGE28_CACHE_ARTIFACTS",
        "evidence_class": "cache_artifact_reuse",
        "embeddings_path": str(emb_path),
        "graph_path": str(graph_path),
        "boundary_path": str(boundary_path),
        "boundary_loaded": int(boundary is not None),
        "reachability_path": str(reachability_path),
        "reachability_loaded": int(reach_model is not None),
        "embeddings_shape": list(embeddings.shape),
        "graph_nodes": graph.num_nodes,
        "graph_edges": graph.num_edges,
    }
    return embeddings, graph, boundary, reach_model, artifact_meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage28 graph-method audit over cached BARS/GAS artifacts.")
    parser.add_argument("--config", required=True, help="Audit config. By default, <run-dir>/config.json is reused and this file supplies stage28_audit knobs.")
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
    audit_cfg = load_json(args.config)
    cfg = _merge_stage28_config(audit_cfg, run_dir)
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
    embeddings, graph, boundary, reach_model, artifact_meta = _load_cached_artifacts(cfg, run_dir, device)
    logger = CSVLogger(str(out_path), _default_fields(cfg, run_dir, out_path))
    logger.log(artifact_meta)
    run_graph_method_audit(dataset, embeddings, graph, cfg, logger, reach_model=reach_model, device=device, boundary=boundary)
    print(str(out_path))


if __name__ == "__main__":
    main()
