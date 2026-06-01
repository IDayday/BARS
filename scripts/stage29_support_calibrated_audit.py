#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.common.config import apply_dotlist, load_json
from bars.common.device import get_torch_device
from bars.common.logging import CSVLogger
from bars.experiments.pipeline import _apply_routeb_backbone_config, _load_data, _load_reachability_if_available
from bars.graph.audit import AuditPair, _path_edge_stats
from bars.graph.boundary import BoundaryIndex
from bars.graph.planner import PlanResult, plan_path
from bars.graph.stage29_support import (
    SupportEvidenceGraph,
    build_support_evidence_graph,
    edge_type_counts,
    edge_type_name,
    graph_support_summary,
    nearest_nodes,
    plan_support_budgeted,
    plan_support_lexicographic,
)
from bars.graph.types import BARSGraph


_PAIR_WORKER_STATE: Dict[str, object] = {}


def _default_fields(cfg: dict, run_dir: Path, out_path: Path) -> dict:
    return {
        "run_id": cfg.get("run_id", run_dir.name),
        "env": cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown")),
        "seed": cfg.get("seed", 0),
        "variant": cfg.get("planner", {}).get("variant", "stage29_support_calibrated_gas"),
        "stage": "stage29_support_calibrated_graph",
        "report_file": str(out_path),
        "baseline_graph_role": "sota_study_baseline_cached_gas_graph",
    }


def _merge_stage29_config(audit_cfg: dict, run_dir: Path) -> dict:
    run_cfg_path = run_dir / "config.json"
    use_run_config = bool(audit_cfg.get("stage29_support", {}).get("use_run_config", True))
    if not use_run_config or not run_cfg_path.exists():
        return copy.deepcopy(audit_cfg)
    cfg = load_json(str(run_cfg_path))
    for key in ("stage29_support", "ann", "experiment"):
        if key in audit_cfg:
            cfg.setdefault(key, {}).update(copy.deepcopy(audit_cfg[key]))
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
    if bool(cfg.get("stage29_support", {}).get("load_boundary", cfg.get("boundary", {}).get("enabled", True))) and boundary_path.exists():
        boundary = BoundaryIndex.load_npz(str(boundary_path))
    reach_model = None
    if bool(cfg.get("stage29_support", {}).get("load_reachability", True)):
        reach_model = _load_reachability_if_available(cfg, str(run_dir), embeddings.shape[1], device)
    artifact_meta = {
        "phase": "stage29_cache_artifacts",
        "event": "completed",
        "gate": "PASS_STAGE29_GAS_CACHE_ARTIFACTS",
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


def _same_traj_cross_rate(dataset, graph: BARSGraph, edge_path: Iterable[int]) -> float:
    e = np.asarray(list(edge_path), dtype=np.int64)
    if len(e) == 0:
        return float("nan")
    same = dataset.traj_id[graph.node_indices[graph.src[e]]] == dataset.traj_id[graph.node_indices[graph.dst[e]]]
    return float((~same).mean())


def _largest_edge_cost_ratio(graph: BARSGraph, edge_path: Iterable[int]) -> float:
    e = np.asarray(list(edge_path), dtype=np.int64)
    if len(e) == 0:
        return float("nan")
    total = float(np.sum(graph.cost[e]))
    return float(np.max(graph.cost[e]) / max(total, 1e-8))


def _label_failure(row: Dict[str, float | int | str], cfg: Dict) -> str:
    scfg = cfg.get("stage29_support", {})
    found = int(row.get("found", 0))
    data_supported = int(row.get("data_supported", 0))
    cross_rate = float(row.get("path_cross_edge_rate", 0.0) or 0.0)
    hop_ratio = float(row.get("path_largest_edge_cost_ratio", 0.0) or 0.0)
    unsupported = int(row.get("unsupported_edges", 0) or 0)
    stage29_graph = str(row.get("graph_id", "")) == "stage29_support_calibrated"
    if data_supported and not found:
        return "BASE_LOST_SUPPORTED_PATH_OR_EDGE_PRUNING"
    if found and data_supported and cross_rate >= float(scfg.get("risky_cross_rate_threshold", 0.50)) and (not stage29_graph or unsupported > 0):
        return "BASE_USES_CROSS_TRAJ_SHORTCUT_FOR_SUPPORTED_PAIR"
    if found and data_supported and cross_rate >= float(scfg.get("risky_cross_rate_threshold", 0.50)) and unsupported == 0:
        return "GRAPH_PATH_EXISTS_EXECUTION_PROBE_NEEDED"
    if found and hop_ratio >= float(scfg.get("largest_hop_ratio_threshold", 0.55)):
        return "BASE_SINGLE_HOP_DOMINATED_PATH"
    if found:
        return "GRAPH_PATH_EXISTS_EXECUTION_PROBE_NEEDED"
    return "NO_GRAPH_PATH_UNRESOLVED"


def _base_plan_row(dataset, graph: BARSGraph, pair: AuditPair, start_node: int, goal_node: int, cfg: Dict) -> Dict[str, float | int | str]:
    scfg = cfg.get("stage29_support", {})
    max_edges = int(scfg.get("max_path_edges", cfg.get("planner", {}).get("max_edges", 0))) or None
    variant = str(scfg.get("base_planner_variant", "reachability"))
    result = plan_path(
        graph,
        start_node,
        goal_node,
        variant=variant,
        lambda_risk=float(scfg.get("lambda_risk", cfg.get("planner", {}).get("lambda_risk", 1.0))),
        max_edges=max_edges,
    )
    path_stats = _path_edge_stats(dataset, graph, result)
    row = {
        "phase": "stage29_path_probe",
        "event": "base_cached",
        "gate": "PASS_STAGE29_OFFLINE_SUPPORT_AUDIT",
        "evidence_class": "base_cached_path_probe",
        "graph_id": "base_cached",
        "planner_id": f"base_{variant}",
        "pair_id": pair.pair_id,
        "pair_type": pair.pair_type,
        "start_index": pair.start_index,
        "goal_index": pair.goal_index,
        "true_dt": pair.true_dt,
        "data_supported": pair.data_supported,
        "start_node": start_node,
        "goal_node": goal_node,
        "unsupported_edges": 0,
        "support_risk": float("nan"),
        "cross_support_risk": float("nan"),
        **result.to_row(),
        **path_stats,
    }
    row["path_largest_edge_cost_ratio"] = path_stats["path_largest_edge_cost_ratio"]
    row["failure_label"] = _label_failure(row, cfg)
    return row


def _stage29_plan_rows(dataset, evidence: SupportEvidenceGraph, pair: AuditPair, start_node: int, goal_node: int, cfg: Dict) -> List[Dict[str, float | int | str]]:
    scfg = cfg.get("stage29_support", {})
    rows: list[Dict[str, float | int | str]] = []
    max_edges = int(scfg.get("max_path_edges", cfg.get("planner", {}).get("max_edges", 0))) or None
    if bool(scfg.get("run_lexicographic", True)):
        res = plan_support_lexicographic(
            dataset,
            evidence,
            start_node,
            goal_node,
            lambda_support_risk=float(scfg.get("lambda_support_risk", 1.0)),
            max_edges=max_edges,
        )
        rows.append(_stage29_row_from_result(dataset, evidence, pair, start_node, goal_node, "stage29_lexicographic", res, cfg))
    unsupported_budgets = _parse_ints(scfg.get("unsupported_budgets", [0, 1, 2]))
    support_budgets = _parse_floats(scfg.get("support_risk_budgets", [0.0, 0.5, 1.0, 2.0]))
    for k in unsupported_budgets:
        for b in support_budgets:
            res = plan_support_budgeted(
                dataset,
                evidence,
                start_node,
                goal_node,
                unsupported_budget=int(k),
                support_risk_budget=float(b),
                support_risk_bin=float(scfg.get("support_risk_bin", 0.05)),
                max_edges=max_edges,
            )
            rows.append(_stage29_row_from_result(dataset, evidence, pair, start_node, goal_node, f"stage29_budgeted_k{k}_r{b:g}", res, cfg))
    return rows


def _stage29_row_from_result(dataset, evidence: SupportEvidenceGraph, pair: AuditPair, start_node: int, goal_node: int, planner_id: str, result, cfg: Dict) -> Dict[str, float | int | str]:
    row = {
        "phase": "stage29_path_probe",
        "event": "stage29_support_calibrated",
        "gate": "PASS_STAGE29_OFFLINE_SUPPORT_AUDIT",
        "evidence_class": "support_constrained_path_probe",
        "graph_id": "stage29_support_calibrated",
        "planner_id": planner_id,
        "pair_id": pair.pair_id,
        "pair_type": pair.pair_type,
        "start_index": pair.start_index,
        "goal_index": pair.goal_index,
        "true_dt": pair.true_dt,
        "data_supported": pair.data_supported,
        "start_node": start_node,
        "goal_node": goal_node,
        **result.to_row(),
    }
    row["path_largest_edge_cost_ratio"] = _largest_edge_cost_ratio(evidence.graph, result.plan.edge_path)
    row["failure_label"] = _label_failure(row, cfg)
    return row


def _parse_ints(value) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _parse_floats(value) -> list[float]:
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return [float(x) for x in value]


def _sample_stage29_pairs(dataset, cfg: Dict) -> List[AuditPair]:
    scfg = cfg.get("stage29_support", {})
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 2903)
    num_future = int(scfg.get("num_future_pairs", scfg.get("num_pairs", 256)))
    num_cross = int(scfg.get("num_cross_pairs", max(0, num_future // 2)))
    min_dt = int(scfg.get("path_min_dt", cfg.get("reachability", {}).get("horizon", 50) * 2))
    max_dt = int(scfg.get("path_max_dt", 250))
    pairs: list[AuditPair] = []
    if num_future > 0:
        try:
            s, g, dt = dataset.sample_future_pairs(num_future, max_dt, rng, min_dt=min_dt)
            for i in range(len(s)):
                pairs.append(AuditPair(len(pairs), "future_same_traj", int(s[i]), int(g[i]), int(dt[i]), 1))
        except Exception:
            pass
    if num_cross > 0 and dataset.num_trajectories > 1:
        attempts = 0
        max_attempts = max(1000, num_cross * 50)
        cross_count = 0
        while cross_count < num_cross and attempts < max_attempts:
            attempts += 1
            i = int(rng.integers(0, dataset.size))
            j = int(rng.integers(0, dataset.size))
            if dataset.traj_id[i] == dataset.traj_id[j] or i == j:
                continue
            pairs.append(AuditPair(len(pairs), "cross_traj_random", i, j, -1, 0))
            cross_count += 1
    return pairs


def _log_taxonomy_summary(rows: List[Dict[str, float | int | str]], logger: CSVLogger, planner_id: str, event: str) -> None:
    labels: Dict[str, int] = {}
    for row in rows:
        label = str(row.get("failure_label", "UNKNOWN"))
        labels[label] = labels.get(label, 0) + 1
    total = max(1, len(rows))
    for label, count in sorted(labels.items(), key=lambda kv: (-kv[1], kv[0])):
        logger.log(
            {
                "phase": "stage29_failure_taxonomy_summary",
                "event": event,
                "gate": "PASS_STAGE29_FAILURE_TAXONOMY_DELTA",
                "evidence_class": "failure_taxonomy_delta",
                "planner_id": planner_id,
                "failure_label": label,
                "count": int(count),
                "rate": float(count / total),
                "num_pairs": int(total),
            }
        )


def _set_pair_worker_state(state: Dict[str, object]) -> None:
    global _PAIR_WORKER_STATE
    _PAIR_WORKER_STATE = state


def _pair_worker(pair: AuditPair) -> tuple[Dict[str, float | int | str], list[Dict[str, float | int | str]]]:
    state = _PAIR_WORKER_STATE
    dataset = state["dataset"]
    base_graph = state["base_graph"]
    evidence = state["evidence"]
    cfg = state["cfg"]
    base_start = state["base_start"]
    base_goal = state["base_goal"]
    stage29_start = state["stage29_start"]
    stage29_goal = state["stage29_goal"]
    base_row = _base_plan_row(dataset, base_graph, pair, int(base_start[pair.pair_id]), int(base_goal[pair.pair_id]), cfg)
    stage29_rows = _stage29_plan_rows(dataset, evidence, pair, int(stage29_start[pair.pair_id]), int(stage29_goal[pair.pair_id]), cfg)
    return base_row, stage29_rows


def _write_edge_type_rows(evidence: SupportEvidenceGraph, logger: CSVLogger) -> None:
    counts = edge_type_counts(evidence)
    total = max(1, evidence.graph.num_edges)
    for name, count in sorted(counts.items()):
        logger.log(
            {
                "phase": "stage29_edge_type_summary",
                "event": "completed",
                "gate": "PASS_STAGE29_TYPED_EVIDENCE_GRAPH",
                "evidence_class": "typed_edge_distribution",
                "edge_type": name,
                "count": int(count),
                "rate": float(count / total),
            }
        )


def run_stage29_audit(dataset, embeddings: np.ndarray, base_graph: BARSGraph, cfg: Dict, logger: CSVLogger) -> SupportEvidenceGraph:
    traj_lens = [sl.end - sl.start for sl in dataset.traj_slices]
    logger.log(
        {
            "phase": "stage29_dataset_support",
            "event": "completed",
            "gate": "PASS_STAGE29_DATASET_SUPPORT_AUDIT",
            "evidence_class": "dataset_support",
            "dataset_size": dataset.size,
            "num_trajectories": dataset.num_trajectories,
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "traj_len_mean": float(np.mean(traj_lens)) if traj_lens else 0.0,
            "traj_len_p10": float(np.quantile(traj_lens, 0.10)) if traj_lens else 0.0,
            "traj_len_p90": float(np.quantile(traj_lens, 0.90)) if traj_lens else 0.0,
            "base_graph_nodes": base_graph.num_nodes,
            "base_graph_edges": base_graph.num_edges,
            "seed_for_audit": int(cfg.get("seed", 0)),
        }
    )
    evidence = build_support_evidence_graph(dataset, embeddings, base_graph, cfg)
    logger.log(
        {
            "phase": "stage29_graph_summary",
            "event": "completed",
            "gate": "PASS_STAGE29_TYPED_EVIDENCE_GRAPH",
            "evidence_class": "support_calibrated_graph_summary",
            "graph_id": "stage29_support_calibrated",
            **graph_support_summary(dataset, evidence),
        }
    )
    _write_edge_type_rows(evidence, logger)

    pairs = _sample_stage29_pairs(dataset, cfg)
    logger.log(
        {
            "phase": "stage29_pair_sampling",
            "event": "completed",
            "gate": "PASS_STAGE29_PAIR_SAMPLING",
            "evidence_class": "audit_pair_sampling",
            "num_pairs": len(pairs),
            "num_future_pairs": sum(p.pair_type == "future_same_traj" for p in pairs),
            "num_cross_pairs": sum(p.pair_type == "cross_traj_random" for p in pairs),
        }
    )
    if not pairs:
        return evidence

    start_idx = np.asarray([p.start_index for p in pairs], dtype=np.int64)
    goal_idx = np.asarray([p.goal_index for p in pairs], dtype=np.int64)
    stage29_start = nearest_nodes(evidence, embeddings, start_idx, cfg)
    stage29_goal = nearest_nodes(evidence, embeddings, goal_idx, cfg)
    from bars.graph.audit import _nearest_node_map

    base_start = _nearest_node_map(base_graph, embeddings, start_idx, cfg)
    base_goal = _nearest_node_map(base_graph, embeddings, goal_idx, cfg)

    base_rows: list[Dict[str, float | int | str]] = []
    stage29_rows_by_planner: Dict[str, list[Dict[str, float | int | str]]] = {}
    default_planner = str(cfg.get("stage29_support", {}).get("default_planner_id", "stage29_budgeted_k1_r1"))
    worker_state: Dict[str, object] = {
        "dataset": dataset,
        "base_graph": base_graph,
        "evidence": evidence,
        "cfg": cfg,
        "base_start": base_start,
        "base_goal": base_goal,
        "stage29_start": stage29_start,
        "stage29_goal": stage29_goal,
    }
    num_workers = int(cfg.get("stage29_support", {}).get("num_workers", 1) or 1)
    num_workers = max(1, min(num_workers, len(pairs), os.cpu_count() or num_workers))
    pool = None
    if num_workers > 1:
        try:
            ctx = mp.get_context("fork")
            _set_pair_worker_state(worker_state)
            pool = ctx.Pool(processes=num_workers)
            chunksize = int(cfg.get("stage29_support", {}).get("worker_chunksize", max(1, len(pairs) // max(1, num_workers * 4))))
            iterator = pool.imap(_pair_worker, pairs, chunksize=chunksize)
        except Exception:
            if pool is not None:
                pool.terminate()
                pool.join()
            pool = None
            num_workers = 1
    if num_workers == 1:
        _set_pair_worker_state(worker_state)
        iterator = map(_pair_worker, pairs)
    try:
        for b_row, stage29_rows in iterator:
            base_rows.append(b_row)
            logger.log(b_row)
            for row in stage29_rows:
                stage29_rows_by_planner.setdefault(str(row["planner_id"]), []).append(row)
                logger.log(row)
                if str(row["planner_id"]) == default_planner:
                    proxy = dict(row)
                    proxy.update(
                        {
                            "phase": "stage29_failure_taxonomy_proxy",
                            "event": "stage29_default",
                            "gate": "PASS_STAGE29_FAILURE_TAXONOMY_DELTA",
                            "evidence_class": "failure_taxonomy_proxy",
                        }
                    )
                    logger.log(proxy)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _set_pair_worker_state({})

    _log_taxonomy_summary(base_rows, logger, "base_cached", "base_cached")
    for planner_id, rows in sorted(stage29_rows_by_planner.items()):
        _log_taxonomy_summary(rows, logger, planner_id, "stage29_support_calibrated")
    return evidence


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage29 Support-Calibrated Graph Stitching offline audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--env", dest="env_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--num-cross-pairs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    audit_cfg = load_json(args.config)
    cfg = _merge_stage29_config(audit_cfg, run_dir)
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
    cfg.setdefault("stage29_support", {})
    if args.num_pairs is not None:
        cfg["stage29_support"]["num_future_pairs"] = int(args.num_pairs)
    if args.num_cross_pairs is not None:
        cfg["stage29_support"]["num_cross_pairs"] = int(args.num_cross_pairs)
    if args.num_workers is not None:
        cfg["stage29_support"]["num_workers"] = int(args.num_workers)

    out_path = Path(args.out) if args.out else run_dir / "logs" / "stage29_support_calibrated_audit.csv"
    if args.clear and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_torch_device(str(cfg.get("device", "cpu")))
    _, dataset = _load_data(cfg)
    embeddings, base_graph, _boundary, _reach_model, artifact_meta = _load_cached_artifacts(cfg, run_dir, device)
    logger = CSVLogger(str(out_path), _default_fields(cfg, run_dir, out_path))
    logger.log(artifact_meta)
    run_stage29_audit(dataset, embeddings, base_graph, cfg, logger)
    print(str(out_path))


if __name__ == "__main__":
    main()
