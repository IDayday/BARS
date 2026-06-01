#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.common.config import apply_dotlist, load_json
from bars.common.device import get_torch_device
from bars.common.logging import CSVLogger
from bars.experiments.pipeline import _apply_routeb_backbone_config, _load_data
from bars.gas_bars.edge_execution import try_set_state_from_observation
from bars.graph.edges import _score_reachability
from bars.graph.stage29_support import SupportEvidenceGraph, build_support_evidence_graph, edge_type_name
from scripts.stage29_support_calibrated_audit import _load_cached_artifacts, _maybe_realign_cached_dataset, _merge_stage29_config


def _default_fields(cfg: dict, run_dir: Path, out_path: Path) -> dict:
    return {
        "run_id": cfg.get("run_id", run_dir.name),
        "env": cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown")),
        "seed": cfg.get("seed", 0),
        "variant": cfg.get("planner", {}).get("variant", "stage29_support_calibrated_gas"),
        "stage": "stage29_edge_execution_probe",
        "report_file": str(out_path),
        "baseline_graph_role": "sota_study_baseline_cached_gas_graph",
    }


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _mean(values: Iterable[Any]) -> float:
    xs = [_safe_float(x) for x in values]
    xs = [x for x in xs if math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _rate(values: Iterable[Any]) -> float:
    xs = [_safe_float(x) for x in values]
    xs = [x for x in xs if math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _edge_metadata(dataset, evidence: SupportEvidenceGraph, edge_id: int) -> Dict[str, Any]:
    graph = evidence.graph
    eid = int(edge_id)
    u = int(graph.src[eid])
    v = int(graph.dst[eid])
    si = int(graph.node_indices[u])
    gi = int(graph.node_indices[v])
    same_traj = int(dataset.traj_id[si] == dataset.traj_id[gi])
    return {
        "edge_id": eid,
        "edge_type": edge_type_name(int(evidence.edge_type[eid])),
        "u": u,
        "v": v,
        "src_index": si,
        "dst_index": gi,
        "same_traj": same_traj,
        "dt": int(dataset.timestep[gi] - dataset.timestep[si]) if same_traj else -1,
        "cost": float(graph.cost[eid]),
        "graph_p_exec": float(graph.p_exec[eid]),
        "graph_risk": float(graph.risk[eid]),
        "support_count": int(evidence.support_count[eid]),
        "support_density": float(evidence.support_density[eid]),
        "support_score": float(evidence.support_score[eid]),
        "support_risk": float(evidence.support_risk[eid]),
        "unsupported_shortcut": int(evidence.unsupported_shortcut[eid]),
    }


def sample_edges_by_type(
    dataset,
    evidence: SupportEvidenceGraph,
    *,
    per_type: int,
    seed: int,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    wanted = {str(x) for x in edge_types} if edge_types is not None else None
    rng = np.random.default_rng(int(seed))
    rows: list[Dict[str, Any]] = []
    for et in sorted(np.unique(evidence.edge_type).tolist(), key=int):
        name = edge_type_name(int(et))
        if wanted is not None and name not in wanted:
            continue
        ids = np.flatnonzero(evidence.edge_type == int(et))
        if len(ids) == 0:
            continue
        take = len(ids) if int(per_type) <= 0 else min(len(ids), int(per_type))
        chosen = rng.choice(ids, size=take, replace=False) if take < len(ids) else ids
        for eid in sorted(int(x) for x in chosen.tolist()):
            row = _edge_metadata(dataset, evidence, eid)
            row["sample_count_for_type"] = int(take)
            row["available_count_for_type"] = int(len(ids))
            rows.append(row)
    return rows


def _score_sample_with_reachability(rows: list[Dict[str, Any]], evidence: SupportEvidenceGraph, reach_model, device, cfg: Dict) -> None:
    if not rows:
        return
    src = np.asarray([int(r["u"]) for r in rows], dtype=np.int64)
    dst = np.asarray([int(r["v"]) for r in rows], dtype=np.int64)
    p = _score_reachability(
        reach_model,
        evidence.graph.node_embeddings,
        src,
        dst,
        device,
        batch_size=int(cfg.get("stage29_support", {}).get("edge_probe_score_batch_size", 32768)),
        fallback_scale=float(np.median(evidence.graph.cost) + 1e-6),
    )
    for row, val in zip(rows, p.tolist()):
        row["reachability_p_exec"] = float(val)


def _infer_gas_artifact_root(cfg: Dict, env_name: str, seed: int) -> Optional[Path]:
    eg = cfg.get("external_gas", {})
    paths = [eg.get("dataset_embeddings_path"), eg.get("keygraph_path")]
    seed_dir = f"seed{int(seed)}"
    for raw in paths:
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        parts = p.parts
        for i, part in enumerate(parts):
            if part == env_name and i + 1 < len(parts) and parts[i + 1] == seed_dir:
                return Path(*parts[:i])
    return None


def _load_gas_backbone(cfg: Dict, env_name: str, seed: int, args) -> tuple[Any, str, Optional[Path]]:
    root = Path(args.gas_artifact_root).expanduser() if args.gas_artifact_root else _infer_gas_artifact_root(cfg, env_name, seed)
    if root is None:
        return None, "missing_gas_artifact_root", None
    dataset_dir = cfg.get("data", {}).get("dataset_dir")
    if dataset_dir:
        os.environ["OGBENCH_DATASET_DIR"] = os.path.expandvars(os.path.expanduser(str(dataset_dir)))
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    try:
        from bars.external.gas_backbone import GASBackbone

        bb = GASBackbone.load_or_train(
            env_name,
            int(seed),
            root,
            args.gas_repo_path,
            args.gpu,
            prefer_pretrained=False,
            train_if_missing=False,
            quick=True,
        )
        if bb.agent is None or bb.actor_fn is None:
            return None, "gas_policy_not_loaded", root
        return bb, "loaded", root
    except Exception as exc:
        return None, f"gas_load_error:{type(exc).__name__}:{exc}", root


def _rollout_edge(backbone, env, dataset, evidence: SupportEvidenceGraph, row: Dict[str, Any], cfg: Dict, args) -> Dict[str, Any]:
    start = time.time()
    graph = evidence.graph
    eid = int(row["edge_id"])
    src_node = int(row["u"])
    dst_node = int(row["v"])
    start_obs = np.asarray(dataset.observations[int(row["src_index"])])
    target_phi = np.asarray(graph.node_embeddings[dst_node], dtype=np.float32)
    initial_dist = float(np.linalg.norm(np.asarray(graph.node_embeddings[src_node]) - target_phi))
    horizon = int(args.horizon or cfg.get("stage29_support", {}).get("edge_probe_horizon", cfg.get("reachability", {}).get("horizon", 50)))
    reach_threshold = float(args.reach_threshold or cfg.get("stage29_support", {}).get("edge_probe_reach_threshold", 2.0))
    divergence_margin = float(args.divergence_margin)
    stuck_progress = float(args.stuck_progress_threshold)
    try:
        env.reset(seed=(int(cfg.get("seed", 0)) * 1000003 + eid) % 2_147_483_647)
        ok, reason = try_set_state_from_observation(env, start_obs)
        if not ok:
            row.update(
                {
                    "execution_mode": "rollout_reset_failed",
                    "reset_mode": reason,
                    "reach": float("nan"),
                    "progress": float("nan"),
                    "progress_norm": float("nan"),
                    "divergence": float("nan"),
                    "stuck": float("nan"),
                    "min_dist": float("nan"),
                    "final_dist": float("nan"),
                    "steps": 0,
                    "duration_sec": time.time() - start,
                }
            )
            return row
        obs = start_obs.copy()
        min_dist = float("inf")
        final_dist = float("inf")
        steps = 0
        done = False
        for steps in range(1, horizon + 1):
            action = backbone.sample_action(obs, target_phi, final_goal=False)
            obs, _, done, _ = backbone.step_env(env, backbone.env_name or "", action)
            phi = backbone.get_phi(obs)
            final_dist = float(np.linalg.norm(phi - target_phi))
            min_dist = min(min_dist, final_dist)
            if final_dist <= reach_threshold or done:
                break
        reach = int(min_dist <= reach_threshold)
        progress = float(initial_dist - final_dist)
        row.update(
            {
                "execution_mode": "rollout",
                "reset_mode": "set_state",
                "reach": reach,
                "initial_dist": initial_dist,
                "min_dist": min_dist,
                "final_dist": final_dist,
                "progress": progress,
                "progress_norm": progress / max(initial_dist, 1e-6),
                "divergence": int(final_dist > initial_dist + divergence_margin),
                "stuck": int((not reach) and progress <= stuck_progress),
                "steps": int(steps),
                "timeout": int(steps >= horizon and not reach and not done),
                "duration_sec": time.time() - start,
            }
        )
        return row
    except Exception as exc:
        row.update(
            {
                "execution_mode": "rollout_error",
                "reset_mode": "error",
                "error": f"{type(exc).__name__}:{exc}",
                "reach": float("nan"),
                "progress": float("nan"),
                "progress_norm": float("nan"),
                "divergence": float("nan"),
                "stuck": float("nan"),
                "min_dist": float("nan"),
                "final_dist": float("nan"),
                "steps": 0,
                "duration_sec": time.time() - start,
            }
        )
        return row


def summary_rows_by_type(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    by_type: dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("edge_type", "UNKNOWN")), []).append(row)
    for name, part in sorted(by_type.items()):
        out.append(
            {
                "phase": "stage29_edge_execution_summary",
                "event": "completed",
                "gate": "PASS_STAGE29_EDGE_EXECUTION_PROBE",
                "evidence_class": "edge_type_execution_probe",
                "edge_type": name,
                "count": int(len(part)),
                "reach_rate": _rate(r.get("reach") for r in part),
                "progress_mean": _mean(r.get("progress") for r in part),
                "progress_norm_mean": _mean(r.get("progress_norm") for r in part),
                "divergence_rate": _rate(r.get("divergence") for r in part),
                "stuck_rate": _rate(r.get("stuck") for r in part),
                "support_score_mean": _mean(r.get("support_score") for r in part),
                "graph_p_exec_mean": _mean(r.get("graph_p_exec") for r in part),
                "reachability_p_exec_mean": _mean(r.get("reachability_p_exec") for r in part),
            }
        )
    return out


def calibration_rows(rows: list[Dict[str, Any]], *, bins: int = 5) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    if not rows or bins <= 0:
        return out
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    support = np.asarray([_safe_float(r.get("support_score")) for r in rows], dtype=np.float32)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (support >= lo) & (support < hi if hi < 1.0 else support <= hi)
        part = [r for r, ok in zip(rows, mask.tolist()) if ok]
        if not part:
            continue
        out.append(
            {
                "phase": "stage29_edge_calibration",
                "event": "completed",
                "gate": "PASS_STAGE29_EDGE_EXECUTION_PROBE",
                "evidence_class": "support_score_p_exec_calibration",
                "support_bin_lo": float(lo),
                "support_bin_hi": float(hi),
                "count": int(len(part)),
                "reach_rate": _rate(r.get("reach") for r in part),
                "support_score_mean": _mean(r.get("support_score") for r in part),
                "graph_p_exec_mean": _mean(r.get("graph_p_exec") for r in part),
                "reachability_p_exec_mean": _mean(r.get("reachability_p_exec") for r in part),
                "progress_norm_mean": _mean(r.get("progress_norm") for r in part),
            }
        )
    return out


def _write_markdown_report(path: Path, rows: list[Dict[str, Any]], summaries: list[Dict[str, Any]], calibrations: list[Dict[str, Any]], artifact_row: Dict[str, Any]) -> None:
    lines = [
        "# Stage29 Edge Execution Probe",
        "",
        "- Evidence class: edge execution probe for Stage29 typed evidence graph.",
        f"- Boundary loaded: {artifact_row.get('boundary_loaded', 0)}.",
        f"- Reachability loaded: {artifact_row.get('reachability_loaded', 0)}.",
        f"- GAS policy status: {artifact_row.get('gas_policy_status', 'not_requested')}.",
        f"- Sampled edges: {len(rows)}.",
        "",
        "## Edge Type Summary",
        "",
    ]
    if summaries:
        header = ["edge_type", "count", "reach_rate", "progress_norm_mean", "divergence_rate", "stuck_rate", "support_score_mean", "graph_p_exec_mean", "reachability_p_exec_mean"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in summaries:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    else:
        lines.append("No summary rows generated.")
    lines.extend(["", "## Calibration", ""])
    if calibrations:
        header = ["support_bin_lo", "support_bin_hi", "count", "reach_rate", "support_score_mean", "graph_p_exec_mean", "reachability_p_exec_mean"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in calibrations:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    else:
        lines.append("No calibration rows generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_edge_probe(dataset, embeddings: np.ndarray, base_graph, boundary, reach_model, cfg: Dict, logger: CSVLogger, args) -> None:
    evidence = build_support_evidence_graph(dataset, embeddings, base_graph, cfg)
    rows = sample_edges_by_type(
        dataset,
        evidence,
        per_type=int(args.edges_per_type),
        seed=int(cfg.get("seed", 0)) + 2917,
        edge_types=[x.strip() for x in args.edge_types.split(",") if x.strip()] if args.edge_types else None,
    )
    device = get_torch_device(str(cfg.get("device", "cpu")))
    _score_sample_with_reachability(rows, evidence, reach_model, device, cfg)

    env_name = cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown"))
    seed = int(cfg.get("seed", 0))
    backbone = None
    gas_status = "not_requested"
    gas_root = None
    if args.rollout:
        backbone, gas_status, gas_root = _load_gas_backbone(cfg, env_name, seed, args)
    artifact_row = {
        "phase": "stage29_edge_probe_artifacts",
        "event": "completed",
        "gate": "PASS_STAGE29_EDGE_PROBE_ARTIFACT_LOAD",
        "evidence_class": "cache_and_probe_artifact_load",
        "boundary_loaded": int(boundary is not None),
        "reachability_loaded": int(reach_model is not None),
        "boundary_applicable_to_stage29_edges": int(boundary is not None and getattr(boundary, "has_dep", np.empty(0)).shape[0] == evidence.graph.num_edges),
        "gas_policy_status": gas_status,
        "gas_artifact_root": str(gas_root) if gas_root is not None else "",
        "rollout_requested": int(bool(args.rollout)),
        "sampled_edges": int(len(rows)),
    }
    logger.log(artifact_row)

    if args.rollout and backbone is not None:
        env, _, _ = backbone.load_env_and_dataset()
        for row in rows:
            _rollout_edge(backbone, env, dataset, evidence, row, cfg, args)
    else:
        for row in rows:
            row.setdefault("execution_mode", "sample_only" if not args.rollout else gas_status)
            row.setdefault("reach", float("nan"))
            row.setdefault("progress", float("nan"))
            row.setdefault("progress_norm", float("nan"))
            row.setdefault("divergence", float("nan"))
            row.setdefault("stuck", float("nan"))

    for row in rows:
        logger.log(
            {
                "phase": "stage29_edge_execution_probe",
                "event": "edge_result" if row.get("execution_mode") == "rollout" else "edge_sample",
                "gate": "PASS_STAGE29_EDGE_EXECUTION_PROBE",
                "evidence_class": "edge_execution_probe" if row.get("execution_mode") == "rollout" else "edge_probe_sample",
                **row,
            }
        )
    summaries = summary_rows_by_type(rows)
    calibrations = calibration_rows(rows, bins=int(args.calibration_bins))
    for row in summaries:
        logger.log(row)
    for row in calibrations:
        logger.log(row)
    report_path = Path(args.report) if args.report else Path(logger.path).with_suffix(".md")
    _write_markdown_report(report_path, rows, summaries, calibrations, artifact_row)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage29 edge execution probe for support-calibrated typed evidence graphs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--env", dest="env_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--edges-per-type", type=int, default=32)
    parser.add_argument("--edge-types", default=None, help="Comma-separated Stage29 edge type names. Defaults to all present types.")
    parser.add_argument("--calibration-bins", type=int, default=5)
    parser.add_argument("--rollout", action="store_true", help="Run GAS policy rollouts from edge source observations. Without this, only sampling/load/calibration inputs are reported.")
    parser.add_argument("--gas-artifact-root", default=None)
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--gpu", default="cpu")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--reach-threshold", type=float, default=None)
    parser.add_argument("--divergence-margin", type=float, default=0.5)
    parser.add_argument("--stuck-progress-threshold", type=float, default=0.05)
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
    out_path = Path(args.out) if args.out else run_dir / "logs" / "stage29_edge_execution_probe.csv"
    if args.clear and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_torch_device(str(cfg.get("device", "cpu")))
    embeddings, base_graph, boundary, reach_model, artifact_meta = _load_cached_artifacts(cfg, run_dir, device)
    _, dataset = _load_data(cfg)
    dataset, realign_meta = _maybe_realign_cached_dataset(cfg, dataset, embeddings, base_graph)
    artifact_meta.update(realign_meta)
    logger = CSVLogger(str(out_path), _default_fields(cfg, run_dir, out_path))
    artifact_meta = copy.deepcopy(artifact_meta)
    artifact_meta.update(
        {
            "phase": "stage29_edge_probe_cache_artifacts",
            "gate": "PASS_STAGE29_EDGE_PROBE_ARTIFACT_LOAD",
            "evidence_class": "cache_artifact_reuse_for_edge_probe",
        }
    )
    logger.log(artifact_meta)
    run_edge_probe(dataset, embeddings, base_graph, boundary, reach_model, cfg, logger, args)
    print(str(out_path))


if __name__ == "__main__":
    main()
