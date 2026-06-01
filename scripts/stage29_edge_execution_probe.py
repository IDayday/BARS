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


REQUIRED_STAGE29B_EDGE_TYPES = [
    "TEMPORAL_BACKBONE",
    "PROJECTED_TEMPORAL_SUPPORT",
    "SUPPORTED_CROSS_BRIDGE",
    "CROSS_TRAJ_CANDIDATE",
    "UNSUPPORTED_SHORTCUT",
]

EXECUTION_STATUS_PASS = "PASS"
EXECUTION_STATUS_INSUFFICIENT = "INSUFFICIENT_SAMPLE"
EXECUTION_STATUS_FAIL = "FAIL"
PROMOTION_BLOCKED = "BLOCKED_EXECUTION_EVIDENCE_NOT_VALID"
PROMOTION_READY = "READY_FOR_20EP_ONLINE_EVAL_GATE"
TEMPORAL_SANITY_READY = "TEMPORAL_SANITY_PASS_RUN_FULL_CALIBRATION"


def _default_fields(cfg: dict, run_dir: Path, out_path: Path) -> dict:
    return {
        "run_id": cfg.get("run_id", run_dir.name),
        "env": cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown")),
        "seed": cfg.get("seed", 0),
        "variant": cfg.get("planner", {}).get("variant", "stage29_support_calibrated_gas"),
        "stage": "stage29_edge_execution_probe",
        "report_file": str(out_path),
        "baseline_graph_role": "sota_study_baseline_cached_gas_graph",
        "stage29a_offline_scg_status": "PASS",
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


def _edge_same_dt(dataset, evidence: SupportEvidenceGraph) -> tuple[np.ndarray, np.ndarray]:
    graph = evidence.graph
    src_idx = graph.node_indices[graph.src]
    dst_idx = graph.node_indices[graph.dst]
    same = dataset.traj_id[src_idx] == dataset.traj_id[dst_idx]
    dt = dataset.timestep[dst_idx] - dataset.timestep[src_idx]
    return same.astype(bool), dt.astype(np.int64)


def edge_type_availability(evidence: SupportEvidenceGraph) -> Dict[str, int]:
    out = {name: 0 for name in REQUIRED_STAGE29B_EDGE_TYPES}
    for et in np.unique(evidence.edge_type).tolist():
        name = edge_type_name(int(et))
        out[name] = int(np.sum(evidence.edge_type == int(et)))
    return out


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


def _choice(rng: np.random.Generator, ids: np.ndarray, take: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0 or take <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(ids, size=take, replace=False) if take < len(ids) else ids


def sample_edges_by_type(
    dataset,
    evidence: SupportEvidenceGraph,
    *,
    per_type: int,
    seed: int,
    edge_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    wanted = [str(x) for x in edge_types] if edge_types is not None else None
    rng = np.random.default_rng(int(seed))
    rows: list[Dict[str, Any]] = []
    if wanted is None:
        names = [edge_type_name(int(et)) for et in sorted(np.unique(evidence.edge_type).tolist(), key=int)]
    else:
        names = wanted
    for name in names:
        ids = np.asarray([i for i, et in enumerate(evidence.edge_type.tolist()) if edge_type_name(int(et)) == name], dtype=np.int64)
        if len(ids) == 0:
            continue
        take = len(ids) if int(per_type) <= 0 else min(len(ids), int(per_type))
        chosen = _choice(rng, ids, take)
        for eid in sorted(int(x) for x in chosen.tolist()):
            row = _edge_metadata(dataset, evidence, eid)
            row["probe_mode"] = "full_calibration"
            row["sample_count_for_type"] = int(take)
            row["available_count_for_type"] = int(len(ids))
            row["required_count_for_type"] = int(per_type)
            rows.append(row)
    return rows


def sample_temporal_sanity_edges(
    dataset,
    evidence: SupportEvidenceGraph,
    *,
    min_edges: int,
    seed: int,
    support_min: float,
    subgoal_horizon: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    same, dt = _edge_same_dt(dataset, evidence)
    ok = same & (dt > 0) & (dt <= int(subgoal_horizon)) & (evidence.support_score > float(support_min))
    ids = np.flatnonzero(ok)
    take = min(len(ids), int(min_edges))
    chosen = _choice(rng, ids, take)
    rows: list[Dict[str, Any]] = []
    for eid in sorted(int(x) for x in chosen.tolist()):
        row = _edge_metadata(dataset, evidence, eid)
        row["probe_mode"] = "temporal_sanity"
        row["temporal_sanity_support_min"] = float(support_min)
        row["subgoal_horizon"] = int(subgoal_horizon)
        row["sample_count_for_type"] = int(take)
        row["available_count_for_type"] = int(len(ids))
        row["required_count_for_type"] = int(min_edges)
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
        row["reset_success"] = 1
        ok, reason = try_set_state_from_observation(env, start_obs)
        row["set_state_success"] = int(ok)
        if not ok:
            row.update(
                {
                    "execution_mode": "rollout_reset_failed",
                    "reset_mode": reason,
                    "initial_dist": initial_dist,
                    "reach": float("nan"),
                    "progress": float("nan"),
                    "progress_norm": float("nan"),
                    "divergence": float("nan"),
                    "stuck": float("nan"),
                    "min_dist": float("nan"),
                    "final_dist": float("nan"),
                    "steps": 0,
                    "timeout": 0,
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
                "reset_success": int(row.get("reset_success", 0) or 0),
                "set_state_success": int(row.get("set_state_success", 0) or 0),
                "initial_dist": initial_dist,
                "reach": float("nan"),
                "progress": float("nan"),
                "progress_norm": float("nan"),
                "divergence": float("nan"),
                "stuck": float("nan"),
                "min_dist": float("nan"),
                "final_dist": float("nan"),
                "steps": 0,
                "timeout": 0,
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
                "gate": "STAGE29B_EDGE_EXECUTION_MEASUREMENT",
                "evidence_class": "edge_type_execution_probe",
                "edge_type": name,
                "count": int(len(part)),
                "reach_rate": _rate(r.get("reach") for r in part),
                "reset_success_rate": _rate(r.get("reset_success") for r in part),
                "set_state_success_rate": _rate(r.get("set_state_success") for r in part),
                "progress_mean": _mean(r.get("progress") for r in part),
                "progress_norm_mean": _mean(r.get("progress_norm") for r in part),
                "divergence_rate": _rate(r.get("divergence") for r in part),
                "stuck_rate": _rate(r.get("stuck") for r in part),
                "timeout_rate": _rate(r.get("timeout") for r in part),
                "support_score_mean": _mean(r.get("support_score") for r in part),
                "support_count_mean": _mean(r.get("support_count") for r in part),
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
                "gate": "STAGE29B_EDGE_EXECUTION_MEASUREMENT",
                "evidence_class": "support_score_p_exec_calibration",
                "support_bin_lo": float(lo),
                "support_bin_hi": float(hi),
                "count": int(len(part)),
                "reach_rate": _rate(r.get("reach") for r in part),
                "support_score_mean": _mean(r.get("support_score") for r in part),
                "support_count_mean": _mean(r.get("support_count") for r in part),
                "graph_p_exec_mean": _mean(r.get("graph_p_exec") for r in part),
                "reachability_p_exec_mean": _mean(r.get("reachability_p_exec") for r in part),
                "progress_norm_mean": _mean(r.get("progress_norm") for r in part),
            }
        )
    return out


def _count_by_type(rows: list[Dict[str, Any]]) -> Dict[str, int]:
    out = {name: 0 for name in REQUIRED_STAGE29B_EDGE_TYPES}
    for row in rows:
        name = str(row.get("edge_type", "UNKNOWN"))
        out[name] = out.get(name, 0) + 1
    return out


def _finite_rate(rows: list[Dict[str, Any]], key: str) -> float:
    return _rate(row.get(key) for row in rows)


def _monotonic_calibration(calibrations: list[Dict[str, Any]], *, min_bin_count: int, tolerance: float) -> tuple[bool, str]:
    vals = []
    for row in calibrations:
        count = int(_safe_float(row.get("count"), 0.0) or 0)
        reach = _safe_float(row.get("reach_rate"))
        score = _safe_float(row.get("support_score_mean"))
        if count >= int(min_bin_count) and math.isfinite(reach) and math.isfinite(score):
            vals.append((score, reach, count))
    vals.sort(key=lambda x: x[0])
    if len(vals) < 2:
        return False, "fewer_than_two_calibration_bins"
    for (s0, r0, _), (s1, r1, _) in zip(vals[:-1], vals[1:]):
        if r1 + float(tolerance) < r0:
            return False, f"non_monotonic:{s0:.4f}->{s1:.4f}:{r0:.4f}->{r1:.4f}"
    return True, "monotonic"


def validate_probe(
    rows: list[Dict[str, Any]],
    summaries: list[Dict[str, Any]],
    calibrations: list[Dict[str, Any]],
    availability: Dict[str, int],
    artifact_row: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    mode = str(args.probe_mode)
    reasons: list[str] = []
    status = EXECUTION_STATUS_PASS
    required_min = int(args.temporal_sanity_min_edges if mode == "temporal_sanity" else max(200, int(args.edges_per_type)))
    gas_status = str(artifact_row.get("gas_policy_status", "not_requested"))
    if not bool(args.rollout):
        reasons.append("rollout_not_requested")
        status = EXECUTION_STATUS_INSUFFICIENT
    elif gas_status != "loaded":
        reasons.append(f"missing_policy:{gas_status}")
        status = EXECUTION_STATUS_FAIL

    executed = [r for r in rows if str(r.get("execution_mode", "")) == "rollout"]
    reset_success_rate = _finite_rate(executed, "reset_success")
    set_state_success_rate = _finite_rate(executed, "set_state_success")
    reach_rate = _finite_rate(executed, "reach")
    progress_norm_mean = _mean(r.get("progress_norm") for r in executed)
    divergence_rate = _finite_rate(executed, "divergence")
    stuck_rate = _finite_rate(executed, "stuck")
    timeout_rate = _finite_rate(executed, "timeout")
    counts = _count_by_type(rows)

    if mode == "temporal_sanity":
        available = int(rows[0].get("available_count_for_type", 0)) if rows else 0
        if available < required_min or len(rows) < required_min or len(executed) < required_min:
            reasons.append(f"temporal_sanity_sample_{len(executed)}_lt_{required_min}_available_{available}")
            if status == EXECUTION_STATUS_PASS:
                status = EXECUTION_STATUS_INSUFFICIENT
        if status == EXECUTION_STATUS_PASS:
            if reset_success_rate < float(args.min_reset_success_rate):
                reasons.append(f"reset_success_rate_{reset_success_rate:.4f}_lt_{float(args.min_reset_success_rate):.4f}")
                status = EXECUTION_STATUS_FAIL
            if set_state_success_rate < float(args.min_reset_success_rate):
                reasons.append(f"set_state_success_rate_{set_state_success_rate:.4f}_lt_{float(args.min_reset_success_rate):.4f}")
                status = EXECUTION_STATUS_FAIL
            if reach_rate < float(args.temporal_sanity_min_reach_rate):
                reasons.append(f"temporal_reach_rate_{reach_rate:.4f}_lt_{float(args.temporal_sanity_min_reach_rate):.4f}")
                status = EXECUTION_STATUS_FAIL
            if progress_norm_mean < float(args.temporal_sanity_min_progress_norm):
                reasons.append(f"progress_norm_{progress_norm_mean:.4f}_lt_{float(args.temporal_sanity_min_progress_norm):.4f}")
                status = EXECUTION_STATUS_FAIL
            if divergence_rate > float(args.max_divergence_rate):
                reasons.append(f"divergence_rate_{divergence_rate:.4f}_gt_{float(args.max_divergence_rate):.4f}")
                status = EXECUTION_STATUS_FAIL
            if stuck_rate > float(args.max_stuck_rate):
                reasons.append(f"stuck_rate_{stuck_rate:.4f}_gt_{float(args.max_stuck_rate):.4f}")
                status = EXECUTION_STATUS_FAIL
            if timeout_rate > float(args.max_timeout_rate):
                reasons.append(f"timeout_rate_{timeout_rate:.4f}_gt_{float(args.max_timeout_rate):.4f}")
                status = EXECUTION_STATUS_FAIL
    else:
        missing_types = [name for name in REQUIRED_STAGE29B_EDGE_TYPES if int(availability.get(name, 0)) <= 0]
        insufficient_types = [
            name
            for name in REQUIRED_STAGE29B_EDGE_TYPES
            if int(availability.get(name, 0)) < required_min or int(counts.get(name, 0)) < required_min
        ]
        if missing_types:
            reasons.append("missing_required_edge_types:" + ",".join(missing_types))
            if status == EXECUTION_STATUS_PASS:
                status = EXECUTION_STATUS_INSUFFICIENT
        if insufficient_types:
            reasons.append("insufficient_edge_type_samples:" + ",".join(insufficient_types))
            if status == EXECUTION_STATUS_PASS:
                status = EXECUTION_STATUS_INSUFFICIENT
        if len(executed) < required_min * len(REQUIRED_STAGE29B_EDGE_TYPES):
            reasons.append(f"executed_edges_{len(executed)}_lt_{required_min * len(REQUIRED_STAGE29B_EDGE_TYPES)}")
            if status == EXECUTION_STATUS_PASS:
                status = EXECUTION_STATUS_INSUFFICIENT
        if status == EXECUTION_STATUS_PASS:
            ok, reason = _monotonic_calibration(
                calibrations,
                min_bin_count=int(args.calibration_min_bin_count),
                tolerance=float(args.monotonic_tolerance),
            )
            if not ok:
                reasons.append("support_reach_calibration_" + reason)
                status = EXECUTION_STATUS_FAIL

    gate = {
        EXECUTION_STATUS_PASS: "PASS_STAGE29B_EXECUTION_PROBE_VALIDATION",
        EXECUTION_STATUS_INSUFFICIENT: "INSUFFICIENT_SAMPLE_STAGE29B_EXECUTION_PROBE",
        EXECUTION_STATUS_FAIL: "FAIL_STAGE29B_EXECUTION_PROBE_VALIDATION",
    }[status]
    promotion_allowed = int(status == EXECUTION_STATUS_PASS and mode == "full_calibration")
    if promotion_allowed:
        promotion_conclusion = PROMOTION_READY
    elif status == EXECUTION_STATUS_PASS and mode == "temporal_sanity":
        promotion_conclusion = TEMPORAL_SANITY_READY
    else:
        promotion_conclusion = PROMOTION_BLOCKED
    return {
        "phase": "stage29b_execution_validation",
        "event": "completed",
        "gate": gate,
        "evidence_class": "stage29b_execution_probe_validity",
        "probe_mode": mode,
        "stage29a_offline_scg_status": "PASS",
        "execution_evidence_status": status,
        "promotion_allowed": promotion_allowed,
        "online_eval_allowed": promotion_allowed,
        "promotion_conclusion": promotion_conclusion,
        "validation_reasons": reasons,
        "required_edges_per_type": int(max(200, int(args.edges_per_type))),
        "temporal_sanity_min_edges": int(args.temporal_sanity_min_edges),
        "required_edge_types": REQUIRED_STAGE29B_EDGE_TYPES,
        "sampled_edges": int(len(rows)),
        "executed_edges": int(len(executed)),
        "edge_type_sample_counts": counts,
        "edge_type_available_counts": availability,
        "reset_success_rate": reset_success_rate,
        "set_state_success_rate": set_state_success_rate,
        "reach_rate": reach_rate,
        "progress_norm_mean": progress_norm_mean,
        "divergence_rate": divergence_rate,
        "stuck_rate": stuck_rate,
        "timeout_rate": timeout_rate,
    }


def _write_markdown_report(
    path: Path,
    rows: list[Dict[str, Any]],
    summaries: list[Dict[str, Any]],
    calibrations: list[Dict[str, Any]],
    artifact_row: Dict[str, Any],
    validation_row: Dict[str, Any],
) -> None:
    lines = [
        "# Stage29 Edge Execution Probe",
        "",
        "- Evidence class: edge execution probe for Stage29 typed evidence graph.",
        "- Stage29-A offline SCG status: PASS.",
        f"- Stage29-B execution evidence status: {validation_row.get('execution_evidence_status', EXECUTION_STATUS_INSUFFICIENT)}.",
        f"- Promotion conclusion: {validation_row.get('promotion_conclusion', PROMOTION_BLOCKED)}.",
        f"- Validation reasons: {validation_row.get('validation_reasons', [])}.",
        f"- Boundary loaded: {artifact_row.get('boundary_loaded', 0)}.",
        f"- Reachability loaded: {artifact_row.get('reachability_loaded', 0)}.",
        f"- GAS policy status: {artifact_row.get('gas_policy_status', 'not_requested')}.",
        f"- Probe mode: {validation_row.get('probe_mode', '')}.",
        f"- Sampled edges: {len(rows)}.",
        f"- Executed edges: {validation_row.get('executed_edges', 0)}.",
        "",
        "## Edge Type Summary",
        "",
    ]
    if summaries:
        header = ["edge_type", "count", "reach_rate", "reset_success_rate", "set_state_success_rate", "progress_norm_mean", "divergence_rate", "stuck_rate", "timeout_rate", "support_score_mean", "support_count_mean", "graph_p_exec_mean", "reachability_p_exec_mean"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in summaries:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    else:
        lines.append("No summary rows generated.")
    lines.extend(["", "## Calibration", ""])
    if calibrations:
        header = ["support_bin_lo", "support_bin_hi", "count", "reach_rate", "support_score_mean", "support_count_mean", "graph_p_exec_mean", "reachability_p_exec_mean"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in calibrations:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    else:
        lines.append("No calibration rows generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_edge_probe(dataset, embeddings: np.ndarray, base_graph, boundary, reach_model, cfg: Dict, logger: CSVLogger, args) -> None:
    evidence = build_support_evidence_graph(dataset, embeddings, base_graph, cfg)
    availability = edge_type_availability(evidence)
    subgoal_horizon = int(args.subgoal_horizon or cfg.get("stage29_support", {}).get("edge_probe_subgoal_horizon", cfg.get("reachability", {}).get("horizon", 50)))
    if str(args.probe_mode) == "temporal_sanity":
        rows = sample_temporal_sanity_edges(
            dataset,
            evidence,
            min_edges=int(args.temporal_sanity_min_edges),
            seed=int(cfg.get("seed", 0)) + 2917,
            support_min=float(args.temporal_sanity_support_min),
            subgoal_horizon=subgoal_horizon,
        )
    else:
        edges_per_type = max(200, int(args.edges_per_type))
        rows = sample_edges_by_type(
            dataset,
            evidence,
            per_type=edges_per_type,
            seed=int(cfg.get("seed", 0)) + 2917,
            edge_types=[x.strip() for x in args.edge_types.split(",") if x.strip()] if args.edge_types else REQUIRED_STAGE29B_EDGE_TYPES,
        )
    device = get_torch_device(str(cfg.get("device", "cpu")))
    _score_sample_with_reachability(rows, evidence, reach_model, device, cfg)
    for row in rows:
        row["boundary_loaded"] = int(boundary is not None)
        row["reachability_loaded"] = int(reach_model is not None)

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
        "probe_mode": str(args.probe_mode),
        "stage29a_offline_scg_status": "PASS",
        "edge_type_available_counts": availability,
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
                "gate": "STAGE29B_EDGE_EXECUTION_MEASUREMENT",
                "evidence_class": "edge_execution_probe" if row.get("execution_mode") == "rollout" else "edge_probe_sample",
                **row,
            }
        )
    summaries = summary_rows_by_type(rows)
    calibrations = calibration_rows(rows, bins=int(args.calibration_bins))
    validation = validate_probe(rows, summaries, calibrations, availability, artifact_row, args)
    for row in summaries:
        logger.log(row)
    for row in calibrations:
        logger.log(row)
    logger.log(validation)
    report_path = Path(args.report) if args.report else Path(logger.path).with_suffix(".md")
    _write_markdown_report(report_path, rows, summaries, calibrations, artifact_row, validation)


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
    parser.add_argument("--probe-mode", choices=["temporal_sanity", "full_calibration"], default="temporal_sanity")
    parser.add_argument("--edges-per-type", type=int, default=200)
    parser.add_argument("--edge-types", default=None, help="Comma-separated Stage29 edge type names. Defaults to all present types.")
    parser.add_argument("--calibration-bins", type=int, default=5)
    parser.add_argument("--calibration-min-bin-count", type=int, default=20)
    parser.add_argument("--monotonic-tolerance", type=float, default=0.05)
    parser.add_argument("--temporal-sanity-min-edges", type=int, default=200)
    parser.add_argument("--temporal-sanity-support-min", type=float, default=0.8)
    parser.add_argument("--temporal-sanity-min-reach-rate", type=float, default=0.6)
    parser.add_argument("--temporal-sanity-min-progress-norm", type=float, default=0.05)
    parser.add_argument("--min-reset-success-rate", type=float, default=0.95)
    parser.add_argument("--max-divergence-rate", type=float, default=0.10)
    parser.add_argument("--max-stuck-rate", type=float, default=0.25)
    parser.add_argument("--max-timeout-rate", type=float, default=0.25)
    parser.add_argument("--rollout", action="store_true", help="Run GAS policy rollouts from edge source observations. Without this, only sampling/load/calibration inputs are reported.")
    parser.add_argument("--gas-artifact-root", default=None)
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--gpu", default="cpu")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--subgoal-horizon", type=int, default=None)
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
