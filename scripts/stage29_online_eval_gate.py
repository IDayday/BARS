#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bars.common.config import apply_dotlist, load_json
from bars.common.device import get_torch_device
from bars.common.logging import CSVLogger
from bars.experiments.pipeline import _apply_routeb_backbone_config, _load_data
from bars.external.gas_backbone import GASBackbone
from bars.graph.planner import PlanResult, nearest_graph_node, plan_path
from bars.graph.stage29_support import (
    SupportEvidenceGraph,
    build_support_evidence_graph,
    edge_type_name,
    graph_support_summary,
    plan_support_budgeted,
    plan_support_lexicographic,
)
from bars.graph.types import BARSGraph
from scripts.stage29_edge_execution_probe import (
    EXECUTION_STATUS_PASS,
    REQUIRED_STAGE29B_EDGE_TYPES,
)
from scripts.stage29_support_calibrated_audit import (
    _load_cached_artifacts,
    _maybe_realign_cached_dataset,
    _merge_stage29_config,
)


DEFAULT_PLANNERS = [
    "BARS_BASE",
    "STAGE29_LEXICOGRAPHIC",
    "SUPPORT_BUDGET_K0",
    "SUPPORT_BUDGET_K1",
    "SUPPORT_BUDGET_K2",
]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _mean(values: Iterable[Any]) -> float:
    xs = [_safe_float(x) for x in values]
    xs = [x for x in xs if math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _rate(values: Iterable[Any]) -> float:
    return _mean(values)


def _quantile(values: Iterable[Any], q: float) -> float:
    xs = [_safe_float(x) for x in values]
    xs = [x for x in xs if math.isfinite(x)]
    return float(np.quantile(xs, q)) if xs else float("nan")


def _parse_csv(path: Path) -> list[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _last_phase(rows: Sequence[Dict[str, str]], phase: str) -> Dict[str, str]:
    matches = [r for r in rows if r.get("phase") == phase]
    if not matches:
        return {}
    return matches[-1]


def _rows_phase(rows: Sequence[Dict[str, str]], phase: str) -> list[Dict[str, str]]:
    return [r for r in rows if r.get("phase") == phase]


def _monotonic_calibration(rows: Sequence[Dict[str, str]], tolerance: float = 0.05) -> tuple[bool, float, float]:
    vals: list[tuple[float, float]] = []
    for row in rows:
        reach = _safe_float(row.get("reach_rate"))
        score = _safe_float(row.get("support_score_mean"), _safe_float(row.get("support_bin_lo")))
        if math.isfinite(reach) and math.isfinite(score):
            vals.append((score, reach))
    vals.sort(key=lambda x: x[0])
    if len(vals) < 2:
        return False, float("nan"), float("nan")
    ok = all(b[1] + float(tolerance) >= a[1] for a, b in zip(vals[:-1], vals[1:]))
    return ok, float(vals[0][1]), float(vals[-1][1])


def load_stage29b_gate(full_calibration_csv: Path, temporal_sanity_csv: Optional[Path] = None) -> Dict[str, Any]:
    full_rows = _parse_csv(full_calibration_csv)
    validation = _last_phase(full_rows, "stage29b_execution_validation")
    artifacts = _last_phase(full_rows, "stage29_edge_probe_artifacts")
    summaries = _rows_phase(full_rows, "stage29_edge_execution_summary")
    calibrations = _rows_phase(full_rows, "stage29_edge_calibration")
    if not validation:
        raise ValueError(f"{full_calibration_csv} has no stage29b_execution_validation row")
    if not artifacts:
        raise ValueError(f"{full_calibration_csv} has no stage29_edge_probe_artifacts row")

    summary_by_type = {str(r.get("edge_type", "")): r for r in summaries}
    counts = {name: _safe_int(summary_by_type.get(name, {}).get("count")) for name in REQUIRED_STAGE29B_EDGE_TYPES}
    missing = [name for name, count in counts.items() if count <= 0]
    insufficient = [name for name, count in counts.items() if count < 200]
    monotonic, bin_reach_min, bin_reach_max = _monotonic_calibration(calibrations)

    temporal_reach = float("nan")
    temporal_edges = 0
    if temporal_sanity_csv is not None and temporal_sanity_csv.exists():
        temporal_rows = _parse_csv(temporal_sanity_csv)
        temporal_validation = _last_phase(temporal_rows, "stage29b_execution_validation")
        temporal_summary = _last_phase(temporal_rows, "stage29_edge_execution_summary")
        temporal_reach = _safe_float(temporal_validation.get("reach_rate"), _safe_float(temporal_summary.get("reach_rate")))
        temporal_edges = _safe_int(temporal_validation.get("executed_edges"), _safe_int(temporal_summary.get("count")))

    status = str(validation.get("execution_evidence_status", ""))
    online_allowed = _safe_int(validation.get("online_eval_allowed")) == 1
    executed_edges = _safe_int(validation.get("executed_edges"))
    sampled_edges = _safe_int(validation.get("sampled_edges"))
    reasons: list[str] = []
    if status != EXECUTION_STATUS_PASS:
        reasons.append(f"stage29b_status_{status}")
    if not online_allowed:
        reasons.append("online_eval_allowed_0")
    if sampled_edges < 1000 or executed_edges < 1000:
        reasons.append(f"calibration_edges_sampled_{sampled_edges}_executed_{executed_edges}")
    if missing:
        reasons.append("missing_edge_types:" + ",".join(missing))
    if insufficient:
        reasons.append("insufficient_edge_type_samples:" + ",".join(insufficient))
    if not monotonic:
        reasons.append("support_score_calibration_not_monotonic")

    boundary_loaded = _safe_int(artifacts.get("boundary_loaded"))
    reachability_loaded = _safe_int(artifacts.get("reachability_loaded"))
    gate_status = "PASS" if not reasons else "FAIL"
    return {
        "phase": "stage29_online_eval_precondition",
        "event": "completed",
        "gate": "PASS_STAGE29B_EXECUTION_VALIDATION" if gate_status == "PASS" else "FAIL_STAGE29B_EXECUTION_VALIDATION",
        "evidence_class": "stage29b_execution_probe_validity",
        "stage29a_offline_scg_status": "PASS",
        "stage29b_execution_evidence_status": status,
        "stage29b_online_eval_allowed": int(online_allowed),
        "stage29b_promotion_conclusion": validation.get("promotion_conclusion", ""),
        "stage29b_gate_status": gate_status,
        "stage29b_gate_reasons": reasons,
        "stage29b_sampled_edges": sampled_edges,
        "stage29b_executed_edges": executed_edges,
        "stage29b_edges_per_required_type": counts,
        "support_score_calibration_signal": "VALIDATED" if gate_status == "PASS" else "NOT_VALIDATED_FOR_PROMOTION",
        "support_score_bin_reach_min": bin_reach_min,
        "support_score_bin_reach_max": bin_reach_max,
        "support_score_calibration_monotonic": int(monotonic),
        "temporal_sanity_high_support_reach": temporal_reach,
        "temporal_sanity_high_support_edges": temporal_edges,
        "supported_cross_bridge_reach": _safe_float(summary_by_type.get("SUPPORTED_CROSS_BRIDGE", {}).get("reach_rate")),
        "unsupported_shortcut_reach": _safe_float(summary_by_type.get("UNSUPPORTED_SHORTCUT", {}).get("reach_rate")),
        "boundary_loaded": boundary_loaded,
        "reachability_loaded": reachability_loaded,
        "boundary_validation_status": "UNVALIDATED_NOT_LOADED" if boundary_loaded == 0 else "LOADED_NOT_SEPARATELY_VALIDATED",
        "reachability_validation_status": "UNVALIDATED_NOT_LOADED" if reachability_loaded == 0 else "LOADED_NOT_SEPARATELY_VALIDATED",
        "full_calibration_report_file": str(full_calibration_csv),
        "temporal_sanity_report_file": str(temporal_sanity_csv or ""),
    }


@dataclass
class OnlinePlan:
    planner_id: str
    graph_id: str
    found: bool
    start_node: int
    goal_node: int
    node_path: list[int]
    edge_path: list[int]
    subgoal_phis: list[np.ndarray]
    execution_subgoal_phis: list[np.ndarray]
    execution_subgoal_edge_index: list[int]
    execution_subgoal_completes_edge: list[int]
    edge_type_sequence: list[str]
    support_scores: np.ndarray
    support_risks: np.ndarray
    unsupported_flags: np.ndarray
    cross_flags: np.ndarray
    metrics: Dict[str, Any]


def _default_fields(cfg: dict, run_dir: Path, out_path: Path, episodes: int) -> dict:
    return {
        "run_id": cfg.get("run_id", run_dir.name),
        "env": cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown")),
        "seed": cfg.get("seed", 0),
        "variant": cfg.get("planner", {}).get("variant", "stage29_support_calibrated_gas"),
        "stage": "stage29_online_eval_gate",
        "report_file": str(out_path),
        "baseline_graph_role": "sota_study_baseline_cached_gas_graph",
        "stage29a_offline_scg_status": "PASS",
        "fallback_mode": "none",
        "online_eval_scope": "20ep_only",
        "requested_episodes": int(episodes),
        "confirm_50ep_launched": 0,
    }


def _same_traj_flags(dataset, graph: BARSGraph, edges: Sequence[int]) -> np.ndarray:
    if not edges:
        return np.empty(0, dtype=bool)
    e = np.asarray(edges, dtype=np.int64)
    return dataset.traj_id[graph.node_indices[graph.src[e]]] == dataset.traj_id[graph.node_indices[graph.dst[e]]]


def _base_plan(dataset, graph: BARSGraph, initial_phi: np.ndarray, goal_phi: np.ndarray, cfg: Dict[str, Any]) -> OnlinePlan:
    scfg = cfg.get("stage29_support", {})
    start = nearest_graph_node(graph, initial_phi)
    goal = nearest_graph_node(graph, goal_phi)
    max_edges = int(scfg.get("max_path_edges", cfg.get("planner", {}).get("max_edges", 0))) or None
    variant = str(scfg.get("base_planner_variant", "reachability"))
    plan = plan_path(
        graph,
        start,
        goal,
        variant=variant,
        lambda_risk=float(scfg.get("lambda_risk", cfg.get("planner", {}).get("lambda_risk", 1.0))),
        max_edges=max_edges,
    )
    edge_path = [int(e) for e in plan.edge_path]
    same = _same_traj_flags(dataset, graph, edge_path)
    cross = ~same
    support_scores = np.full(len(edge_path), np.nan, dtype=np.float32)
    support_risks = np.full(len(edge_path), np.nan, dtype=np.float32)
    unsupported = cross.astype(bool)
    edge_types = ["BASE_SAME_TRAJ_EDGE" if bool(s) else "BASE_CROSS_TRAJ_EDGE" for s in same.tolist()]
    subgoals = [np.asarray(graph.node_embeddings[int(n)], dtype=np.float32) for n in plan.node_path[1:]] if plan.found else []
    metrics = {
        "found": int(plan.found),
        "base_planner_variant": variant,
        "num_edges": len(edge_path),
        "num_subgoals": max(0, len(plan.node_path) - 2),
        "execution_subgoal_count": len(subgoals),
        "densified_subgoal_count": 0,
        "total_cost": float(plan.total_cost),
        "total_risk": float(plan.total_risk),
        "objective": float(plan.objective),
        "path_cross_rate": float(cross.mean()) if len(cross) else float("nan"),
        "path_cross_edge_rate": float(cross.mean()) if len(cross) else float("nan"),
        "cross_edge_count": int(cross.sum()),
        "unsupported_edge_count": int(cross.sum()),
        "unsupported_edge_count_definition": "base_cross_edge_proxy",
        "support_risk": float("nan"),
        "path_unsupported_shortcut_rate": float(cross.mean()) if len(cross) else float("nan"),
    }
    return OnlinePlan(
        planner_id="BARS_BASE",
        graph_id="base_cached_gas_aligned",
        found=bool(plan.found),
        start_node=start,
        goal_node=goal,
        node_path=[int(n) for n in plan.node_path],
        edge_path=edge_path,
        subgoal_phis=subgoals,
        execution_subgoal_phis=subgoals,
        execution_subgoal_edge_index=list(range(len(edge_path))),
        execution_subgoal_completes_edge=[1] * len(edge_path),
        edge_type_sequence=edge_types,
        support_scores=support_scores,
        support_risks=support_risks,
        unsupported_flags=unsupported,
        cross_flags=cross,
        metrics=metrics,
    )


def _stage29_result_for_planner(dataset, evidence: SupportEvidenceGraph, planner_id: str, start: int, goal: int, cfg: Dict[str, Any], support_risk_budget: float):
    scfg = cfg.get("stage29_support", {})
    max_edges = int(scfg.get("max_path_edges", cfg.get("planner", {}).get("max_edges", 0))) or None
    if planner_id == "STAGE29_LEXICOGRAPHIC":
        return plan_support_lexicographic(
            dataset,
            evidence,
            start,
            goal,
            lambda_support_risk=float(scfg.get("lambda_support_risk", 1.0)),
            lambda_execution_risk=float(scfg.get("lambda_execution_risk", 1.0)),
            max_edges=max_edges,
            variant="STAGE29_LEXICOGRAPHIC",
        )
    if planner_id.startswith("SUPPORT_BUDGET_K"):
        k = int(planner_id.rsplit("K", 1)[-1])
        return plan_support_budgeted(
            dataset,
            evidence,
            start,
            goal,
            unsupported_budget=k,
            support_risk_budget=float(support_risk_budget),
            support_risk_bin=float(scfg.get("support_risk_bin", 0.05)),
            lambda_execution_risk=float(scfg.get("lambda_execution_risk", 1.0)),
            max_edges=max_edges,
            variant=planner_id,
        )
    raise ValueError(f"Unknown Stage29 planner: {planner_id}")


def _stage29_plan(
    dataset,
    evidence: SupportEvidenceGraph,
    embeddings: np.ndarray,
    initial_phi: np.ndarray,
    goal_phi: np.ndarray,
    planner_id: str,
    cfg: Dict[str, Any],
    support_risk_budget: float,
) -> OnlinePlan:
    graph = evidence.graph
    start = nearest_graph_node(graph, initial_phi)
    goal = nearest_graph_node(graph, goal_phi)
    res = _stage29_result_for_planner(dataset, evidence, planner_id, start, goal, cfg, support_risk_budget)
    plan: PlanResult = res.plan
    edge_path = [int(e) for e in plan.edge_path]
    e = np.asarray(edge_path, dtype=np.int64)
    same = _same_traj_flags(dataset, graph, edge_path)
    cross = ~same
    edge_types = [edge_type_name(int(evidence.edge_type[int(eid)])) for eid in edge_path]
    support_scores = evidence.support_score[e].astype(np.float32) if len(e) else np.empty(0, dtype=np.float32)
    support_risks = evidence.support_risk[e].astype(np.float32) if len(e) else np.empty(0, dtype=np.float32)
    unsupported = evidence.unsupported_shortcut[e].astype(bool) if len(e) else np.empty(0, dtype=bool)
    subgoals = [np.asarray(graph.node_embeddings[int(n)], dtype=np.float32) for n in plan.node_path[1:]] if plan.found else []
    exec_subgoals, exec_edge_index, exec_completes = _stage29_execution_subgoals(dataset, evidence, embeddings, edge_path, cfg) if plan.found else ([], [], [])
    metrics = res.to_row()
    metrics.update(
        {
            "found": int(plan.found),
            "num_edges": len(edge_path),
            "execution_subgoal_count": len(exec_subgoals),
            "densified_subgoal_count": max(0, len(exec_subgoals) - len(edge_path)),
            "path_cross_rate": float(cross.mean()) if len(cross) else float("nan"),
            "cross_edge_count": int(cross.sum()),
            "unsupported_edge_count": int(unsupported.sum()),
            "unsupported_edge_count_definition": "stage29_unsupported_shortcut",
            "support_risk_budget": float(support_risk_budget),
        }
    )
    return OnlinePlan(
        planner_id=planner_id,
        graph_id="stage29_support_calibrated",
        found=bool(plan.found),
        start_node=start,
        goal_node=goal,
        node_path=[int(n) for n in plan.node_path],
        edge_path=edge_path,
        subgoal_phis=subgoals,
        execution_subgoal_phis=exec_subgoals,
        execution_subgoal_edge_index=exec_edge_index,
        execution_subgoal_completes_edge=exec_completes,
        edge_type_sequence=edge_types,
        support_scores=support_scores,
        support_risks=support_risks,
        unsupported_flags=unsupported,
        cross_flags=cross,
        metrics=metrics,
    )


def _stage29_execution_subgoals(
    dataset,
    evidence: SupportEvidenceGraph,
    embeddings: np.ndarray,
    edge_path: Sequence[int],
    cfg: Dict[str, Any],
) -> tuple[list[np.ndarray], list[int], list[int]]:
    graph = evidence.graph
    scfg = cfg.get("stage29_support", {})
    enabled = bool(scfg.get("execution_densify_temporal_edges", False))
    step = int(scfg.get("execution_temporal_densify_step", cfg.get("reachability", {}).get("horizon", 50)))
    min_dt = int(scfg.get("execution_temporal_densify_min_dt", step))
    max_insert = int(scfg.get("execution_temporal_densify_max_insert_per_edge", 4))
    out_phis: list[np.ndarray] = []
    out_edge: list[int] = []
    out_complete: list[int] = []
    for edge_i, raw_eid in enumerate(edge_path):
        eid = int(raw_eid)
        src_node = int(graph.src[eid])
        dst_node = int(graph.dst[eid])
        src_idx = int(graph.node_indices[src_node])
        dst_idx = int(graph.node_indices[dst_node])
        same = bool(dataset.traj_id[src_idx] == dataset.traj_id[dst_idx])
        dt = int(dataset.timestep[dst_idx] - dataset.timestep[src_idx]) if same else -1
        if enabled and same and step > 0 and dt > min_dt and dst_idx > src_idx:
            inserted = 0
            for mid_idx in range(src_idx + step, dst_idx, step):
                if inserted >= max_insert:
                    break
                if int(dataset.traj_id[mid_idx]) != int(dataset.traj_id[src_idx]):
                    break
                out_phis.append(np.asarray(embeddings[mid_idx], dtype=np.float32))
                out_edge.append(edge_i)
                out_complete.append(0)
                inserted += 1
        out_phis.append(np.asarray(graph.node_embeddings[dst_node], dtype=np.float32))
        out_edge.append(edge_i)
        out_complete.append(1)
    return out_phis, out_edge, out_complete


def _make_plan(
    planner_id: str,
    dataset,
    base_graph: BARSGraph,
    evidence: SupportEvidenceGraph,
    embeddings: np.ndarray,
    initial_phi: np.ndarray,
    goal_phi: np.ndarray,
    cfg: Dict[str, Any],
    support_risk_budget: float,
) -> OnlinePlan:
    if planner_id == "BARS_BASE":
        return _base_plan(dataset, base_graph, initial_phi, goal_phi, cfg)
    if planner_id == "SUPPORT_PLUS_ENDPOINT":
        raise NotImplementedError("SUPPORT_PLUS_ENDPOINT is not implemented in this branch")
    return _stage29_plan(dataset, evidence, embeddings, initial_phi, goal_phi, planner_id, cfg, support_risk_budget)


def _info_success(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    ep_info = info.get("episode", {}) if isinstance(info.get("episode", {}), dict) else {}
    return bool(
        ep_info.get("success", False)
        or info.get("success", False)
        or info.get("goal_achieved", False)
        or info.get("is_success", False)
    )


def _dist_stats(prefix: str, values: Sequence[Any]) -> Dict[str, Any]:
    vals = [_safe_float(x) for x in values]
    vals = [x for x in vals if math.isfinite(x)]
    return {
        f"{prefix}_count": int(len(vals)),
        f"{prefix}_mean": float(np.mean(vals)) if vals else float("nan"),
        f"{prefix}_min": float(np.min(vals)) if vals else float("nan"),
        f"{prefix}_p10": float(np.quantile(vals, 0.10)) if vals else float("nan"),
        f"{prefix}_p50": float(np.quantile(vals, 0.50)) if vals else float("nan"),
        f"{prefix}_p90": float(np.quantile(vals, 0.90)) if vals else float("nan"),
        f"{prefix}_max": float(np.max(vals)) if vals else float("nan"),
    }


def _execute_episode(
    backbone: GASBackbone,
    env,
    env_name: str,
    dataset,
    base_graph: BARSGraph,
    evidence: SupportEvidenceGraph,
    embeddings: np.ndarray,
    cfg: Dict[str, Any],
    planner_id: str,
    task_id: int,
    episode_id: int,
    episode_seed: int,
    args,
) -> Dict[str, Any]:
    start_time = time.time()
    env, observation, goal, _, done, _ = backbone.setup_task_env(env, env_name, int(task_id), int(episode_seed), render_goal=False)
    actual_goal_phi = backbone.get_phi(goal)
    initial_phi = backbone.get_phi(observation)
    initial_goal_dist = float(np.linalg.norm(actual_goal_phi - initial_phi))
    planning_start = time.time()
    plan = _make_plan(
        planner_id,
        dataset,
        base_graph,
        evidence,
        embeddings,
        initial_phi,
        actual_goal_phi,
        cfg,
        float(args.support_risk_budget),
    )
    planning_sec = time.time() - planning_start

    row: Dict[str, Any] = {
        "phase": "stage29_online_eval_episode",
        "event": "episode_result",
        "gate": "PASS_STAGE29_20EP_ONLINE_EVAL_MEASUREMENT",
        "evidence_class": "controlled_20ep_online_eval",
        "planner_id": planner_id,
        "graph_id": plan.graph_id,
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "start_node": int(plan.start_node),
        "goal_node": int(plan.goal_node),
        "planning_duration_sec": planning_sec,
        "initial_goal_dist_phi": initial_goal_dist,
        **plan.metrics,
    }
    if not plan.found:
        row.update(
            {
                "success": 0,
                "return": 0.0,
                "steps": 0,
                "no_path": 1,
                "no_path_rate": 1.0,
                "subgoal_reach_rate": float("nan"),
                "edge_reach_rate": float("nan"),
                "edge_reach": float("nan"),
                "subgoal_reached_count": 0,
                "planned_subgoal_count": 0,
                "execution_subgoal_count": 0,
                "densified_subgoal_count": 0,
                "planned_edge_count": 0,
                "attempted_edge_count": 0,
                "executed_unsupported_edge_count": 0,
                "executed_support_risk": float("nan"),
                "timeout": 0,
                "stuck": 0,
                "divergence": 0,
                "first_failed_edge_type": "NO_GRAPH_PATH",
                "final_goal_dist_phi": initial_goal_dist,
                "progress_norm": 0.0,
                "duration_sec": time.time() - start_time,
                "false_shortcut_proxy_rate": float(row.get("path_unsupported_shortcut_rate", row.get("path_cross_rate", float("nan")))),
            }
        )
        row.update(_dist_stats("executed_support_score", []))
        return row

    max_steps = int(args.max_steps)
    final_goal_threshold = float(args.final_goal_threshold)
    subgoal_threshold = float(args.subgoal_reach_threshold)
    divergence_margin = float(args.divergence_margin)
    stuck_progress = float(args.stuck_progress_threshold)

    total_reward = 0.0
    steps = 0
    success = False
    subgoal_idx = 0
    edge_reached = np.zeros(len(plan.edge_path), dtype=bool)
    max_attempted_edge = 0
    min_goal_dist = initial_goal_dist
    final_goal_dist = initial_goal_dist
    last_info: Dict[str, Any] = {}

    while not done and steps < max_steps:
        phi_obs = backbone.get_phi(observation)
        final_goal_dist = float(np.linalg.norm(actual_goal_phi - phi_obs))
        min_goal_dist = min(min_goal_dist, final_goal_dist)
        if final_goal_dist <= final_goal_threshold:
            success = True
            break
        while subgoal_idx < len(plan.execution_subgoal_phis):
            subgoal_dist = float(np.linalg.norm(np.asarray(plan.execution_subgoal_phis[subgoal_idx]) - phi_obs))
            if subgoal_dist > subgoal_threshold:
                break
            edge_i = int(plan.execution_subgoal_edge_index[subgoal_idx]) if subgoal_idx < len(plan.execution_subgoal_edge_index) else subgoal_idx
            completes_edge = int(plan.execution_subgoal_completes_edge[subgoal_idx]) if subgoal_idx < len(plan.execution_subgoal_completes_edge) else 1
            if completes_edge and edge_i < len(edge_reached):
                edge_reached[edge_i] = True
            subgoal_idx += 1
        if subgoal_idx >= len(plan.execution_subgoal_phis):
            target_phi = actual_goal_phi
        else:
            target_phi = np.asarray(plan.execution_subgoal_phis[subgoal_idx], dtype=np.float32)
            edge_i = int(plan.execution_subgoal_edge_index[subgoal_idx]) if subgoal_idx < len(plan.execution_subgoal_edge_index) else subgoal_idx
            max_attempted_edge = max(max_attempted_edge, edge_i + 1)
        action = backbone.sample_action(observation, target_phi)
        observation, reward, done, info = backbone.step_env(env, env_name, action)
        last_info = info if isinstance(info, dict) else {}
        total_reward += float(reward)
        steps += 1
        success = bool(success or _info_success(last_info))

    final_phi = backbone.get_phi(observation)
    final_goal_dist = float(np.linalg.norm(actual_goal_phi - final_phi))
    min_goal_dist = min(min_goal_dist, final_goal_dist)
    success = bool(success or final_goal_dist <= final_goal_threshold or _info_success(last_info))
    reached_edges = int(edge_reached.sum())
    attempted_edges = int(min(len(plan.edge_path), max(max_attempted_edge, reached_edges)))
    if success:
        first_failed = ""
    elif len(plan.edge_path) == 0:
        first_failed = "FINAL_GOAL"
    elif reached_edges < len(plan.edge_path):
        failed_ids = np.flatnonzero(~edge_reached)
        failed_i = int(failed_ids[0]) if len(failed_ids) else reached_edges
        first_failed = plan.edge_type_sequence[failed_i] if failed_i < len(plan.edge_type_sequence) else "UNKNOWN_EDGE"
    else:
        first_failed = "FINAL_GOAL"

    attempted_slice = slice(0, attempted_edges)
    attempted_support_scores = plan.support_scores[attempted_slice]
    attempted_support_risks = plan.support_risks[attempted_slice]
    attempted_unsupported = plan.unsupported_flags[attempted_slice]
    progress_norm = (initial_goal_dist - final_goal_dist) / max(initial_goal_dist, 1e-6)
    row.update(
        {
            "success": int(success),
            "return": float(total_reward),
            "steps": int(steps),
            "no_path": 0,
            "no_path_rate": 0.0,
            "subgoal_reach_rate": float(reached_edges / max(1, len(plan.edge_path))) if plan.edge_path else 1.0,
            "edge_reach_rate": float(reached_edges / max(1, len(plan.edge_path))) if plan.edge_path else 1.0,
            "edge_reach": float(reached_edges / max(1, len(plan.edge_path))) if plan.edge_path else 1.0,
            "subgoal_reached_count": reached_edges,
            "planned_subgoal_count": len(plan.subgoal_phis),
            "execution_subgoal_count": len(plan.execution_subgoal_phis),
            "densified_subgoal_count": max(0, len(plan.execution_subgoal_phis) - len(plan.edge_path)),
            "planned_edge_count": len(plan.edge_path),
            "attempted_edge_count": attempted_edges,
            "executed_unsupported_edge_count": int(attempted_unsupported.sum()) if len(attempted_unsupported) else 0,
            "executed_support_risk": float(np.nansum(attempted_support_risks)) if len(attempted_support_risks) else float("nan"),
            "timeout": int(steps >= max_steps and not success),
            "stuck": int((not success) and progress_norm <= stuck_progress),
            "divergence": int(final_goal_dist > initial_goal_dist + divergence_margin),
            "env_done": int(bool(done)),
            "first_failed_edge_type": first_failed,
            "min_goal_dist_phi": min_goal_dist,
            "final_goal_dist_phi": final_goal_dist,
            "progress_norm": progress_norm,
            "duration_sec": time.time() - start_time,
            "false_shortcut_proxy_rate": float(row.get("path_unsupported_shortcut_rate", row.get("path_cross_rate", float("nan")))),
        }
    )
    row.update(_dist_stats("executed_support_score", attempted_support_scores.tolist()))
    return row


def _summary_rows(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    by_planner: dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        by_planner.setdefault(str(row.get("planner_id", "UNKNOWN")), []).append(row)
    for planner_id, part in sorted(by_planner.items()):
        out.append(
            {
                "phase": "stage29_online_eval_summary",
                "event": "completed",
                "gate": "PASS_STAGE29_20EP_ONLINE_EVAL_SUMMARY",
                "evidence_class": "controlled_20ep_online_eval_summary",
                "planner_id": planner_id,
                "episodes": len(part),
                "success_rate": _rate(r.get("success") for r in part),
                "no_path_rate": _rate(r.get("no_path") for r in part),
                "path_cross_rate": _mean(r.get("path_cross_rate") for r in part),
                "unsupported_edge_count_mean": _mean(r.get("unsupported_edge_count") for r in part),
                "executed_unsupported_edge_count_mean": _mean(r.get("executed_unsupported_edge_count") for r in part),
                "support_risk_mean": _mean(r.get("support_risk") for r in part),
                "executed_support_risk_mean": _mean(r.get("executed_support_risk") for r in part),
                "subgoal_reach_rate": _mean(r.get("subgoal_reach_rate") for r in part),
                "edge_reach_rate": _mean(r.get("edge_reach_rate") for r in part),
                "timeout_rate": _rate(r.get("timeout") for r in part),
                "stuck_rate": _rate(r.get("stuck") for r in part),
                "divergence_rate": _rate(r.get("divergence") for r in part),
                "first_failed_edge_type_mode": _mode(r.get("first_failed_edge_type") for r in part if r.get("first_failed_edge_type")),
                "executed_support_score_mean": _mean(r.get("executed_support_score_mean") for r in part),
                "executed_support_score_p50": _mean(r.get("executed_support_score_p50") for r in part),
                "false_shortcut_proxy_rate": _mean(r.get("false_shortcut_proxy_rate") for r in part),
            }
        )
    return out


def _mode(values: Iterable[Any]) -> str:
    counts: Dict[str, int] = {}
    for value in values:
        s = str(value)
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _write_report(path: Path, precondition: Dict[str, Any], summaries: Sequence[Dict[str, Any]], skipped: Sequence[str]) -> None:
    lines = [
        "# Stage29 20ep Online Eval Gate",
        "",
        "- Evidence class: controlled 20ep online evaluation with fallback_mode=none.",
        "- Stage29-A offline SCG status: PASS.",
        f"- Stage29-B execution evidence status: {precondition.get('stage29b_execution_evidence_status')}.",
        f"- Support score calibration signal: {precondition.get('support_score_calibration_signal')}.",
        f"- Boundary validation status: {precondition.get('boundary_validation_status')}.",
        f"- Reachability validation status: {precondition.get('reachability_validation_status')}.",
        "- 50ep confirm launched: 0.",
    ]
    if skipped:
        lines.append(f"- Skipped planners: {', '.join(skipped)}.")
    lines.extend(["", "## Summary", ""])
    if summaries:
        header = [
            "planner_id",
            "episodes",
            "success_rate",
            "no_path_rate",
            "path_cross_rate",
            "unsupported_edge_count_mean",
            "subgoal_reach_rate",
            "edge_reach_rate",
            "timeout_rate",
            "stuck_rate",
            "divergence_rate",
            "false_shortcut_proxy_rate",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in summaries:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in header) + " |")
    else:
        lines.append("No online eval summary rows generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_planners(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return vals or list(DEFAULT_PLANNERS)


def run_online_eval(dataset, embeddings: np.ndarray, base_graph: BARSGraph, cfg: Dict[str, Any], logger: CSVLogger, args) -> None:
    if int(args.episodes) != 20:
        raise ValueError("Stage29 online eval gate is 20ep-only; refusing a non-20 episode run.")
    precondition = load_stage29b_gate(Path(args.stage29b_full_calibration_csv), Path(args.stage29b_temporal_sanity_csv) if args.stage29b_temporal_sanity_csv else None)
    logger.log(precondition)
    if precondition.get("stage29b_gate_status") != "PASS":
        raise RuntimeError(f"Stage29-B gate is not PASS: {precondition.get('stage29b_gate_reasons')}")

    evidence = build_support_evidence_graph(dataset, embeddings, base_graph, cfg)
    logger.log(
        {
            "phase": "stage29_online_eval_graph_summary",
            "event": "completed",
            "gate": "PASS_STAGE29_TYPED_EVIDENCE_GRAPH_FOR_ONLINE_EVAL",
            "evidence_class": "support_calibrated_graph_summary",
            "graph_id": "stage29_support_calibrated",
            **graph_support_summary(dataset, evidence),
        }
    )

    dataset_dir = cfg.get("data", {}).get("dataset_dir")
    if dataset_dir:
        os.environ["OGBENCH_DATASET_DIR"] = os.path.expandvars(os.path.expanduser(str(dataset_dir)))
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    backbone = GASBackbone.load_or_train(
        cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown")),
        int(cfg.get("seed", 0)),
        args.gas_artifact_root,
        args.gas_repo_path,
        args.gpu,
        prefer_pretrained=False,
        train_if_missing=False,
        quick=True,
    )
    if backbone.agent is None or backbone.actor_fn is None:
        raise RuntimeError("GAS policy was not loaded; online eval is not valid without the low-level policy.")
    env, _, _ = backbone.load_env_and_dataset()
    env_name = cfg.get("data", {}).get("env_name", cfg.get("env_name", "unknown"))
    task_ids = backbone.get_task_ids(env)
    if not task_ids:
        task_ids = [1]

    requested = _parse_planners(str(args.planners))
    skipped: list[str] = []
    planners: list[str] = []
    for planner_id in requested:
        if planner_id == "SUPPORT_PLUS_ENDPOINT":
            skipped.append(planner_id)
            logger.log(
                {
                    "phase": "stage29_online_eval_planner_skipped",
                    "event": "skipped",
                    "gate": "SKIP_NOT_IMPLEMENTED_STAGE29_ONLINE_EVAL_PLANNER",
                    "evidence_class": "planner_availability",
                    "planner_id": planner_id,
                    "reason": "not_implemented",
                }
            )
        else:
            planners.append(planner_id)
    if not planners:
        raise ValueError("No implemented planners requested.")

    rows: list[Dict[str, Any]] = []
    seed = int(cfg.get("seed", 0))
    for episode_id in range(int(args.episodes)):
        task_id = int(task_ids[episode_id % len(task_ids)])
        episode_seed = int(seed * 1_000_003 + episode_id)
        for planner_id in planners:
            row = _execute_episode(
                backbone,
                env,
                env_name,
                dataset,
                base_graph,
                evidence,
                embeddings,
                cfg,
                planner_id,
                task_id,
                episode_id,
                episode_seed,
                args,
            )
            rows.append(row)
            logger.log(row)
    summaries = _summary_rows(rows)
    for row in summaries:
        logger.log(row)
    report_path = Path(args.report) if args.report else Path(logger.path).with_suffix(".md")
    _write_report(report_path, precondition, summaries, skipped)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage29-B controlled 20ep online eval gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--env", dest="env_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--planners", default=",".join(DEFAULT_PLANNERS))
    parser.add_argument("--support-risk-budget", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--final-goal-threshold", type=float, default=2.0)
    parser.add_argument("--subgoal-reach-threshold", type=float, default=2.0)
    parser.add_argument("--divergence-margin", type=float, default=0.5)
    parser.add_argument("--stuck-progress-threshold", type=float, default=0.05)
    parser.add_argument("--gas-artifact-root", default="artifacts/gas_ogbench_offline_full_20260522_165138")
    parser.add_argument("--gas-repo-path", default="external_src/GAS")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage29b-full-calibration-csv", required=True)
    parser.add_argument("--stage29b-temporal-sanity-csv", default=None)
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

    out_path = Path(args.out) if args.out else run_dir / "logs" / "stage29_online_eval_gate.csv"
    if args.clear and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_torch_device(str(cfg.get("device", "cpu")))
    embeddings, base_graph, _boundary, _reach_model, artifact_meta = _load_cached_artifacts(cfg, run_dir, device)
    _, dataset = _load_data(cfg)
    dataset, realign_meta = _maybe_realign_cached_dataset(cfg, dataset, embeddings, base_graph)
    artifact_meta = copy.deepcopy(artifact_meta)
    artifact_meta.update(realign_meta)
    artifact_meta.update(
        {
            "phase": "stage29_online_eval_cache_artifacts",
            "event": "completed",
            "gate": "PASS_STAGE29_ONLINE_EVAL_CACHE_ARTIFACT_LOAD",
            "evidence_class": "cache_artifact_reuse_for_online_eval",
            "stage29a_offline_scg_status": "PASS",
        }
    )
    logger = CSVLogger(str(out_path), _default_fields(cfg, run_dir, out_path, int(args.episodes)))
    logger.log(artifact_meta)
    run_online_eval(dataset, embeddings, base_graph, cfg, logger, args)
    print(str(out_path))


if __name__ == "__main__":
    main()
