from __future__ import annotations

import numpy as np

from bars.data.toy_dataset import make_toy_dataset
from bars.graph.stage29_support import (
    TEMPORAL_BACKBONE,
    UNSUPPORTED_SHORTCUT,
    build_support_evidence_graph,
    edge_type_counts,
    plan_support_budgeted,
    plan_support_lexicographic,
)
from bars.graph.types import BARSGraph, EDGE_KIND_KNN, EDGE_KIND_TEMPORAL
from scripts.stage29_edge_execution_probe import calibration_rows, sample_edges_by_type, summary_rows_by_type


def _toy_base_graph(dataset, embeddings):
    node_indices = np.arange(0, dataset.size, 3, dtype=np.int64)
    src = np.arange(len(node_indices) - 1, dtype=np.int64)
    dst = src + 1
    # Add an optimistic cross-trajectory shortcut to verify it is typed, not banned.
    src = np.r_[src, 0]
    dst = np.r_[dst, len(node_indices) - 1]
    cost = np.ones(len(src), dtype=np.float32)
    p_exec = np.full(len(src), 0.8, dtype=np.float32)
    risk = -np.log(p_exec).astype(np.float32)
    kind = np.full(len(src), EDGE_KIND_TEMPORAL, dtype=np.int32)
    kind[-1] = EDGE_KIND_KNN
    return BARSGraph(node_indices, embeddings[node_indices], src, dst, cost, risk, p_exec, kind)


def test_stage29_builds_typed_evidence_graph():
    dataset = make_toy_dataset(num_traj=4, length=18, seed=0)
    embeddings = dataset.observations.copy()
    graph = _toy_base_graph(dataset, embeddings)
    cfg = {
        "seed": 0,
        "stage29_support": {
            "max_nodes": 64,
            "endpoint_max_nodes": 16,
            "trajectory_anchor_nodes": 32,
            "trajectory_anchor_stride": 4,
            "projection_change_nodes": 16,
            "bottleneck_nodes": 8,
            "projected_support_offsets": [1, 2, 4],
            "projected_support_top_per_src": 8,
            "temporal_backbone_connect": 4,
        },
    }
    ev = build_support_evidence_graph(dataset, embeddings, graph, cfg)
    counts = edge_type_counts(ev)
    assert ev.graph.num_nodes >= graph.num_nodes
    assert ev.graph.num_edges > graph.num_edges
    assert counts["TEMPORAL_BACKBONE"] > 0
    assert ev.support_score.shape[0] == ev.graph.num_edges
    assert ev.edge_type.shape[0] == ev.graph.num_edges


def test_stage29_support_planners_return_rows():
    dataset = make_toy_dataset(num_traj=3, length=15, seed=1)
    embeddings = dataset.observations.copy()
    graph = _toy_base_graph(dataset, embeddings)
    cfg = {
        "seed": 1,
        "stage29_support": {
            "max_nodes": 48,
            "endpoint_max_nodes": 12,
            "trajectory_anchor_nodes": 24,
            "trajectory_anchor_stride": 3,
            "projected_support_offsets": [1, 2, 4],
            "projected_support_top_per_src": 8,
            "temporal_backbone_connect": 4,
            "support_risk_bin": 0.1,
        },
    }
    ev = build_support_evidence_graph(dataset, embeddings, graph, cfg)
    start = 0
    goal = ev.graph.num_nodes - 1
    lex = plan_support_lexicographic(dataset, ev, start, goal, max_edges=12)
    budget = plan_support_budgeted(dataset, ev, start, goal, unsupported_budget=2, support_risk_budget=2.0, support_risk_bin=0.1, max_edges=12)
    assert "unsupported_edges" in lex.to_row()
    assert "support_risk" in budget.to_row()
    assert np.isfinite(budget.to_row()["support_risk"]) or not budget.plan.found


def test_stage29_edge_probe_samples_and_summarizes_by_type():
    dataset = make_toy_dataset(num_traj=3, length=15, seed=2)
    embeddings = dataset.observations.copy()
    graph = _toy_base_graph(dataset, embeddings)
    cfg = {
        "seed": 2,
        "stage29_support": {
            "max_nodes": 48,
            "endpoint_max_nodes": 12,
            "trajectory_anchor_nodes": 24,
            "trajectory_anchor_stride": 3,
            "projected_support_offsets": [1, 2, 4],
            "projected_support_top_per_src": 8,
            "temporal_backbone_connect": 4,
        },
    }
    ev = build_support_evidence_graph(dataset, embeddings, graph, cfg)
    rows = sample_edges_by_type(dataset, ev, per_type=3, seed=0)
    assert rows
    assert {"edge_id", "edge_type", "support_score", "graph_p_exec"}.issubset(rows[0])
    for row in rows:
        row["reach"] = 1 if row["support_score"] >= 0.5 else 0
        row["progress_norm"] = row["support_score"]
        row["divergence"] = 0
        row["stuck"] = 0
    summaries = summary_rows_by_type(rows)
    calibrations = calibration_rows(rows, bins=3)
    assert summaries
    assert calibrations
    assert all("reach_rate" in row for row in summaries)
