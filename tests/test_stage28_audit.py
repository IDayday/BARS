from __future__ import annotations

import numpy as np

from bars.common.logging import CSVLogger
from bars.data.toy_dataset import make_toy_dataset
from bars.graph.audit import build_audit_graph_variants, run_graph_method_audit
from bars.graph.types import BARSGraph, EDGE_KIND_TEMPORAL


def _toy_graph(dataset, embeddings):
    node_indices = np.arange(0, dataset.size, 3, dtype=np.int64)
    src = np.arange(len(node_indices) - 1, dtype=np.int64)
    dst = src + 1
    cost = np.ones(len(src), dtype=np.float32)
    p = np.full(len(src), 0.8, dtype=np.float32)
    risk = -np.log(p).astype(np.float32)
    kind = np.full(len(src), EDGE_KIND_TEMPORAL, dtype=np.int32)
    return BARSGraph(node_indices, embeddings[node_indices], src, dst, cost, risk, p, kind)


def test_build_stage28_audit_variants():
    dataset = make_toy_dataset(num_traj=4, length=12, seed=0)
    embeddings = dataset.observations.copy()
    graph = _toy_graph(dataset, embeddings)
    cfg = {
        "seed": 0,
        "stage28_audit": {
            "graph_variants": ["base_cached", "projection_temporal", "endpoint_aug"],
            "num_future_pairs": 4,
            "num_cross_pairs": 2,
            "edge_knn": 4,
        },
    }
    variants = build_audit_graph_variants(dataset, embeddings, graph, cfg)
    assert "base_cached" in variants
    assert "projection_temporal" in variants
    assert variants["projection_temporal"].num_edges > 0
    assert variants["endpoint_aug"].num_nodes >= variants["base_cached"].num_nodes


def test_run_stage28_audit_logs(tmp_path):
    dataset = make_toy_dataset(num_traj=4, length=12, seed=1)
    embeddings = dataset.observations.copy()
    graph = _toy_graph(dataset, embeddings)
    cfg = {
        "seed": 1,
        "stage28_audit": {
            "graph_variants": ["base_cached", "projection_temporal"],
            "num_future_pairs": 4,
            "num_cross_pairs": 2,
            "edge_knn": 4,
            "enable_path_diversity_probe": False,
        },
    }
    logger = CSVLogger(str(tmp_path / "stage28_graph_audit.csv"), {"env": "toy", "seed": 1})
    out = run_graph_method_audit(dataset, embeddings, graph, cfg, logger)
    assert "base_cached" in out
    text = (tmp_path / "stage28_graph_audit.csv").read_text()
    assert "stage28_graph_summary" in text
    assert "stage28_failure_taxonomy_proxy" in text
    assert "PASS_STAGE28_GRAPH_COUNTERFACTUALS" in text
    assert "path_search_counterfactual" in text
    assert "report_file" in text
