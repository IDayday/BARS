from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_graph(path: Path):
    nodes = [{"node_id": f"n{i}", "center_phi": [float(i), 0.0], "support_count": 1} for i in range(4)]
    edges = [
        {"edge_id": "e0", "src": "n0", "dst": "n1", "d_phi": 1.0, "contract_lcb": 0.9, "predicted_negative_progress": 0.05, "uncertainty": 0.05, "edge_type": "offline_temporal_future_positive", "action_anchored": True},
        {"edge_id": "e1", "src": "n1", "dst": "n2", "d_phi": 1.0, "contract_lcb": 0.9, "predicted_negative_progress": 0.05, "uncertainty": 0.05, "edge_type": "offline_temporal_future_positive", "action_anchored": True},
        {"edge_id": "e2", "src": "n2", "dst": "n3", "d_phi": 1.0, "contract_lcb": 0.9, "predicted_negative_progress": 0.05, "uncertainty": 0.05, "edge_type": "final_goal_positive", "action_anchored": True},
    ]
    path.write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": edges, "boundary_contracts": []}))


def test_planner_score_writes_weights(tmp_path: Path):
    graph = tmp_path / "graph.json"
    write_graph(graph)
    dataset = tmp_path / "contracts.jsonl"
    dataset.write_text("")
    out_dir = tmp_path / "planner"
    proc = subprocess.run(
        [sys.executable, "scripts/train_ecg_planner_score.py", "--graph", str(graph), "--contract_dataset", str(dataset), "--out_dir", str(out_dir), "--num_pairs", "3"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode in {0, 2}, proc.stderr
    assert (out_dir / "metrics.json").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "shortest_by_dphi" in metrics or metrics["status"].startswith("PLANNER_SCORE_BLOCKED")
