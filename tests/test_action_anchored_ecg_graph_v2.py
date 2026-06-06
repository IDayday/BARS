from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_action_anchored_graph_has_no_knn_main_edges(tmp_path: Path):
    dataset = tmp_path / "contracts.jsonl"
    rows = []
    for i in range(8):
        rows.append(
            {
                "env_name": "antmaze-giant-navigate-v0",
                "phi_s": [float(i), 0.0],
                "phi_g": [float(i + 1), 0.0],
                "d_phi": 1.0,
                "target_mode": "final_goal" if i == 7 else "offline_future_positive",
                "horizon": 1,
                "label_positive_contract": True,
                "label_final_goal": i == 7,
                "action_available": True,
                "action_source": "offline_dataset",
            }
        )
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    model = tmp_path / "model.pt"
    model.write_text(json.dumps({"model_type": "linear_logistic_contract", "mean": [0.0] * 20, "std": [1.0] * 20, "weights": {"contract_positive": [0.0] * 21, "negative_progress": [-10.0] * 21, "hit": [0.0] * 21}}))
    out_dir = tmp_path / "graph"
    proc = subprocess.run(
        [sys.executable, "scripts/build_action_anchored_ecg_graph_v2.py", "--action_contract_dataset", str(dataset), "--contract_model", str(model), "--out_dir", str(out_dir), "--min_contract_lcb", "0.0", "--clear"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    graph = json.loads((out_dir / "contract_graph.json").read_text())
    assert graph["edges"]
    assert all(edge["action_anchored"] for edge in graph["edges"])
    assert not any("knn" in str(edge.get("edge_type", "")).lower() for edge in graph["edges"])
