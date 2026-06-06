from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external_src" / "GAS"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cage_eval_command import build_eval_command  # noqa: E402
from cage.ecg_planner_runtime import ECGPlannerRuntime  # noqa: E402
from cage.ecg_policy_adapter import ECGPolicyAdapter  # noqa: E402


def write_graph(path: Path):
    payload = {
        "metadata": {},
        "nodes": [
            {"node_id": "n0", "center_phi": [0.0, 0.0], "support_count": 1},
            {"node_id": "n1", "center_phi": [1.0, 0.0], "support_count": 1},
        ],
        "edges": [
            {"edge_id": "e0", "src": "n0", "dst": "n1", "d_phi": 1.0, "contract_lcb": 0.8, "predicted_negative_progress": 0.1, "uncertainty": 0.1, "edge_type": "offline_temporal_future_positive", "action_anchored": True}
        ],
        "boundary_contracts": [],
    }
    path.write_text(json.dumps(payload))


def test_ecg_runtime_selects_trace_only_gas_target(tmp_path: Path):
    graph_path = tmp_path / "graph.json"
    write_graph(graph_path)
    runtime = ECGPlannerRuntime.from_paths(graph_path)
    sel = runtime.select_target(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.0]), 0, trace_only=True)
    assert np.allclose(sel.target_phi, [0.5, 0.0])
    assert sel.trace["ecg_trace_only"] is True


def test_ecg_policy_adapter_predicts(tmp_path: Path):
    model_path = tmp_path / "adapter.pt"
    model_path.write_text(json.dumps({"model_type": "linear_ecg_policy_adapter", "mean": [0.0] * 10, "std": [1.0] * 10, "weights": [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], "bias": [0.25]}))
    adapter = ECGPolicyAdapter.from_path(model_path)
    action = adapter.predict(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert action.shape == (1,)
    assert float(action[0]) == 0.25


def test_ecg_eval_command_contains_paths(tmp_path: Path):
    row = {
        "variant": "cage_ecg_adapter",
        "job_id": "job",
        "env_name": "antmaze-giant-navigate-v0",
        "seed": 42,
        "output_root": str(tmp_path),
        "keygraph_path": str(tmp_path / "keygraph.pkl"),
        "policy_path": str(tmp_path / "params_1000000.pkl"),
        "cage_trace_path": str(tmp_path / "trace.jsonl"),
        "result_path": str(tmp_path / "eval.csv"),
        "episodes_per_goal": 1,
        "goals_per_env": 1,
        "ecg_graph_path": str(tmp_path / "graph.json"),
        "ecg_planner_score_path": str(tmp_path / "weights.json"),
        "ecg_policy_adapter_path": str(tmp_path / "adapter.pt"),
        "ecg_contract_model_path": str(tmp_path / "contract.pt"),
    }
    cmd = build_eval_command(row)
    assert "--cage_ecg_adapter" in cmd
    assert "--ecg_graph_path" in cmd
    assert "--ecg_policy_adapter_path" in cmd
