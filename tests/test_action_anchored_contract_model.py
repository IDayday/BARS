from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_action_anchored_contract_model_trains(tmp_path: Path):
    dataset = tmp_path / "contracts.jsonl"
    rows = []
    for i in range(80):
        positive = i < 40
        phi_s = [0.0, 0.0]
        phi_g = [float(i % 7), 0.0] if positive else [float(i % 7), 1.0]
        rows.append(
            {
                "phi_s": phi_s,
                "phi_g": phi_g,
                "d_phi": 1.0,
                "target_mode": "offline_future_positive" if positive else "hard_negative",
                "env_name": "antmaze-giant-navigate-v0",
                "label_positive_contract": positive,
                "label_negative_contract": not positive,
                "label_final_goal": False,
                "action_available": True,
            }
        )
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    out_dir = tmp_path / "model"
    proc = subprocess.run(
        [sys.executable, "scripts/train_action_anchored_ecg_contract.py", "--dataset", str(dataset), "--out_dir", str(out_dir), "--epochs", "2"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["status"] == "CONTRACT_MODEL_READY"
    assert (out_dir / "model.pt").exists()
