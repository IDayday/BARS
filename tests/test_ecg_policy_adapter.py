from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_ecg_policy_adapter_beats_mean_baseline(tmp_path: Path):
    dataset = tmp_path / "contracts.jsonl"
    rows = []
    for i in range(120):
        obs = [float(i % 3), float(i % 5)]
        phi_s = [float(i) / 100.0, 0.0]
        phi_g = [phi_s[0] + 0.1, 0.0]
        action = [obs[0] * 0.2 + phi_g[0], obs[1] * -0.1]
        rows.append(
            {
                "env_name": "antmaze-giant-navigate-v0",
                "observation": obs,
                "phi_s": phi_s,
                "phi_g": phi_g,
                "action": action,
                "label_positive_contract": True,
                "action_available": True,
                "trainable_for_bc": True,
            }
        )
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    contract_model = tmp_path / "contract.pt"
    contract_model.write_text("{}")
    out_dir = tmp_path / "adapter"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/train_ecg_policy_adapter.py",
            "--dataset",
            str(dataset),
            "--contract_model",
            str(contract_model),
            "--out_dir",
            str(out_dir),
            "--envs",
            "antmaze-giant-navigate-v0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["gate"]["beats_mean_action_baseline"] is True
    assert (out_dir / "model.pt").exists()
