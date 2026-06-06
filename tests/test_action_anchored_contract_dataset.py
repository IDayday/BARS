from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_action_anchored_dataset_from_npz(tmp_path: Path):
    env = "antmaze-giant-navigate-v0"
    root = tmp_path / "datasets" / env / "gas_seed0"
    root.mkdir(parents=True)
    n = 20
    obs = np.random.randn(n, 5).astype("float32")
    actions = np.random.randn(n, 2).astype("float32")
    traj_ids = np.repeat(np.arange(2), n // 2)
    terminals = np.zeros(n, dtype="float32")
    terminals[9] = 1
    terminals[-1] = 1
    phi = np.random.randn(n, 3).astype("float32")
    np.savez(root / "dataset.npz", observations=obs, next_observations=np.roll(obs, -1, axis=0), actions=actions, traj_ids=traj_ids, terminals=terminals, tdr_emb=phi)
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/build_action_anchored_contract_dataset.py",
        "--envs",
        env,
        "--dataset_roots",
        str(tmp_path / "datasets"),
        "--checkpoint_root",
        str(tmp_path),
        "--out_dir",
        str(out_dir),
        "--max_rows_per_env",
        "64",
        "--positive_horizons",
        "1",
        "2",
        "--negative_samples_per_state",
        "1",
        "--final_goal_samples_per_traj",
        "1",
        "--clear",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out_dir / "dataset_summary.json").read_text())
    assert summary["positive_with_action_count"] > 0
    assert summary["action_supervision_rate"] > 0
    first = json.loads((out_dir / "action_contracts.jsonl").read_text().splitlines()[0])
    assert first["action_available"] is True
    assert first["action_source"] == "offline_dataset"
