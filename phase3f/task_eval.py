from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_preflight_status(dataset_name: str, preflight_root: str | Path = "results/phase3/env_preflight") -> dict[str, Any]:
    key = dataset_name.replace("-v0", "").replace("-", "_")
    path = Path(preflight_root) / key / "preflight_summary.json"
    if not path.exists():
        return {"status": "env_unavailable", "failure_reason": f"missing_preflight_summary:{path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def write_env_unavailable_skip(output_dir: str | Path, dataset_name: str, method: str, reason: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "method": method,
                "num_episodes": 0,
                "success_rate": 0.0,
                "skipped": True,
                "skipped_reason": reason,
            }
        ]
    ).to_csv(out / "task_rollout_summary.csv", index=False)
    pd.DataFrame([{"failure_reason": reason, "count": 1}]).to_csv(
        out / "failure_reason_summary.csv",
        index=False,
    )
    (out / "episode_traces.jsonl").write_text("", encoding="utf-8")
