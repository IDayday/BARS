from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_ecg_mechanism_script_handles_small_data_and_marks_inconclusive(tmp_path: Path):
    rows = [
        {
            "env_name": "toy",
            "variant": "gas",
            "success_rate_mean": 0.8,
            "segment_target_reach_rate_mean": 0.1,
            "mean_segment_progress_mean": 0.2,
            "final_goal_on_rate_mean": 0.8,
            "stall_count_mean": 0.0,
        },
        {
            "env_name": "toy",
            "variant": "cage_contract_rank",
            "success_rate_mean": 0.4,
            "segment_target_reach_rate_mean": 0.8,
            "mean_segment_progress_mean": 0.1,
            "final_goal_on_rate_mean": 0.3,
            "stall_count_mean": 20.0,
        },
    ]
    stage33 = tmp_path / "stage33.csv"
    stage34 = tmp_path / "stage34.csv"
    summary = tmp_path / "summary.csv"
    write_csv(stage33, rows)
    write_csv(stage34, rows)
    write_csv(summary, rows)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"num_examples": 2}), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"status": "ok", "total_examples": 2}), encoding="utf-8")
    out_csv = tmp_path / "laws.csv"
    out_json = tmp_path / "laws.json"
    out_md = tmp_path / "laws.md"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_cage_ecg_mechanism.py"),
            "--stage33_analysis",
            str(stage33),
            "--stage34_analysis",
            str(stage34),
            "--stage34_summary",
            str(summary),
            "--contract_metrics",
            str(metrics),
            "--split_summary",
            str(split),
            "--out_csv",
            str(out_csv),
            "--out_json",
            str(out_json),
            "--out_md",
            str(out_md),
        ],
        cwd=str(ROOT),
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["laws"]
    assert any(row["status"] == "INCONCLUSIVE" for row in payload["laws"])
    assert out_md.exists()
