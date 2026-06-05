from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_aggregation_handles_missing_fields_and_paired_delta(tmp_path: Path):
    output_root = tmp_path / "out"
    manifest_path = output_root / "manifests" / "focused_manifest.jsonl"
    gas_trace = output_root / "jobs" / "gas_trace.jsonl"
    cage_trace = output_root / "jobs" / "cage_trace.jsonl"
    failed_trace = output_root / "jobs" / "failed_trace.jsonl"

    write_jsonl(
        gas_trace,
        [
            {
                "record_type": "episode",
                "env_name": "antmaze-giant-navigate-v0",
                "seed": 0,
                "task_id": 1,
                "success": 0,
                "target_switch_count": 4,
                "stall_count": 2,
                "drift_count": 3,
                "recovery_attempt_count": 0,
                "segment_target_reach_rate": 0.25,
                "final_goal_stall_count": 1,
            }
        ],
    )
    write_jsonl(
        cage_trace,
        [
            {
                "record_type": "episode",
                "env_name": "antmaze-giant-navigate-v0",
                "seed": 0,
                "task_id": 1,
                "success": 1,
                "normalized_score": 0.8,
                "no_path": False,
                "target_switch_count": 2,
                "early_switch_count": 0,
                "mean_commitment_length": 5,
                "stall_count": 1,
                "drift_count": 1,
                "recovery_attempt_count": 1,
                "recovery_success_count": 1,
                "global_replan_request_count": 0,
                "segment_target_reach_rate": 0.75,
                "mean_segment_progress": 0.4,
                "mean_distance_to_path": 1.5,
                "final_goal_on_step": 12,
                "final_goal_switch_count": 0,
                "final_goal_stall_count": 0,
                "timeout": False,
            }
        ],
    )
    manifest_rows = [
        {
            "job_id": "gas0",
            "env_name": "antmaze-giant-navigate-v0",
            "seed": 0,
            "variant": "gas",
            "status": "initialized",
            "cage_trace_path": str(gas_trace),
            "result_path": str(output_root / "jobs" / "gas.csv"),
        },
        {
            "job_id": "cage0",
            "env_name": "antmaze-giant-navigate-v0",
            "seed": 0,
            "variant": "cage_full",
            "status": "initialized",
            "cage_trace_path": str(cage_trace),
            "result_path": str(output_root / "jobs" / "cage.csv"),
        },
        {
            "job_id": "failed1",
            "env_name": "antmaze-giant-navigate-v0",
            "seed": 1,
            "variant": "cage_full",
            "status": "initialized",
            "cage_trace_path": str(failed_trace),
            "result_path": str(output_root / "jobs" / "failed.csv"),
        },
    ]
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(
        output_root / "status" / "focused_status.jsonl",
        [
            {"job_id": "gas0", "status": "succeeded"},
            {"job_id": "cage0", "status": "succeeded"},
            {"job_id": "failed1", "status": "failed", "return_code": 1},
        ],
    )

    out_json = output_root / "tables" / "focused_summary.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_cage_experiments.py"),
        "--input_root",
        str(output_root),
        "--manifest_path",
        str(manifest_path),
        "--out_csv",
        str(output_root / "tables" / "focused_summary.csv"),
        "--out_md",
        str(output_root / "tables" / "focused_summary.md"),
        "--out_json",
        str(out_json),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    data = json.loads(out_json.read_text(encoding="utf-8"))
    grouped = {(row["env_name"], row["variant"]): row for row in data["grouped"]}
    assert grouped[("antmaze-giant-navigate-v0", "gas")]["success_rate_mean"] == 0
    assert grouped[("antmaze-giant-navigate-v0", "cage_full")]["num_jobs"] == 2
    assert grouped[("antmaze-giant-navigate-v0", "cage_full")]["num_succeeded"] == 1
    assert grouped[("antmaze-giant-navigate-v0", "cage_full")]["target_switch_count_mean"] == 2

    paired = data["paired"][0]
    assert paired["variant"] == "cage_full"
    assert paired["matched_pairs"] == 1
    assert paired["delta_success_rate"] == 1
    assert paired["delta_target_switch_count"] == -2
