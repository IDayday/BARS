from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_cage_eval_command import UnsupportedVariant, build_eval_command, validate_variant


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_manifest_generation_creates_expected_jobs(tmp_path: Path):
    manifest_path = tmp_path / "out" / "manifests" / "focused_manifest.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cage_experiment_manifest.py"),
        "--checkpoint_root",
        str(tmp_path / "checkpoints"),
        "--output_root",
        str(tmp_path / "out"),
        "--envs",
        "antmaze-giant-navigate-v0",
        "--seeds",
        "0",
        "1",
        "--variants",
        "gas",
        "cage_trace_only",
        "cage_fixed_commit",
        "cage_drift_only",
        "cage_recovery_only",
        "cage_full",
        "cage_safe_full",
        "cage_contract_rank",
        "--episodes_per_goal",
        "1",
        "--goals_per_env",
        "1",
        "--manifest_path",
        str(manifest_path),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    rows = load_jsonl(manifest_path)

    assert len(rows) == 16
    for row in rows:
        assert row["job_id"]
        assert row["env_name"] == "antmaze-giant-navigate-v0"
        assert row["variant"]
        assert row["command"]
        assert row["cage_trace_path"]
        assert row["result_path"]


def test_command_builder_cage_flags_only_for_cage_variants(tmp_path: Path):
    base = {
        "job_id": "job",
        "env_name": "antmaze-giant-navigate-v0",
        "seed": 0,
        "checkpoint_root": str(tmp_path),
        "output_root": str(tmp_path / "out"),
        "keygraph_path": str(tmp_path / "graph" / "keygraph.pkl"),
        "policy_path": str(tmp_path / "policy" / "params_1000000.pkl"),
        "cage_trace_path": str(tmp_path / "trace.jsonl"),
        "result_path": str(tmp_path / "eval.csv"),
        "episodes_per_goal": 1,
        "goals_per_env": 1,
    }

    gas_cmd = build_eval_command({**base, "variant": "gas"})
    full_cmd = build_eval_command({**base, "variant": "cage_full"})
    fixed_cmd = build_eval_command({**base, "variant": "cage_fixed_commit"})
    trace_only_cmd = build_eval_command({**base, "variant": "cage_trace_only"})
    safe_cmd = build_eval_command({**base, "variant": "cage_safe_full"})
    rank_cmd = build_eval_command(
        {
            **base,
            "variant": "cage_contract_rank",
            "cage_debug_light": True,
            "cage_trace_phi_vectors": False,
        }
    )

    assert "--use_cage" not in gas_cmd
    assert "--use_cage" in full_cmd
    assert "--cage_trace_path" in full_cmd
    assert "--cage_disable_recovery" in fixed_cmd
    assert "--cage_disable_final_phase_controller" in fixed_cmd
    assert "--use_cage" in trace_only_cmd
    assert "--cage_trace_only" in trace_only_cmd
    assert "--cage_disable_recovery" not in trace_only_cmd
    assert "--cage_enable_churn_guard" in safe_cmd
    assert "--cage_disable_recovery_after_churn" in safe_cmd
    assert "--cage_enable_churn_guard" not in full_cmd
    assert "--cage_contract_rank" in rank_cmd
    assert "--cage_debug_light" in rank_cmd
    assert "--cage_trace_phi_vectors=false" in rank_cmd


def test_unsupported_variant_fails_clearly():
    with pytest.raises(UnsupportedVariant, match="risk-aware path"):
        validate_variant("cage_risk_path")
