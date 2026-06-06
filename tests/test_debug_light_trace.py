from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_src" / "GAS"))

from cage.config import CAGEConfig
from cage.state_machine import CAGEController


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_debug_light_drops_state_refs_and_phi_vectors_and_caps_rows():
    cfg = CAGEConfig(
        use_cage=True,
        trace_only=True,
        debug=True,
        debug_light=True,
        disable_exact_state_ref_trace=True,
        trace_phi_vectors=False,
        max_debug_steps_per_episode=1,
    )
    ctrl = CAGEController(cfg, dist)
    path = np.asarray([[0.0], [1.0]])
    original = np.asarray([0.0])
    ctrl.reset_episode(np.asarray([0.0]), np.asarray([2.0]), path)

    _, _, _, _, trace = ctrl.select_subgoal(
        np.asarray([0.0]),
        np.asarray([2.0]),
        path,
        original,
        0,
        info={
            "original_subgoal": original,
            "original_subgoal_index": 0,
            "state_ref": {"exact_reset": True, "reset_mode": "exact_mujoco_state", "qpos": [1.0]},
        },
    )

    assert "trace_only_state_ref" not in trace
    assert "selected_subgoal_phi" not in trace
    assert "original_gas_subgoal_phi" not in trace
    ctrl.record_step_trace("env", 1, 0, 0, trace)
    ctrl.record_step_trace("env", 1, 0, 0, trace)
    assert len(ctrl._step_records) == 1
