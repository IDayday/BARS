import numpy as np
import pytest
import sys
from pathlib import Path

GAS_ROOT = Path(__file__).resolve().parents[1] / "external_src" / "GAS"
if str(GAS_ROOT) not in sys.path:
    sys.path.insert(0, str(GAS_ROOT))

from cage.state_ref import (
    EXACT_MUJOCO_STATE,
    OBSERVATION_ONLY_NOT_EXACT,
    StateRef,
    capture_state_ref,
    deserialize_state_ref,
    is_exact_state_ref,
    make_state_ref_from_env,
    restore_state_ref,
    restore_env_from_state_ref,
    serialize_state_ref,
    state_ref_is_exact,
)


class FakeData:
    def __init__(self):
        self.qpos = np.array([1.0, 2.0])
        self.qvel = np.array([3.0])


class FakeUnwrapped:
    def __init__(self):
        self.data = FakeData()
        self.restored = None

    def set_state(self, qpos, qvel):
        self.restored = (np.asarray(qpos), np.asarray(qvel))


class FakeEnv:
    def __init__(self):
        self.unwrapped = FakeUnwrapped()


def test_make_state_ref_from_env_is_exact_when_qpos_qvel_available():
    env = FakeEnv()
    ref = make_state_ref_from_env(env, obs=np.array([1.0, 2.0, 3.0]), metadata={"env_name": "fake"})
    assert ref.reset_mode == EXACT_MUJOCO_STATE
    assert state_ref_is_exact(ref)
    record = serialize_state_ref(ref)
    loaded = deserialize_state_ref(record)
    assert state_ref_is_exact(loaded)
    restore_env_from_state_ref(env, loaded)
    assert np.allclose(env.unwrapped.restored[0], [1.0, 2.0])
    assert np.allclose(env.unwrapped.restored[1], [3.0])


def test_obs_only_state_ref_is_not_exact_and_restore_raises():
    ref = StateRef(env_name="fake", obs=np.array([1.0]), reset_mode=OBSERVATION_ONLY_NOT_EXACT)
    assert not state_ref_is_exact(ref)
    with pytest.raises(RuntimeError, match="not exactly restorable"):
        restore_env_from_state_ref(FakeEnv(), ref)


def test_clp1_aliases_and_goal_fields_roundtrip():
    env = FakeEnv()
    ref = capture_state_ref(
        env,
        obs=np.array([1.0, 2.0, 3.0]),
        phi=np.array([0.1, 0.2]),
        metadata={"env_name": "fake", "goal_phi": np.array([0.3, 0.4]), "source_variant": "gas"},
    )
    assert is_exact_state_ref(ref)
    record = serialize_state_ref(ref)
    assert np.allclose(record["goal_phi"], [0.3, 0.4])
    assert record["source_variant"] == "gas"
    loaded = deserialize_state_ref(record)
    restore_state_ref(env, loaded)
    assert np.allclose(env.unwrapped.restored[0], [1.0, 2.0])
