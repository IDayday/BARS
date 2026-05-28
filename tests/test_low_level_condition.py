import numpy as np

from bars.conditioning import LowCondStats, LowLevelConditionBuilder, MazeXYFactorAdapter, ObjectFactorAdapter


class IdentityEncoder:
    def encode(self, obs):
        return np.asarray(obs, dtype=np.float32)


def test_full_condition_shape_and_values():
    stats = LowCondStats.identity(z_dim=4, factor_dim=2, factor_dim_max=4)
    builder = LowLevelConditionBuilder(IdentityEncoder(), stats, MazeXYFactorAdapter())
    cond = builder.encode(
        obs=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        local_target_obs=np.asarray([3.0, 4.0, 0.0, 0.0], dtype=np.float32),
        task_goal=np.asarray([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
        task_id=1,
    )
    assert cond.shape == (4 + 1 + 4 + 4,)
    np.testing.assert_allclose(cond[:4], np.asarray([0.6, 0.8, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert cond[4] > 0.0
    np.testing.assert_allclose(cond[5:9], np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(cond[9:], np.asarray([1.0, 2.0, 0.0, 0.0], dtype=np.float32))


def test_zero_distance_has_zero_direction_and_scale():
    stats = LowCondStats.identity(z_dim=2, factor_dim=2)
    builder = LowLevelConditionBuilder(IdentityEncoder(), stats, MazeXYFactorAdapter())
    obs = np.asarray([1.0, 2.0], dtype=np.float32)
    cond = builder.encode(obs=obs, local_target_obs=obs, task_goal=obs, task_id=1)
    np.testing.assert_allclose(cond[:2], np.zeros((2,), dtype=np.float32))
    assert cond[2] == 0.0


def test_mask_is_explicit_and_masks_residual():
    stats = LowCondStats.identity(z_dim=3, factor_dim=3)
    adapter = ObjectFactorAdapter(factor_indices=(0, 1, 2), task_masks={7: [1.0, 0.0, 1.0]})
    builder = LowLevelConditionBuilder(IdentityEncoder(), stats, adapter)
    cond = builder.encode(
        obs=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        local_target_obs=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        task_goal=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        task_id=7,
    )
    mask_start = 3 + 1
    np.testing.assert_allclose(cond[mask_start : mask_start + 3], np.asarray([1.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(cond[mask_start + 3 :], np.asarray([1.0, 0.0, 3.0], dtype=np.float32))


def test_final_task_goal_differs_from_local_subgoal():
    stats = LowCondStats.identity(z_dim=2, factor_dim=2)
    obs = np.asarray([0.0, 0.0], dtype=np.float32)
    local = np.asarray([1.0, 0.0], dtype=np.float32)
    final = np.asarray([0.0, 2.0], dtype=np.float32)
    task_builder = LowLevelConditionBuilder(IdentityEncoder(), stats, MazeXYFactorAdapter(), residual_target="task")
    local_builder = LowLevelConditionBuilder(IdentityEncoder(), stats, MazeXYFactorAdapter(), residual_target="local")
    task_cond = task_builder.encode(obs=obs, local_target_obs=local, task_goal=final, task_id=1)
    local_cond = local_builder.encode(obs=obs, local_target_obs=local, task_goal=final, task_id=1)
    np.testing.assert_allclose(task_cond[-2:], np.asarray([0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(local_cond[-2:], np.asarray([1.0, 0.0], dtype=np.float32))


def test_task_goal_fallback_is_logged():
    stats = LowCondStats.identity(z_dim=2, factor_dim=2)
    builder = LowLevelConditionBuilder(IdentityEncoder(), stats, MazeXYFactorAdapter())
    builder.encode(obs=np.asarray([0.0, 0.0]), local_target_obs=np.asarray([1.0, 1.0]), task_goal=None)
    assert builder.last_info["task_goal_fallback_to_local"] is True
