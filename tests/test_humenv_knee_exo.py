from __future__ import annotations

import numpy as np

from hit_exo_humenv import ExoKneeWalkingConfig, make_humenv_knee_exo_walking_env


def test_knee_exo_env_smoke():
    env = make_humenv_knee_exo_walking_env(
        ExoKneeWalkingConfig(max_episode_steps=2, max_knee_torque=50.0, render_mode=None)
    )
    try:
        obs, _ = env.reset(seed=123)
        assert set(obs) == {"proprio", "exo"}
        assert env.knee_dof_ids.tolist() == [9, 21]
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        action[-2:] = [1.0, -1.0]
        _, _, _, _, info = env.step(action)
        np.testing.assert_allclose(info["exo_torque"], [50.0, -50.0])
    finally:
        env.close()


def test_exo_only_action_space():
    env = make_humenv_knee_exo_walking_env(
        ExoKneeWalkingConfig(action_mode="exo_only", render_mode=None)
    )
    try:
        assert env.action_space.shape == (2,)
    finally:
        env.close()
