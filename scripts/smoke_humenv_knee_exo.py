#!/usr/bin/env python
from __future__ import annotations

import numpy as np

from hit_exo_humenv import ExoKneeWalkingConfig, make_humenv_knee_exo_walking_env


def main() -> None:
    env = make_humenv_knee_exo_walking_env(
        ExoKneeWalkingConfig(max_episode_steps=5, render_mode=None)
    )
    obs, info = env.reset(seed=0)
    print("obs:", {k: v.shape for k, v in obs.items()})
    print("action_dim:", env.action_space.shape)
    print("knee_dof_ids:", env.knee_dof_ids.tolist())

    total_reward = 0.0
    for _ in range(5):
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    print("steps_ok total_reward:", round(total_reward, 6))
    print("last_exo_torque:", info["exo_torque"].tolist())
    env.close()


if __name__ == "__main__":
    main()
