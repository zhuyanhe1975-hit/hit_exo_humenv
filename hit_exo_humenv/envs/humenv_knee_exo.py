from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from humenv.env import HumEnv


@dataclass(frozen=True)
class ExoKneeWalkingConfig:
    """Configuration for the abstract knee-assist walking task."""

    target_speed: float = 2.0
    target_angle_deg: float = 0.0
    max_knee_torque: float = 80.0
    action_mode: str = "exo_only"
    base_controller: str = "zero"
    max_episode_steps: int = 300
    exo_torque_cost: float = 1.0e-4
    exo_torque_rate_cost: float = 5.0e-5
    terminate_on_fall: bool = True
    min_head_height: float = 0.9
    render_mode: str | None = None


class HumEnvKneeExoWalkingEnv(gym.Env):
    """HumEnv walking task with abstract bilateral knee exoskeleton torque.

    The environment keeps the original HumEnv humanoid and applies two additional
    generalized forces to the primary knee flexion-extension DoFs:
    `L_Knee_x` and `R_Knee_x`. No exoskeleton rigid bodies or contacts are added.

    Action layout:
      - `human_plus_exo`: original HumEnv controls followed by two normalized exo
        torques in `[-1, 1]`.
      - `exo_only`: two normalized exo torques; human controls are held at zero.
    """

    metadata = HumEnv.metadata

    def __init__(self, cfg: ExoKneeWalkingConfig | None = None, **humenv_kwargs: Any) -> None:
        self.cfg = cfg or ExoKneeWalkingConfig()
        if self.cfg.action_mode not in {"human_plus_exo", "exo_only"}:
            raise ValueError(f"Unsupported action_mode: {self.cfg.action_mode}")
        if self.cfg.base_controller not in {"zero", "metamotivo_s1"}:
            raise ValueError(f"Unsupported base_controller: {self.cfg.base_controller}")

        humenv_kwargs.setdefault("task", f"move-ego-{self.cfg.target_angle_deg:g}-{self.cfg.target_speed:g}")
        humenv_kwargs.setdefault("state_init", "Default")
        humenv_kwargs.setdefault("render_mode", self.cfg.render_mode)
        self.base_env = HumEnv(**humenv_kwargs)
        self._base_policy = self._build_base_policy()

        self.knee_joint_names = ("L_Knee_x", "R_Knee_x")
        self.knee_dof_ids = np.asarray(
            [self._joint_dof_id(name) for name in self.knee_joint_names],
            dtype=np.int32,
        )

        self._elapsed_steps = 0
        self._last_exo_torque = np.zeros(2, dtype=np.float64)
        self._last_full_action = np.zeros(self.base_env.model.nu + 2, dtype=np.float64)

        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    @property
    def model(self):
        return self.base_env.model

    @property
    def data(self):
        return self.base_env.data

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._elapsed_steps = 0
        self._last_exo_torque[:] = 0.0
        self._last_full_action[:] = 0.0
        obs, info = self.base_env.reset(seed=seed, options=options)
        return self._augment_obs(obs), info

    def step(self, action: np.ndarray):
        human_action, exo_torque = self._split_action(action)
        data = self.base_env.data

        data.ctrl[:] = human_action
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[self.knee_dof_ids] = exo_torque
        mujoco.mj_step(self.base_env.model, data, nstep=self.base_env.action_repeat)
        if data.warning.number.any():
            warning_index = np.nonzero(data.warning.number)[0][0]
            warning = mujoco.mjtWarning(warning_index).name
            raise ValueError(f"UNSTABLE MUJOCO. Stopped due to divergence ({warning}).")
        mujoco.mj_step1(self.base_env.model, data)
        data.qfrc_applied[:] = 0.0

        base_obs = self.base_env.get_obs()
        task_reward = float(self.base_env.task.compute(self.base_env.model, data))
        exo_cost = self.cfg.exo_torque_cost * float(np.square(exo_torque).sum())
        exo_rate_cost = self.cfg.exo_torque_rate_cost * float(
            np.square(exo_torque - self._last_exo_torque).sum()
        )
        reward = task_reward - exo_cost - exo_rate_cost

        self._elapsed_steps += 1
        terminated = self._is_fallen() if self.cfg.terminate_on_fall else False
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps

        self._last_exo_torque[:] = exo_torque
        self._last_full_action[:] = np.concatenate([human_action, exo_torque])

        info = self.base_env.get_info()
        info.update(
            {
                "task_reward": task_reward,
                "exo_torque": exo_torque.copy(),
                "exo_torque_cost": exo_cost,
                "exo_torque_rate_cost": exo_rate_cost,
                "knee_dof_ids": self.knee_dof_ids.copy(),
            }
        )
        return self._augment_obs(base_obs), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()

    def _joint_dof_id(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.base_env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Missing joint in HumEnv model: {joint_name}")
        return int(self.base_env.model.jnt_dofadr[joint_id])

    def _split_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        if self.cfg.action_mode == "exo_only":
            if action.shape != (2,):
                raise ValueError(f"Expected exo-only action shape (2,), got {action.shape}")
            human_action = self._base_human_action()
            exo_norm = np.clip(action, -1.0, 1.0)
        else:
            expected = self.base_env.model.nu + 2
            if action.shape != (expected,):
                raise ValueError(f"Expected action shape ({expected},), got {action.shape}")
            human_action = np.clip(
                action[: self.base_env.model.nu],
                self.base_env.action_space.low,
                self.base_env.action_space.high,
            )
            exo_norm = np.clip(action[-2:], -1.0, 1.0)
        return human_action, exo_norm * self.cfg.max_knee_torque

    def _build_base_policy(self):
        if self.cfg.base_controller == "zero":
            return None
        from hit_exo_humenv.s1_policy import MetamotivoS1Policy

        return MetamotivoS1Policy(
            task=f"move-ego-{self.cfg.target_angle_deg:g}-{self.cfg.target_speed:g}",
        )

    def _base_human_action(self) -> np.ndarray:
        if self._base_policy is None:
            return np.zeros(self.base_env.model.nu, dtype=np.float64)
        return np.clip(
            self._base_policy(self.base_env.get_obs()["proprio"]),
            self.base_env.action_space.low,
            self.base_env.action_space.high,
        )

    def _augment_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        exo = np.asarray(
            [
                self._last_exo_torque[0] / self.cfg.max_knee_torque,
                self._last_exo_torque[1] / self.cfg.max_knee_torque,
                self.cfg.target_speed,
                self.cfg.target_angle_deg / 180.0,
            ],
            dtype=np.float64,
        )
        return {
            "proprio": obs["proprio"].astype(np.float64, copy=False),
            "exo": exo,
        }

    def _build_observation_space(self) -> gym.spaces.Dict:
        base_obs = self.base_env.get_obs()
        return gym.spaces.Dict(
            {
                "proprio": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=base_obs["proprio"].shape,
                    dtype=np.float64,
                ),
                "exo": gym.spaces.Box(
                    low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float64),
                    high=np.asarray([1.0, 1.0, np.inf, 1.0], dtype=np.float64),
                    dtype=np.float64,
                ),
            }
        )

    def _build_action_space(self) -> gym.spaces.Box:
        if self.cfg.action_mode == "exo_only":
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        low = np.concatenate([self.base_env.action_space.low, -np.ones(2, dtype=np.float64)])
        high = np.concatenate([self.base_env.action_space.high, np.ones(2, dtype=np.float64)])
        return gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def _is_fallen(self) -> bool:
        head_id = mujoco.mj_name2id(self.base_env.model, mujoco.mjtObj.mjOBJ_BODY, "Head")
        if head_id < 0:
            return False
        return bool(self.base_env.data.xpos[head_id, 2] < self.cfg.min_head_height)


def make_humenv_knee_exo_walking_env(
    cfg: ExoKneeWalkingConfig | None = None,
    **humenv_kwargs: Any,
) -> HumEnvKneeExoWalkingEnv:
    return HumEnvKneeExoWalkingEnv(cfg=cfg, **humenv_kwargs)
