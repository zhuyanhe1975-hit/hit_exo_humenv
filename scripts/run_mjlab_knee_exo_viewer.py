from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

import hit_exo_humenv.mjlab  # noqa: F401
from hit_exo_humenv.latent_z_config import cfg_path
from hit_exo_humenv.mjlab.walking_env_cfg import (
    TASK_ID,
    TRAIN_WALKING_DIRECTION_CHOICES_DEG,
    TRAIN_WALKING_SPEED_CHOICES,
    exo_joint_group,
    exo_joint_names,
    exo_torque_limits,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer


DEFAULT_OUTPUT_ROOT = Path(cfg_path("paths", "viewer_power_log_root"))


class KneePdTorqueLoggingEnv:
    """RSL-RL env wrapper that logs actual knee PD torque after every step."""

    def __init__(self, env: RslRlVecEnvWrapper, log_file: Path, flush_every: int = 30) -> None:
        self.env = env
        self.log_file = log_file
        self.flush_every = flush_every
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_file.open("w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "step",
                "time_s",
                "reward_env0",
                "root_z_env0",
                "left_knee_pd_torque_nm",
                "right_knee_pd_torque_nm",
                "knee_pd_torque_l2_env0",
                "left_knee_exo_torque_nm",
                "right_knee_exo_torque_nm",
                "left_knee_joint_vel_rad_s",
                "right_knee_joint_vel_rad_s",
                "human_knee_abs_power_w",
                "human_knee_signed_power_w",
                "exo_knee_abs_power_w",
                "exo_knee_signed_power_w",
                "combined_knee_abs_power_w",
                "combined_knee_signed_power_w",
                "fallen_env0",
                "done_env0",
            ],
        )
        self._writer.writeheader()

        raw_env = self.env.unwrapped
        self._robot = raw_env.scene["robot"]
        knee_joint_ids, _ = self._robot.find_joints(("L_Knee_x", "R_Knee_x"), preserve_order=True)
        self._knee_joint_ids = torch.as_tensor(knee_joint_ids, dtype=torch.long, device=raw_env.device)
        self._knee_exo = raw_env.action_manager.get_term("knee_exo")
        self._step = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def device(self):
        return self.env.device

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def get_observations(self):
        return self.env.get_observations()

    def reset(self):
        return self.env.reset()

    def step(self, actions: torch.Tensor):
        result = self.env.step(actions)
        self._step += 1
        self._log_step(result)
        return result

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        self.env.close()

    def _log_step(self, result) -> None:
        _, reward, dones, _ = result
        raw_env = self.env.unwrapped
        pd_torque = self._robot.data.qfrc_actuator[:, self._knee_joint_ids]
        exo_torque = getattr(self._knee_exo, "_processed_actions", self._knee_exo.raw_action)
        knee_vel = self._robot.data.joint_vel[:, self._knee_joint_ids]
        human_power = pd_torque * knee_vel
        exo_power = exo_torque * knee_vel
        combined_power = (pd_torque + exo_torque) * knee_vel
        root_z = self._robot.data.root_link_pos_w[:, 2]
        fallen = raw_env.termination_manager.get_term("fallen")

        left_pd = float(pd_torque[0, 0].detach().cpu())
        right_pd = float(pd_torque[0, 1].detach().cpu())
        row = {
            "step": self._step,
            "time_s": self._step * raw_env.step_dt,
            "reward_env0": float(reward[0].detach().cpu()),
            "root_z_env0": float(root_z[0].detach().cpu()),
            "left_knee_pd_torque_nm": left_pd,
            "right_knee_pd_torque_nm": right_pd,
            "knee_pd_torque_l2_env0": left_pd * left_pd + right_pd * right_pd,
            "left_knee_exo_torque_nm": float(exo_torque[0, 0].detach().cpu()),
            "right_knee_exo_torque_nm": float(exo_torque[0, 1].detach().cpu()),
            "left_knee_joint_vel_rad_s": float(knee_vel[0, 0].detach().cpu()),
            "right_knee_joint_vel_rad_s": float(knee_vel[0, 1].detach().cpu()),
            "human_knee_abs_power_w": float(torch.sum(torch.abs(human_power[0])).detach().cpu()),
            "human_knee_signed_power_w": float(torch.sum(human_power[0]).detach().cpu()),
            "exo_knee_abs_power_w": float(torch.sum(torch.abs(exo_power[0])).detach().cpu()),
            "exo_knee_signed_power_w": float(torch.sum(exo_power[0]).detach().cpu()),
            "combined_knee_abs_power_w": float(torch.sum(torch.abs(combined_power[0])).detach().cpu()),
            "combined_knee_signed_power_w": float(torch.sum(combined_power[0]).detach().cpu()),
            "fallen_env0": int(bool(fallen[0].detach().cpu())),
            "done_env0": int(bool(dones[0].detach().cpu())),
        }
        self._writer.writerow(row)
        if self._step % self.flush_every == 0:
            self._file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run knee-exo policy in native viewer and log knee PD torque.")
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=cfg_path("viewer", "num_envs"))
    parser.add_argument("--env-spacing", type=float, default=cfg_path("simulation", "env_spacing"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--walk-speed", type=float, default=cfg_path("walking_command", "normal_speed"))
    parser.add_argument("--random-walk-speed", action=argparse.BooleanOptionalAction, default=cfg_path("eval", "random_walk_speed"))
    parser.add_argument("--walk-speed-choices", type=float, nargs="+", default=list(TRAIN_WALKING_SPEED_CHOICES))
    parser.add_argument("--walk-direction", type=float, default=cfg_path("walking_command", "direction_choices_deg")[0])
    parser.add_argument("--random-walk-direction", action=argparse.BooleanOptionalAction, default=cfg_path("eval", "random_walk_direction"))
    parser.add_argument("--walk-direction-choices", type=float, nargs="+", default=list(TRAIN_WALKING_DIRECTION_CHOICES_DEG))
    parser.add_argument("--speed-resampling-time-range", type=float, nargs=2, default=cfg_path("walking_command", "resampling_time_range"))
    parser.add_argument("--s1-latent-speed-scale", type=float, default=cfg_path("human_s1", "latent_speed_scale"))
    parser.add_argument("--human-action-repeat", type=int, default=cfg_path("human_s1", "action_repeat"))
    parser.add_argument("--human-action-smoothing", type=float, default=cfg_path("human_s1", "action_smoothing"))
    parser.add_argument("--human-root-height", type=float, default=cfg_path("human_s1", "root_height"))
    parser.add_argument("--exo-joint-group", choices=("knee", "hip", "ankle", "hip_knee", "knee_ankle", "lower_limb"), default=exo_joint_group())
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _configure_walk_command(env_cfg, args: argparse.Namespace) -> None:
    command_cfg = env_cfg.commands["walk_speed"]
    human_cfg = env_cfg.actions["human_s1"]
    if args.random_walk_speed:
        choices = tuple(args.walk_speed_choices)
        command_cfg.speed_range = (min(choices), max(choices))
        command_cfg.speed_choices = choices
        human_cfg.speed_bins = choices
    else:
        command_cfg.speed_range = (args.walk_speed, args.walk_speed)
        command_cfg.speed_choices = (args.walk_speed,)
        human_cfg.speed_bins = (args.walk_speed,)

    if args.random_walk_direction:
        choices = tuple(args.walk_direction_choices)
        command_cfg.direction_range_deg = (min(choices), max(choices))
        command_cfg.direction_choices_deg = choices
        human_cfg.direction_bins_deg = choices
    else:
        command_cfg.direction_range_deg = (args.walk_direction, args.walk_direction)
        command_cfg.direction_choices_deg = (args.walk_direction,)
        human_cfg.direction_bins_deg = (args.walk_direction,)
    command_cfg.include_direction = True
    command_cfg.resampling_time_range = tuple(args.speed_resampling_time_range)
    command_cfg.reset_on_resample = False


def _configure_human_gait(env_cfg, args: argparse.Namespace) -> None:
    human_cfg = env_cfg.actions["human_s1"]
    human_cfg.latent_speed_scale = args.s1_latent_speed_scale
    human_cfg.action_repeat = max(1, args.human_action_repeat)
    human_cfg.action_smoothing = min(max(args.human_action_smoothing, 0.0), 0.99)
    robot_cfg = env_cfg.scene.entities["robot"]
    robot_cfg.init_state.pos = (0.0, 0.0, args.human_root_height)


def _set_grid_env_origins(raw_env: ManagerBasedRlEnv, spacing: float) -> None:
    num_envs = raw_env.num_envs
    num_rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
    num_cols = math.ceil(num_envs / num_rows)
    ii, jj = torch.meshgrid(
        torch.arange(num_rows, device=raw_env.device),
        torch.arange(num_cols, device=raw_env.device),
        indexing="ij",
    )
    origins = torch.zeros(num_envs, 3, device=raw_env.device)
    origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * spacing
    origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * spacing
    raw_env.scene._default_env_origins = origins


def main() -> None:
    args = parse_args()
    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_file}")

    env_cfg = load_env_cfg(TASK_ID, play=False)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = args.env_spacing
    env_cfg.actions["knee_exo"].joint_names = exo_joint_names(args.exo_joint_group)
    env_cfg.actions["knee_exo"].max_torque = exo_torque_limits(args.exo_joint_group)
    _configure_walk_command(env_cfg, args)
    _configure_human_gait(env_cfg, args)
    agent_cfg = load_rl_cfg(TASK_ID)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _set_grid_env_origins(raw_env, args.env_spacing)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    log_file = args.log_file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = args.output_root / f"{timestamp}_{args.checkpoint_file.parent.name}_{args.checkpoint_file.stem}.csv"

    logging_env = KneePdTorqueLoggingEnv(env, log_file)
    print(f"[INFO] Logging knee PD torque to: {log_file}")
    try:
        NativeMujocoViewer(logging_env, policy).run()
    finally:
        logging_env.close()
        print(f"[INFO] Wrote knee PD torque log: {log_file}")


if __name__ == "__main__":
    main()
