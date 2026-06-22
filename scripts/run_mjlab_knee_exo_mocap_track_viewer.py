from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

import hit_exo_humenv.mjlab  # noqa: F401
from hit_exo_humenv.latent_z_config import cfg_path
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

from run_mjlab_knee_exo_viewer import (
    DEFAULT_OUTPUT_ROOT,
    KneePdTorqueLoggingEnv,
    _set_grid_env_origins,
)
from train_mjlab_knee_exo_mocap_track import DEFAULT_MOTION, TASK_ID, build_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mocap-track knee-exo checkpoint with frozen S-1 in the native viewer."
    )
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument(
        "--training-stage",
        choices=("knee-exo", "stair-compensation", "knee-exo-on-compensation"),
        default="knee-exo",
    )
    parser.add_argument("--base-compensation-checkpoint", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=cfg_path("viewer", "num_envs"))
    parser.add_argument("--env-spacing", type=float, default=cfg_path("simulation", "env_spacing"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--s1-samples", type=int, default=cfg_path("human_s1", "num_samples_per_inference"))
    parser.add_argument("--s1-workers", type=int, default=cfg_path("human_s1", "max_workers"))
    parser.add_argument("--s1-action-smoothing", type=float, default=0.15)
    parser.add_argument("--physics-timestep", type=float, default=cfg_path("simulation", "mujoco_timestep"))
    parser.add_argument("--decimation", type=int, default=cfg_path("simulation", "decimation"))
    parser.add_argument("--terrain", choices=("flat", "stairs", "supports", "mimic-stairs"), default="flat")
    parser.add_argument("--terrain-out-dir", type=Path, default=Path(".omx/mjlab_mocap_track_viewer_terrain"))
    parser.add_argument("--stair-width", type=float, default=1.4)
    parser.add_argument("--stair-height", type=float, default=0.135)
    parser.add_argument("--stair-steps", type=int, default=0)
    parser.add_argument("--max-knee-torque", type=float, default=cfg_path("exo", "max_knee_torque"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _train_config_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        motion=args.motion,
        episode=args.episode,
        training_stage=args.training_stage,
        base_compensation_checkpoint=args.base_compensation_checkpoint,
        num_envs=args.num_envs,
        max_iterations=1,
        num_steps_per_env=cfg_path("train", "num_steps_per_env"),
        save_interval=cfg_path("train", "save_interval"),
        seed=args.seed,
        gpu_id=0,
        cpu=False,
        logger="tensorboard",
        run_name="mocap_track_viewer",
        log_root="logs/rsl_rl",
        actor_init_std=None,
        entropy_coef=None,
        s1_samples=args.s1_samples,
        s1_workers=args.s1_workers,
        s1_action_smoothing=args.s1_action_smoothing,
        human_residual_mode="none",
        human_residual_scale=0.0,
        human_residual_action_weight=0.0,
        human_residual_action_rate_weight=0.0,
        mocap_assist_replacement_weight=0.0,
        mocap_assist_start=0.0,
        mocap_assist_end=0.0,
        mocap_assist_decay_steps=1,
        mocap_assist_decay_fraction=0.0,
        mocap_assist_position_gain=1.0,
        mocap_assist_velocity_gain=0.05,
        mocap_assist_max_action=0.5,
        episode_length_s=1.0e9,
        physics_timestep=args.physics_timestep,
        decimation=args.decimation,
        terrain=args.terrain,
        terrain_out_dir=args.terrain_out_dir,
        stair_width=args.stair_width,
        stair_height=args.stair_height,
        stair_steps=args.stair_steps,
        max_knee_torque=args.max_knee_torque,
        progress_reward_weight=cfg_path("reward", "not_fallen_progress_weight"),
        mocap_root_xyz_weight=0.0,
        mocap_root_xyz_std=0.25,
        mocap_root_orientation_weight=0.0,
        mocap_root_orientation_std=0.35,
        mocap_lower_body_weight=0.0,
        mocap_lower_body_std=0.45,
        mocap_knee_weight=0.0,
        mocap_knee_std=0.25,
        mocap_foot_weight=0.0,
        mocap_foot_std=0.25,
        mocap_foot_sides="L,R",
        mocap_foot_z_weight=1.0,
        mocap_first_foot_weight=0.0,
        mocap_first_foot_std=0.18,
        mocap_first_foot_sides="L,R",
        mocap_first_foot_end_frame=90,
        mocap_foot_event_weight=0.0,
        mocap_foot_event_std=0.18,
        mocap_foot_event_xy_weight=1.0,
        mocap_foot_event_z_weight=1.0,
        mocap_foot_event_speed_threshold=0.18,
        mocap_foot_event_min_stance_frames=4,
        mocap_foot_event_min_height_delta=0.05,
        mocap_foot_event_window_margin=0,
        lower_limb_power_weight=cfg_path("reward", "lower_limb_joint_power_weight"),
        hip_power_weight=cfg_path("reward", "power_joint_weights")["hip"],
        knee_power_weight=cfg_path("reward", "power_joint_weights")["knee"],
        ankle_power_weight=cfg_path("reward", "power_joint_weights")["ankle"],
        positive_work_efficiency=cfg_path("metabolic", "positive_work_efficiency"),
        negative_work_efficiency=cfg_path("metabolic", "negative_work_efficiency"),
        joint_vel_smoothness_weight=cfg_path("reward", "lower_limb_joint_velocity_delta_weight"),
        hip_vel_smoothness_weight=cfg_path("reward", "smoothness_joint_weights")["hip"],
        knee_vel_smoothness_weight=cfg_path("reward", "smoothness_joint_weights")["knee"],
        ankle_vel_smoothness_weight=cfg_path("reward", "smoothness_joint_weights")["ankle"],
        fall_penalty_weight=cfg_path("reward", "fallen_penalty_weight"),
        fall_min_height=cfg_path("reward", "fall_min_height"),
    )


def main() -> None:
    args = parse_args()
    configure_torch_backends()

    if not args.checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_file}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    train_cfg = build_train_config(_train_config_args(args))
    env_cfg = train_cfg.env
    env_cfg.scene.env_spacing = args.env_spacing
    agent_cfg = train_cfg.agent

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
        log_file = (
            args.output_root
            / f"{timestamp}_{args.checkpoint_file.parent.name}_{args.checkpoint_file.stem}_mocap.csv"
        )

    logging_env = KneePdTorqueLoggingEnv(env, log_file)
    print(f"[INFO] Running fixed mocap reference: {args.motion}:{args.episode}")
    print(f"[INFO] Logging knee PD torque to: {log_file}")
    try:
        NativeMujocoViewer(logging_env, policy).run()
    finally:
        logging_env.close()
        print(f"[INFO] Wrote knee PD torque log: {log_file}")


if __name__ == "__main__":
    main()
