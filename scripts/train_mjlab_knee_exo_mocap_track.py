#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import h5py
import mujoco
import numpy as np

import hit_exo_humenv.mjlab  # noqa: F401
import hit_exo_humenv.mjlab.mdp as mdp
from hit_exo_humenv.latent_z_config import cfg_path
from hit_exo_humenv.mjlab.walking_env_cfg import TASK_ID
from hit_exo_humenv.mjlab.walking_env_cfg import HUMENV_XML
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.train import TrainConfig, launch_training


DEFAULT_MOTION = Path(
    "/home/yhzhu/AI/humenv/data_preparation/humenv_from_phc_amass_transitions_kit_cmu/"
    "08_KIT_167_walking_medium_resampled_poses.hdf5"
)
ROOT_XYZ_QPOS_COLUMNS = (0, 1, 2)
ROOT_ORIENTATION_QPOS_COLUMNS = (3, 4, 5, 6)
LOWER_BODY_QPOS_COLUMNS = tuple(range(7, 30))
PRIMARY_KNEE_QPOS_COLUMNS = (10, 22)
LOWER_BODY_RESIDUAL_JOINTS = (
    "L_Hip_.*",
    "R_Hip_.*",
    "L_Knee_.*",
    "R_Knee_.*",
    "L_Ankle_.*",
    "R_Ankle_.*",
    "L_Toe_.*",
    "R_Toe_.*",
)
ALL_BODY_RESIDUAL_JOINTS = (".*",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train mocap stair tracking with frozen S-1, an optional full-body "
            "compensation policy, and/or a knee-exo policy."
        )
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument(
        "--training-stage",
        choices=("knee-exo", "stair-compensation", "knee-exo-on-compensation"),
        default="knee-exo",
        help=(
            "knee-exo keeps the old 2D exo-only action space; stair-compensation "
            "trains S-1 residual torques; knee-exo-on-compensation freezes a "
            "trained residual checkpoint and trains only the knee exo."
        ),
    )
    parser.add_argument("--base-compensation-checkpoint", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=cfg_path("train", "num_envs"))
    parser.add_argument("--max-iterations", type=int, default=cfg_path("train", "max_iterations"))
    parser.add_argument("--num-steps-per-env", type=int, default=cfg_path("train", "num_steps_per_env"))
    parser.add_argument("--save-interval", type=int, default=cfg_path("train", "save_interval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "wandb", "neptune"),
        default=cfg_path("train", "logger"),
    )
    parser.add_argument("--run-name", default="mocap_track")
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--actor-init-std", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--s1-samples", type=int, default=cfg_path("human_s1", "num_samples_per_inference"))
    parser.add_argument("--s1-workers", type=int, default=cfg_path("human_s1", "max_workers"))
    parser.add_argument("--s1-action-smoothing", type=float, default=0.15)
    parser.add_argument("--human-residual-mode", choices=("none", "lower-body", "all"), default="none")
    parser.add_argument("--human-residual-scale", type=float, default=0.0)
    parser.add_argument("--human-residual-action-weight", type=float, default=0.0)
    parser.add_argument("--human-residual-action-rate-weight", type=float, default=0.0)
    parser.add_argument("--mocap-assist-replacement-weight", type=float, default=0.0)
    parser.add_argument("--mocap-assist-start", type=float, default=0.0)
    parser.add_argument("--mocap-assist-end", type=float, default=0.0)
    parser.add_argument("--mocap-assist-decay-steps", type=int, default=0)
    parser.add_argument("--mocap-assist-decay-fraction", type=float, default=0.0)
    parser.add_argument("--mocap-assist-position-gain", type=float, default=1.0)
    parser.add_argument("--mocap-assist-velocity-gain", type=float, default=0.05)
    parser.add_argument("--mocap-assist-max-action", type=float, default=0.5)
    parser.add_argument("--episode-length-s", type=float, default=None)
    parser.add_argument("--physics-timestep", type=float, default=cfg_path("simulation", "mujoco_timestep"))
    parser.add_argument("--decimation", type=int, default=cfg_path("simulation", "decimation"))
    parser.add_argument("--terrain", choices=("flat", "stairs", "supports", "mimic-stairs"), default="flat")
    parser.add_argument("--terrain-out-dir", type=Path, default=Path(".omx/mjlab_mocap_track_terrain"))
    parser.add_argument("--stair-width", type=float, default=1.4)
    parser.add_argument("--stair-height", type=float, default=0.135)
    parser.add_argument("--stair-steps", type=int, default=0)
    parser.add_argument("--max-knee-torque", type=float, default=cfg_path("exo", "max_knee_torque"))
    parser.add_argument("--progress-reward-weight", type=float, default=cfg_path("reward", "not_fallen_progress_weight"))
    parser.add_argument("--mocap-root-xyz-weight", type=float, default=0.0)
    parser.add_argument("--mocap-root-xyz-std", type=float, default=0.25)
    parser.add_argument("--mocap-root-orientation-weight", type=float, default=0.0)
    parser.add_argument("--mocap-root-orientation-std", type=float, default=0.35)
    parser.add_argument("--mocap-lower-body-weight", type=float, default=0.0)
    parser.add_argument("--mocap-lower-body-std", type=float, default=0.45)
    parser.add_argument("--mocap-knee-weight", type=float, default=0.0)
    parser.add_argument("--mocap-knee-std", type=float, default=0.25)
    parser.add_argument("--mocap-foot-weight", type=float, default=0.0)
    parser.add_argument("--mocap-foot-std", type=float, default=0.25)
    parser.add_argument("--mocap-foot-sides", default="L,R")
    parser.add_argument("--mocap-foot-z-weight", type=float, default=1.0)
    parser.add_argument("--mocap-first-foot-weight", type=float, default=0.0)
    parser.add_argument("--mocap-first-foot-std", type=float, default=0.18)
    parser.add_argument("--mocap-first-foot-sides", default="L,R")
    parser.add_argument("--mocap-first-foot-end-frame", type=int, default=90)
    parser.add_argument("--mocap-foot-event-weight", type=float, default=0.0)
    parser.add_argument("--mocap-foot-event-std", type=float, default=0.18)
    parser.add_argument("--mocap-foot-event-xy-weight", type=float, default=1.0)
    parser.add_argument("--mocap-foot-event-z-weight", type=float, default=1.0)
    parser.add_argument("--mocap-foot-event-speed-threshold", type=float, default=0.18)
    parser.add_argument("--mocap-foot-event-min-stance-frames", type=int, default=4)
    parser.add_argument("--mocap-foot-event-min-height-delta", type=float, default=0.05)
    parser.add_argument("--mocap-foot-event-window-margin", type=int, default=0)
    parser.add_argument("--lower-limb-power-weight", type=float, default=cfg_path("reward", "lower_limb_joint_power_weight"))
    parser.add_argument("--hip-power-weight", type=float, default=cfg_path("reward", "power_joint_weights")["hip"])
    parser.add_argument("--knee-power-weight", type=float, default=cfg_path("reward", "power_joint_weights")["knee"])
    parser.add_argument("--ankle-power-weight", type=float, default=cfg_path("reward", "power_joint_weights")["ankle"])
    parser.add_argument("--positive-work-efficiency", type=float, default=cfg_path("metabolic", "positive_work_efficiency"))
    parser.add_argument("--negative-work-efficiency", type=float, default=cfg_path("metabolic", "negative_work_efficiency"))
    parser.add_argument("--joint-vel-smoothness-weight", type=float, default=cfg_path("reward", "lower_limb_joint_velocity_delta_weight"))
    parser.add_argument("--hip-vel-smoothness-weight", type=float, default=cfg_path("reward", "smoothness_joint_weights")["hip"])
    parser.add_argument("--knee-vel-smoothness-weight", type=float, default=cfg_path("reward", "smoothness_joint_weights")["knee"])
    parser.add_argument("--ankle-vel-smoothness-weight", type=float, default=cfg_path("reward", "smoothness_joint_weights")["ankle"])
    parser.add_argument("--fall-penalty-weight", type=float, default=cfg_path("reward", "fallen_penalty_weight"))
    parser.add_argument("--fall-min-height", type=float, default=cfg_path("reward", "fall_min_height"))
    return parser.parse_args()


def load_motion(path: Path, episode: str) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float32)
        qvel = np.asarray(ep["qvel"][:], dtype=np.float32)
        dt = float(ep.attrs.get("dt", hf.attrs.get("dt", 1.0 / 30.0)))
        if "observation" not in ep:
            raise KeyError(f"{path}:{episode} is missing required 'observation' dataset")
    return qpos, qvel, dt


def motion_duration(path: Path, episode: str) -> float:
    qpos, _qvel, dt = load_motion(path, episode)
    return max(dt * (qpos.shape[0] - 1), dt)


def _motion_initial_state(path: Path, episode: str) -> tuple[np.ndarray, np.ndarray]:
    qpos, qvel, _dt = load_motion(path, episode)
    return qpos[0], qvel[0]


def _joint_state_from_motion_initial_frame(
    qpos0: np.ndarray,
    qvel0: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    model = mujoco.MjModel.from_xml_path(str(HUMENV_XML))
    joint_pos = {}
    joint_vel = {}
    for joint_id in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        joint_pos[name] = float(qpos0[qadr])
        joint_vel[name] = float(qvel0[dadr])
    return joint_pos, joint_vel


def _spec_fn_from_xml(xml_path: Path):
    xml_path = Path(xml_path)

    def _spec_fn() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(xml_path))

    return _spec_fn


def _build_training_terrain_xml(
    args: argparse.Namespace,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> Path | None:
    if args.terrain == "flat":
        return None

    from run_s1_mocap_track_visual import build_mimic_matched_stair_xml, build_stair_xml, build_support_xml

    motion = {"qpos": qpos, "qvel": qvel}
    if args.terrain == "stairs":
        return build_stair_xml(
            motion,
            args.terrain_out_dir,
            stair_width=args.stair_width,
            nominal_step_height=args.stair_height,
            requested_steps=args.stair_steps,
        )
    if args.terrain == "supports":
        return build_support_xml(args.motion, args.terrain_out_dir)
    if args.terrain == "mimic-stairs":
        return build_mimic_matched_stair_xml(
            motion,
            args.terrain_out_dir,
            stair_width=args.stair_width,
        )
    raise ValueError(f"Unsupported terrain: {args.terrain}")


def _add_mocap_qpos_reward(
    rewards: dict,
    *,
    name: str,
    weight: float,
    motion_file: Path,
    episode: str,
    columns: tuple[int, ...],
    std: float,
    root_xy_invariant: bool,
) -> None:
    if weight == 0.0:
        return
    if std <= 0.0:
        raise ValueError(f"{name} std must be positive")
    rewards[name] = RewardTermCfg(
        func=mdp.mocap_qpos_tracking_reward,
        weight=weight,
        params={
            "motion_file": str(motion_file),
            "episode": episode,
            "columns": columns,
            "std": std,
            "root_xy_invariant": root_xy_invariant,
        },
    )


def _add_mocap_foot_reward(
    rewards: dict,
    *,
    name: str,
    weight: float,
    motion_file: Path,
    episode: str,
    std: float,
    sides: tuple[str, ...],
    z_weight: float,
    start_frame: int = 0,
    end_frame: int = -1,
) -> None:
    if weight == 0.0:
        return
    if std <= 0.0:
        raise ValueError(f"{name} std must be positive")
    rewards[name] = RewardTermCfg(
        func=mdp.mocap_foot_pos_tracking_reward,
        weight=weight,
        params={
            "motion_file": str(motion_file),
            "episode": episode,
            "robot_xml": str(HUMENV_XML),
            "sides": sides,
            "std": std,
            "z_weight": z_weight,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    )


def _add_mocap_foot_event_reward(
    rewards: dict,
    *,
    name: str,
    weight: float,
    motion_file: Path,
    episode: str,
    std: float,
    xy_weight: float,
    z_weight: float,
    stance_speed_threshold: float,
    min_stance_frames: int,
    min_event_height_delta: float,
    window_margin_frames: int,
) -> None:
    if weight == 0.0:
        return
    if std <= 0.0:
        raise ValueError(f"{name} std must be positive")
    rewards[name] = RewardTermCfg(
        func=mdp.mocap_foot_event_tracking_reward,
        weight=weight,
        params={
            "motion_file": str(motion_file),
            "episode": episode,
            "robot_xml": str(HUMENV_XML),
            "std": std,
            "xy_weight": xy_weight,
            "z_weight": z_weight,
            "stance_speed_threshold": stance_speed_threshold,
            "min_stance_frames": min_stance_frames,
            "min_event_height_delta": min_event_height_delta,
            "window_margin_frames": window_margin_frames,
        },
    )


def _parse_foot_sides(value: str) -> tuple[str, ...]:
    sides = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not sides or any(side not in {"L", "R"} for side in sides):
        raise ValueError(f"Invalid foot side list: {value!r}; expected L, R, or L,R")
    return sides


def _normalize_stage_args(args: argparse.Namespace) -> None:
    if args.mocap_assist_decay_steps <= 0 and args.mocap_assist_decay_fraction > 0.0:
        args.mocap_assist_decay_steps = max(
            1,
            int(args.max_iterations * args.num_steps_per_env * args.mocap_assist_decay_fraction),
        )

    if args.training_stage == "stair-compensation":
        if args.human_residual_mode == "none":
            args.human_residual_mode = "all"
        if args.human_residual_scale <= 0.0:
            args.human_residual_scale = 0.35
        if args.human_residual_action_weight == 0.0:
            args.human_residual_action_weight = -0.001
        if args.human_residual_action_rate_weight == 0.0:
            args.human_residual_action_rate_weight = -0.004
        return

    if args.training_stage == "knee-exo-on-compensation":
        if args.base_compensation_checkpoint is None:
            raise ValueError("--base-compensation-checkpoint is required for knee-exo-on-compensation")
        if not args.base_compensation_checkpoint.exists():
            raise FileNotFoundError(f"Base compensation checkpoint not found: {args.base_compensation_checkpoint}")
        if args.human_residual_mode == "none":
            args.human_residual_mode = "all"
        if args.human_residual_scale <= 0.0:
            args.human_residual_scale = 0.35
        args.human_residual_action_weight = 0.0
        args.human_residual_action_rate_weight = 0.0
        args.mocap_assist_start = 0.0
        args.mocap_assist_end = 0.0


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig.from_task(TASK_ID)
    env_cfg = cfg.env
    agent_cfg = cfg.agent
    _normalize_stage_args(args)

    qpos, qvel, dt = load_motion(args.motion, args.episode)
    mocap_duration = max(dt * (qpos.shape[0] - 1), dt)
    qpos0, qvel0 = qpos[0], qvel[0]
    joint_pos, joint_vel = _joint_state_from_motion_initial_frame(qpos0, qvel0)
    terrain_xml = _build_training_terrain_xml(args, qpos, qvel)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.mujoco.timestep = args.physics_timestep
    env_cfg.decimation = args.decimation
    env_cfg.episode_length_s = args.episode_length_s or mocap_duration
    env_cfg.commands = {}
    for group_cfg in env_cfg.observations.values():
        group_cfg.terms.pop("walk_speed", None)
    robot_init = env_cfg.scene.entities["robot"].init_state
    robot_init.pos = tuple(float(v) for v in qpos0[:3])
    robot_init.rot = tuple(float(v) for v in qpos0[3:7])
    robot_init.joint_pos = joint_pos
    robot_init.joint_vel = joint_vel
    if terrain_xml is not None:
        env_cfg.scene.entities["robot"].spec_fn = _spec_fn_from_xml(terrain_xml)

    human_cfg = env_cfg.actions["human_s1"]
    human_cfg.task = "tracking"
    human_cfg.speed_command_name = None
    human_cfg.num_samples_per_inference = args.s1_samples
    human_cfg.max_workers = args.s1_workers
    human_cfg.action_smoothing = args.s1_action_smoothing
    human_cfg.tracking_motion_file = str(args.motion)
    human_cfg.tracking_episode = args.episode
    human_cfg.mocap_assist_start = args.mocap_assist_start
    human_cfg.mocap_assist_end = args.mocap_assist_end
    human_cfg.mocap_assist_decay_steps = max(args.mocap_assist_decay_steps, 1)
    human_cfg.mocap_assist_position_gain = args.mocap_assist_position_gain
    human_cfg.mocap_assist_velocity_gain = args.mocap_assist_velocity_gain
    human_cfg.mocap_assist_max_action = args.mocap_assist_max_action
    if args.human_residual_mode != "none":
        if args.human_residual_scale <= 0.0:
            raise ValueError("--human-residual-scale must be positive when --human-residual-mode is enabled")
        human_cfg.residual_joint_names = (
            LOWER_BODY_RESIDUAL_JOINTS
            if args.human_residual_mode == "lower-body"
            else ALL_BODY_RESIDUAL_JOINTS
        )
        human_cfg.residual_scale = args.human_residual_scale
        if args.training_stage == "knee-exo-on-compensation":
            human_cfg.residual_policy_checkpoint = str(args.base_compensation_checkpoint)
    else:
        human_cfg.residual_joint_names = ()
        human_cfg.residual_scale = 0.0
        human_cfg.residual_policy_checkpoint = None
    env_cfg.actions["knee_exo"].max_torque = args.max_knee_torque
    if args.training_stage == "stair-compensation":
        env_cfg.actions.pop("knee_exo", None)

    hip_power_cfg = SceneEntityCfg("robot", joint_names=("L_Hip_.*", "R_Hip_.*"))
    knee_power_cfg = SceneEntityCfg("robot", joint_names=("L_Knee_.*", "R_Knee_.*"))
    ankle_power_cfg = SceneEntityCfg("robot", joint_names=("L_Ankle_.*", "R_Ankle_.*"))
    env_cfg.terminations["fallen"].params["min_height"] = args.fall_min_height
    env_cfg.rewards = {
        "not_fallen_progress": RewardTermCfg(
            func=mdp.not_fallen_progress_reward,
            weight=args.progress_reward_weight,
            params={
                "min_height": args.fall_min_height,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "fallen_penalty": RewardTermCfg(
            func=mdp.root_height_below,
            weight=args.fall_penalty_weight,
            params={
                "min_height": args.fall_min_height,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "lower_limb_joint_power_cost": RewardTermCfg(
            func=mdp.lower_limb_joint_power_cost,
            weight=args.lower_limb_power_weight,
            params={
                "hip_cfg": hip_power_cfg,
                "knee_cfg": knee_power_cfg,
                "ankle_cfg": ankle_power_cfg,
                "hip_weight": args.hip_power_weight,
                "knee_weight": args.knee_power_weight,
                "ankle_weight": args.ankle_power_weight,
                "positive_efficiency": args.positive_work_efficiency,
                "negative_efficiency": args.negative_work_efficiency,
            },
        ),
        "lower_limb_joint_velocity_delta_l2": RewardTermCfg(
            func=mdp.lower_limb_joint_velocity_delta_l2,
            weight=args.joint_vel_smoothness_weight,
            params={
                "hip_cfg": hip_power_cfg,
                "knee_cfg": knee_power_cfg,
                "ankle_cfg": ankle_power_cfg,
                "hip_weight": args.hip_vel_smoothness_weight,
                "knee_weight": args.knee_vel_smoothness_weight,
                "ankle_weight": args.ankle_vel_smoothness_weight,
            },
        ),
    }
    if args.human_residual_action_weight != 0.0:
        env_cfg.rewards["human_residual_action_l2"] = RewardTermCfg(
            func=mdp.action_l2,
            weight=args.human_residual_action_weight,
            params={"action_name": "human_s1"},
        )
    if args.human_residual_action_rate_weight != 0.0:
        env_cfg.rewards["human_residual_action_rate_l2"] = RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=args.human_residual_action_rate_weight,
            params={"action_name": "human_s1"},
        )
    if args.mocap_assist_replacement_weight != 0.0:
        env_cfg.rewards["mocap_assist_replacement_l2"] = RewardTermCfg(
            func=mdp.action_assist_replacement_l2,
            weight=args.mocap_assist_replacement_weight,
            params={"action_name": "human_s1"},
        )
    _add_mocap_qpos_reward(
        env_cfg.rewards,
        name="mocap_root_xyz_tracking",
        weight=args.mocap_root_xyz_weight,
        motion_file=args.motion,
        episode=args.episode,
        columns=ROOT_XYZ_QPOS_COLUMNS,
        std=args.mocap_root_xyz_std,
        root_xy_invariant=False,
    )
    _add_mocap_qpos_reward(
        env_cfg.rewards,
        name="mocap_root_orientation_tracking",
        weight=args.mocap_root_orientation_weight,
        motion_file=args.motion,
        episode=args.episode,
        columns=ROOT_ORIENTATION_QPOS_COLUMNS,
        std=args.mocap_root_orientation_std,
        root_xy_invariant=True,
    )
    _add_mocap_qpos_reward(
        env_cfg.rewards,
        name="mocap_lower_body_tracking",
        weight=args.mocap_lower_body_weight,
        motion_file=args.motion,
        episode=args.episode,
        columns=LOWER_BODY_QPOS_COLUMNS,
        std=args.mocap_lower_body_std,
        root_xy_invariant=True,
    )
    _add_mocap_qpos_reward(
        env_cfg.rewards,
        name="mocap_knee_tracking",
        weight=args.mocap_knee_weight,
        motion_file=args.motion,
        episode=args.episode,
        columns=PRIMARY_KNEE_QPOS_COLUMNS,
        std=args.mocap_knee_std,
        root_xy_invariant=True,
    )
    _add_mocap_foot_reward(
        env_cfg.rewards,
        name="mocap_foot_tracking",
        weight=args.mocap_foot_weight,
        motion_file=args.motion,
        episode=args.episode,
        std=args.mocap_foot_std,
        sides=_parse_foot_sides(args.mocap_foot_sides),
        z_weight=args.mocap_foot_z_weight,
    )
    _add_mocap_foot_reward(
        env_cfg.rewards,
        name="mocap_first_foot_tracking",
        weight=args.mocap_first_foot_weight,
        motion_file=args.motion,
        episode=args.episode,
        std=args.mocap_first_foot_std,
        sides=_parse_foot_sides(args.mocap_first_foot_sides),
        z_weight=args.mocap_foot_z_weight,
        end_frame=args.mocap_first_foot_end_frame,
    )
    _add_mocap_foot_event_reward(
        env_cfg.rewards,
        name="mocap_foot_event_tracking",
        weight=args.mocap_foot_event_weight,
        motion_file=args.motion,
        episode=args.episode,
        std=args.mocap_foot_event_std,
        xy_weight=args.mocap_foot_event_xy_weight,
        z_weight=args.mocap_foot_event_z_weight,
        stance_speed_threshold=args.mocap_foot_event_speed_threshold,
        min_stance_frames=args.mocap_foot_event_min_stance_frames,
        min_event_height_delta=args.mocap_foot_event_min_height_delta,
        window_margin_frames=args.mocap_foot_event_window_margin,
    )

    agent_cfg.seed = args.seed
    if args.actor_init_std is not None:
        agent_cfg.actor.distribution_cfg["init_std"] = args.actor_init_std
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy_coef
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.num_steps_per_env = args.num_steps_per_env
    agent_cfg.save_interval = args.save_interval
    agent_cfg.logger = args.logger
    agent_cfg.run_name = args.run_name
    if args.training_stage == "stair-compensation":
        agent_cfg.experiment_name = "humenv_s1_stair_compensation"
    elif args.training_stage == "knee-exo-on-compensation":
        agent_cfg.experiment_name = "humenv_knee_exo_on_stair_compensation"
    else:
        agent_cfg.experiment_name = "humenv_knee_exo_mocap_track"

    return replace(
        cfg,
        env=env_cfg,
        agent=agent_cfg,
        log_root=args.log_root,
        gpu_ids=None if args.cpu else [args.gpu_id],
    )


def main() -> None:
    args = parse_args()
    cfg = build_train_config(args)
    print(f"[INFO] Training stage: {args.training_stage}")
    print(f"[INFO] Frozen S-1 base controller with fixed mocap reference {args.motion}:{args.episode}")
    if args.training_stage == "stair-compensation":
        print("[INFO] Training full-body S-1 residual torque compensation; knee exo is disabled.")
    elif args.training_stage == "knee-exo-on-compensation":
        print(f"[INFO] Frozen compensation base: {args.base_compensation_checkpoint}")
        print("[INFO] Training only the knee-exo policy on top of S-1 + compensation.")
    else:
        print("[INFO] Training only the knee-exo policy on top of frozen S-1.")
    launch_training(TASK_ID, cfg)


if __name__ == "__main__":
    main()
