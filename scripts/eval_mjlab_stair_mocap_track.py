#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import hit_exo_humenv.mjlab  # noqa: F401
from hit_exo_humenv.latent_z_config import cfg_path
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from run_s1_mocap_track_visual import TerrainHeightLookup, compute_mocap_foot_tracks
from train_mjlab_knee_exo_mocap_track import TASK_ID, build_train_config, load_motion


DEFAULT_UPSTAIRS_MOTION = Path(
    "/home/yhzhu/AI/humenv/data_preparation/humenv_from_protomotions/KIT_167_upstairs03_poses.hdf5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless stair rollout check for mocap tracking checkpoints."
    )
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument(
        "--training-stage",
        choices=("stair-compensation", "knee-exo", "knee-exo-on-compensation"),
        required=True,
    )
    parser.add_argument("--base-compensation-checkpoint", type=Path, default=None)
    parser.add_argument("--motion", type=Path, default=DEFAULT_UPSTAIRS_MOTION)
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--terrain-out-dir", type=Path, default=Path(".omx/mjlab_stair_eval_terrain"))
    parser.add_argument("--output-root", type=Path, default=Path(".omx/mjlab_stair_policy_eval"))
    parser.add_argument("--stair-width", type=float, default=1.4)
    parser.add_argument("--stair-height", type=float, default=0.135)
    parser.add_argument("--stair-steps", type=int, default=0)
    parser.add_argument("--human-residual-scale", type=float, default=0.35)
    return parser.parse_args()


def _train_config_args(args: argparse.Namespace, terrain_dir: Path) -> argparse.Namespace:
    return Namespace(
        motion=args.motion,
        episode=args.episode,
        training_stage=args.training_stage,
        base_compensation_checkpoint=args.base_compensation_checkpoint,
        num_envs=1,
        max_iterations=1,
        num_steps_per_env=cfg_path("train", "num_steps_per_env"),
        save_interval=cfg_path("train", "save_interval"),
        seed=42,
        gpu_id=0,
        cpu=args.device == "cpu",
        logger="tensorboard",
        run_name="stair_eval",
        log_root="logs/rsl_rl",
        actor_init_std=None,
        entropy_coef=None,
        s1_samples=cfg_path("human_s1", "num_samples_per_inference"),
        s1_workers=cfg_path("human_s1", "max_workers"),
        s1_action_smoothing=0.15,
        human_residual_mode="all" if args.training_stage != "knee-exo" else "none",
        human_residual_scale=args.human_residual_scale if args.training_stage != "knee-exo" else 0.0,
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
        episode_length_s=None,
        physics_timestep=cfg_path("simulation", "mujoco_timestep"),
        decimation=cfg_path("simulation", "decimation"),
        terrain="stairs",
        terrain_out_dir=terrain_dir,
        stair_width=args.stair_width,
        stair_height=args.stair_height,
        stair_steps=args.stair_steps,
        max_knee_torque=cfg_path("exo", "max_knee_torque"),
        progress_reward_weight=cfg_path("reward", "not_fallen_progress_weight"),
        mocap_root_xyz_weight=1.0,
        mocap_root_xyz_std=0.25,
        mocap_root_orientation_weight=0.5,
        mocap_root_orientation_std=0.35,
        mocap_lower_body_weight=1.0,
        mocap_lower_body_std=0.45,
        mocap_knee_weight=0.8,
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


def _first_elevated_event(landing_events: list[dict[str, object]], terrain: TerrainHeightLookup) -> dict[str, object]:
    for event in landing_events:
        if event["event_type"] != "touchdown":
            continue
        xy = np.array([event["mocap_x"], event["mocap_y"]], dtype=np.float64)
        if terrain.height_at(xy)["name"] != "ground":
            return event
    raise RuntimeError("No elevated mocap touchdown found")


def main() -> None:
    args = parse_args()
    configure_torch_backends()
    if not args.checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{timestamp}_{args.checkpoint_file.parent.name}_{args.checkpoint_file.stem}"
    terrain_dir = out_dir / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    terrain_dir.mkdir(parents=True, exist_ok=True)

    qpos, _qvel, _dt = load_motion(args.motion, args.episode)
    cfg = build_train_config(_train_config_args(args, terrain_dir))
    terrain_xml = terrain_dir / "humenv_mocap_stairs.xml"
    terrain = TerrainHeightLookup(terrain_xml)
    mocap_tracks, landing_events = compute_mocap_foot_tracks(
        qpos,
        terrain_xml,
        stance_speed_threshold=0.18,
        min_stance_frames=4,
    )
    first_event = _first_elevated_event(landing_events, terrain)
    first_frame = int(first_event["start_frame"])
    first_end_frame = int(first_event.get("end_frame", first_frame))
    first_side = str(first_event["side"])
    first_mocap = np.array(
        [first_event["mocap_x"], first_event["mocap_y"], first_event["mocap_z"]],
        dtype=np.float64,
    )
    first_mocap_terrain = terrain.height_at(first_mocap[:2])

    raw_env = ManagerBasedRlEnv(cfg=cfg.env, device=args.device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=cfg.agent.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg.agent), device=args.device)
    runner.load(str(args.checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = raw_env.scene["robot"]
    body_names = list(robot.body_names)
    foot_ids = {
        "L": (body_names.index("L_Ankle"), body_names.index("L_Toe")),
        "R": (body_names.index("R_Ankle"), body_names.index("R_Toe")),
    }

    def foot_center(side: str) -> np.ndarray:
        pos = robot.data.body_link_pos_w.detach().cpu().numpy()[0]
        ankle_id, toe_id = foot_ids[side]
        return 0.5 * (pos[ankle_id] + pos[toe_id])

    obs, _ = env.reset()
    reward_sum = 0.0
    first_done_step = None
    first_fallen_step = None
    sim_at_first = None
    first_window_samples: list[tuple[int, np.ndarray, dict[str, object]]] = []
    max_terrain_z_under_feet = 0.0
    terrain_names_reached: set[str] = set()
    with torch.inference_mode():
        for step in range(1, min(args.steps, len(qpos) - 1) + 1):
            action = policy(obs)
            obs, reward, dones, _info = env.step(action)
            reward_sum += float(reward.detach().cpu().numpy()[0])
            fallen = bool(raw_env.termination_manager.get_term("fallen").detach().cpu().numpy()[0])
            centers = {side: foot_center(side) for side in foot_ids}
            for center in centers.values():
                height = terrain.height_at(center[:2])
                terrain_names_reached.add(str(height["name"]))
                max_terrain_z_under_feet = max(max_terrain_z_under_feet, float(height["z"]))
            if step == first_frame:
                sim_at_first = centers[first_side].copy()
            if first_frame <= step <= first_end_frame:
                center = centers[first_side].copy()
                first_window_samples.append((step, center, terrain.height_at(center[:2])))
            if fallen and first_fallen_step is None:
                first_fallen_step = step
            if bool(dones.detach().cpu().numpy()[0]):
                first_done_step = step
                break

    if sim_at_first is None:
        sim_at_first = foot_center(first_side)
    first_sim_terrain = terrain.height_at(sim_at_first[:2])
    delta = sim_at_first - first_mocap
    if not first_window_samples:
        first_window_samples.append((first_frame, sim_at_first.copy(), first_sim_terrain))
    best_step, best_center, best_terrain = min(
        first_window_samples,
        key=lambda item: float(np.linalg.norm(item[1] - first_mocap)),
    )
    best_delta = best_center - first_mocap
    first_window_terrain_names = sorted({str(item[2]["name"]) for item in first_window_samples})
    first_window_max_terrain_z = max(float(item[2]["z"]) for item in first_window_samples)
    summary = {
        "checkpoint_file": str(args.checkpoint_file),
        "training_stage": args.training_stage,
        "base_compensation_checkpoint": str(args.base_compensation_checkpoint)
        if args.base_compensation_checkpoint
        else None,
        "motion": str(args.motion),
        "output_dir": str(out_dir),
        "steps_run": first_done_step or min(args.steps, len(qpos) - 1),
        "first_done_step": first_done_step,
        "first_fallen_step": first_fallen_step,
        "reward_sum": reward_sum,
        "first_step_frame": first_frame,
        "first_step_end_frame": first_end_frame,
        "first_step_side": first_side,
        "first_sim_terrain": first_sim_terrain["name"],
        "first_mocap_terrain": first_mocap_terrain["name"],
        "first_sim_terrain_z": first_sim_terrain["z"],
        "first_mocap_terrain_z": first_mocap_terrain["z"],
        "first_landing_error": float(np.linalg.norm(delta)),
        "first_landing_xy_error": float(np.linalg.norm(delta[:2])),
        "first_landing_z_error": float(abs(delta[2])),
        "first_window_best_step": int(best_step),
        "first_window_best_terrain": best_terrain["name"],
        "first_window_best_terrain_z": best_terrain["z"],
        "first_window_min_landing_error": float(np.linalg.norm(best_delta)),
        "first_window_min_landing_xy_error": float(np.linalg.norm(best_delta[:2])),
        "first_window_min_landing_z_error": float(abs(best_delta[2])),
        "first_window_max_terrain_z": first_window_max_terrain_z,
        "first_window_terrain_names": first_window_terrain_names,
        "max_terrain_z_under_feet": max_terrain_z_under_feet,
        "terrain_names_reached": sorted(terrain_names_reached),
        "success_first_step_on_elevated": first_sim_terrain["name"] != "ground",
        "success_first_step_window_on_elevated": any(
            item[2]["name"] != "ground" for item in first_window_samples
        ),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
