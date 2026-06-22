from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch

import hit_exo_humenv.mjlab  # noqa: F401
import hit_exo_humenv.mjlab.mdp as mdp
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


DEFAULT_CHECKPOINT_ROOT = Path(cfg_path("paths", "checkpoint_root"))
DEFAULT_OUTPUT_ROOT = Path(cfg_path("paths", "latent_z_power_runs_root"))
POSITIVE_WORK_EFFICIENCY = cfg_path("metabolic", "positive_work_efficiency")
NEGATIVE_WORK_EFFICIENCY = cfg_path("metabolic", "negative_work_efficiency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless latent-z power rollout logger.")
    parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=cfg_path("eval", "num_envs"))
    parser.add_argument("--steps", type=int, default=cfg_path("eval", "steps"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=cfg_path("eval", "seed"))
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
    return parser.parse_args()


def _latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(root.glob("**/model_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under {root}")
    return checkpoints[0]


def _build_policy(
    agent: Literal["trained", "zero", "random"],
    env: RslRlVecEnvWrapper,
    agent_cfg,
    checkpoint_file: Path | None,
    device: str,
):
    if agent == "zero":

        class PolicyZero:
            def __call__(self, obs):
                del obs
                return torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)

        return PolicyZero()

    if agent == "random":

        class PolicyRandom:
            def __call__(self, obs):
                del obs
                return 2.0 * torch.rand(env.unwrapped.action_space.shape, device=env.unwrapped.device) - 1.0

        return PolicyRandom()

    if checkpoint_file is None:
        raise ValueError("trained evaluation requires a checkpoint")

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


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
    env_cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, args.human_root_height)


def _output_csv_path(args: argparse.Namespace, checkpoint_file: Path | None) -> Path:
    if args.output_csv is not None:
        return args.output_csv
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = checkpoint_file.parent.name if checkpoint_file is not None else args.agent
    return args.output_root / f"{timestamp}_{run_name}_{args.agent}_power.csv"


def _row_from_tensors(
    *,
    step: int,
    env_id: int,
    time_s: float,
    reward: torch.Tensor,
    done: torch.Tensor,
    fallen: torch.Tensor,
    lower_torque: torch.Tensor,
    lower_vel: torch.Tensor,
    lower_exo_torque: torch.Tensor,
    lower_total_actuator_torque: torch.Tensor,
    knee_torque: torch.Tensor,
    knee_total_actuator_torque: torch.Tensor,
    knee_exo_torque: torch.Tensor,
    knee_vel: torch.Tensor,
) -> dict[str, float]:
    lower_human_power = lower_torque * lower_vel
    lower_total_actuator_power = lower_total_actuator_torque * lower_vel
    lower_combined_power = (lower_torque + lower_exo_torque) * lower_vel
    lower_exo_power = lower_exo_torque * lower_vel
    knee_human_power = knee_torque * knee_vel
    knee_total_actuator_power = knee_total_actuator_torque * knee_vel
    knee_exo_power = knee_exo_torque * knee_vel
    knee_combined_power = (knee_torque + knee_exo_torque) * knee_vel
    lower_positive_power = torch.sum(torch.clamp(lower_human_power, min=0.0))
    lower_negative_power = torch.sum(torch.clamp(-lower_human_power, min=0.0))
    lower_total_actuator_positive_power = torch.sum(torch.clamp(lower_total_actuator_power, min=0.0))
    lower_total_actuator_negative_power = torch.sum(torch.clamp(-lower_total_actuator_power, min=0.0))
    lower_metabolic_power = (
        lower_positive_power / POSITIVE_WORK_EFFICIENCY
        + lower_negative_power / NEGATIVE_WORK_EFFICIENCY
    )
    lower_total_actuator_metabolic_power = (
        lower_total_actuator_positive_power / POSITIVE_WORK_EFFICIENCY
        + lower_total_actuator_negative_power / NEGATIVE_WORK_EFFICIENCY
    )
    return {
        "step": float(step),
        "env_id": float(env_id),
        "time_s": time_s,
        "reward_env0": float(reward.detach().cpu()),
        "left_knee_pd_torque_nm": float(knee_total_actuator_torque[0].detach().cpu()),
        "right_knee_pd_torque_nm": float(knee_total_actuator_torque[1].detach().cpu()),
        "knee_pd_torque_l2_env0": float(torch.sum(knee_total_actuator_torque.square()).detach().cpu()),
        "left_knee_passive_adjusted_torque_nm": float(knee_torque[0].detach().cpu()),
        "right_knee_passive_adjusted_torque_nm": float(knee_torque[1].detach().cpu()),
        "knee_passive_adjusted_torque_l2_env0": float(torch.sum(knee_torque.square()).detach().cpu()),
        "left_knee_exo_torque_nm": float(knee_exo_torque[0].detach().cpu()),
        "right_knee_exo_torque_nm": float(knee_exo_torque[1].detach().cpu()),
        "left_knee_joint_vel_rad_s": float(knee_vel[0].detach().cpu()),
        "right_knee_joint_vel_rad_s": float(knee_vel[1].detach().cpu()),
        "human_lower_limb_abs_power_w": float(torch.sum(torch.abs(lower_human_power)).detach().cpu()),
        "human_lower_limb_signed_power_w": float(torch.sum(lower_human_power).detach().cpu()),
        "human_lower_limb_positive_power_w": float(lower_positive_power.detach().cpu()),
        "human_lower_limb_negative_power_w": float(lower_negative_power.detach().cpu()),
        "human_lower_limb_metabolic_power_w": float(lower_metabolic_power.detach().cpu()),
        "human_lower_limb_total_actuator_abs_power_w": float(
            torch.sum(torch.abs(lower_total_actuator_power)).detach().cpu()
        ),
        "human_lower_limb_total_actuator_positive_power_w": float(
            lower_total_actuator_positive_power.detach().cpu()
        ),
        "human_lower_limb_total_actuator_negative_power_w": float(
            lower_total_actuator_negative_power.detach().cpu()
        ),
        "human_lower_limb_total_actuator_metabolic_power_w": float(
            lower_total_actuator_metabolic_power.detach().cpu()
        ),
        "exo_lower_limb_abs_power_w": float(torch.sum(torch.abs(lower_exo_power)).detach().cpu()),
        "exo_lower_limb_signed_power_w": float(torch.sum(lower_exo_power).detach().cpu()),
        "combined_lower_limb_abs_power_w": float(torch.sum(torch.abs(lower_combined_power)).detach().cpu()),
        "combined_lower_limb_signed_power_w": float(torch.sum(lower_combined_power).detach().cpu()),
        "human_knee_abs_power_w": float(torch.sum(torch.abs(knee_human_power)).detach().cpu()),
        "human_knee_signed_power_w": float(torch.sum(knee_human_power).detach().cpu()),
        "human_knee_total_actuator_abs_power_w": float(
            torch.sum(torch.abs(knee_total_actuator_power)).detach().cpu()
        ),
        "human_knee_total_actuator_signed_power_w": float(torch.sum(knee_total_actuator_power).detach().cpu()),
        "exo_knee_abs_power_w": float(torch.sum(torch.abs(knee_exo_power)).detach().cpu()),
        "exo_knee_signed_power_w": float(torch.sum(knee_exo_power).detach().cpu()),
        "combined_knee_abs_power_w": float(torch.sum(torch.abs(knee_combined_power)).detach().cpu()),
        "combined_knee_signed_power_w": float(torch.sum(knee_combined_power).detach().cpu()),
        "fallen_env0": int(bool(fallen.detach().cpu())),
        "done_env0": int(bool(done.detach().cpu())),
    }


def main() -> None:
    args = parse_args()
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_file = args.checkpoint_file
    if args.agent == "trained" and checkpoint_file is None:
        checkpoint_file = _latest_checkpoint(args.checkpoint_root)

    env_cfg = load_env_cfg(TASK_ID, play=False)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.actions["knee_exo"].joint_names = exo_joint_names(args.exo_joint_group)
    env_cfg.actions["knee_exo"].max_torque = exo_torque_limits(args.exo_joint_group)
    _configure_walk_command(env_cfg, args)
    _configure_human_gait(env_cfg, args)
    agent_cfg = load_rl_cfg(TASK_ID)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    obs, _ = env.reset()
    policy = _build_policy(args.agent, env, agent_cfg, checkpoint_file, device)

    robot = raw_env.scene["robot"]
    knee_joint_ids, knee_joint_names = robot.find_joints(("L_Knee_x", "R_Knee_x"), preserve_order=True)
    knee_joint_ids = torch.as_tensor(knee_joint_ids, dtype=torch.long, device=raw_env.device)
    hip_joint_ids, hip_joint_names = robot.find_joints(("L_Hip_.*", "R_Hip_.*"), preserve_order=True)
    ankle_joint_ids, ankle_joint_names = robot.find_joints(("L_Ankle_.*", "R_Ankle_.*"), preserve_order=True)
    lower_joint_ids = torch.as_tensor(
        [*hip_joint_ids, *knee_joint_ids.detach().cpu().tolist(), *ankle_joint_ids],
        dtype=torch.long,
        device=raw_env.device,
    )
    knee_exo = raw_env.action_manager.get_term("knee_exo")
    exo_joint_ids, exo_joint_names_used = robot.find_joints(knee_exo.joint_names, preserve_order=True)
    exo_joint_ids = torch.as_tensor(exo_joint_ids, dtype=torch.long, device=raw_env.device)
    lower_exo_positions = _positions_in_reference(lower_joint_ids, exo_joint_ids)
    knee_exo_positions = _positions_in_reference(knee_joint_ids, exo_joint_ids)

    output_csv = _output_csv_path(args, checkpoint_file)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_fallen = 0
    total_done = 0
    try:
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                action = policy(obs)
                obs, reward, dones, _ = env.step(action)
                fallen = raw_env.termination_manager.get_term("fallen")
                lower_torque = mdp.actuator_joint_force_excluding_passive(raw_env, robot, lower_joint_ids)
                lower_total_actuator_torque = robot.data.qfrc_actuator[:, lower_joint_ids]
                lower_vel = robot.data.joint_vel[:, lower_joint_ids]
                knee_torque = mdp.actuator_joint_force_excluding_passive(raw_env, robot, knee_joint_ids)
                knee_total_actuator_torque = robot.data.qfrc_actuator[:, knee_joint_ids]
                exo_torque = getattr(knee_exo, "_processed_actions", knee_exo.raw_action)
                knee_vel = robot.data.joint_vel[:, knee_joint_ids]
                lower_exo_torque = torch.zeros_like(lower_torque)
                lower_exo_torque[:, lower_exo_positions] = exo_torque
                knee_exo_torque = torch.zeros(raw_env.num_envs, len(knee_joint_ids), device=raw_env.device)
                if knee_exo_positions:
                    knee_columns = torch.as_tensor(knee_exo_positions, dtype=torch.long, device=raw_env.device)
                    knee_exo_torque[:, knee_columns] = exo_torque[:, _matching_source_columns(exo_joint_ids, knee_joint_ids)]
                time_s = step * raw_env.step_dt
                total_fallen += int(fallen.detach().sum().cpu())
                total_done += int(dones.detach().sum().cpu())
                for env_id in range(raw_env.num_envs):
                    rows.append(
                        _row_from_tensors(
                            step=step,
                            env_id=env_id,
                            time_s=time_s,
                            reward=reward[env_id],
                            done=dones[env_id],
                            fallen=fallen[env_id],
                            lower_torque=lower_torque[env_id],
                            lower_vel=lower_vel[env_id],
                            lower_exo_torque=lower_exo_torque[env_id],
                            lower_total_actuator_torque=lower_total_actuator_torque[env_id],
                            knee_torque=knee_torque[env_id],
                            knee_total_actuator_torque=knee_total_actuator_torque[env_id],
                            knee_exo_torque=knee_exo_torque[env_id],
                            knee_vel=knee_vel[env_id],
                        )
                    )
    finally:
        env.close()

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "agent": args.agent,
        "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else None,
        "csv": str(output_csv),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "samples": len(rows),
        "device": device,
        "seed": args.seed,
        "total_fallen": total_fallen,
        "total_done": total_done,
        "knee_joint_names": list(knee_joint_names),
        "lower_limb_joint_names": [*hip_joint_names, *knee_joint_names, *ankle_joint_names],
        "exo_joint_group": args.exo_joint_group,
        "exo_joint_names": list(exo_joint_names_used),
        "positive_work_efficiency": POSITIVE_WORK_EFFICIENCY,
        "negative_work_efficiency": NEGATIVE_WORK_EFFICIENCY,
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote latent-z power CSV: {output_csv}")


def _positions_in_reference(reference_ids: torch.Tensor, query_ids: torch.Tensor) -> list[int]:
    reference = [int(value) for value in reference_ids.detach().cpu().tolist()]
    positions = []
    for query in query_ids.detach().cpu().tolist():
        if int(query) in reference:
            positions.append(reference.index(int(query)))
    return positions


def _matching_source_columns(source_ids: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
    source = [int(value) for value in source_ids.detach().cpu().tolist()]
    columns = []
    for query in query_ids.detach().cpu().tolist():
        if int(query) in source:
            columns.append(source.index(int(query)))
    return torch.as_tensor(columns, dtype=torch.long, device=source_ids.device)


if __name__ == "__main__":
    main()
