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
from hit_exo_humenv.latent_z_config import cfg_path
from hit_exo_humenv.mjlab.walking_env_cfg import (
    TASK_ID,
    TRAIN_WALKING_DIRECTION_CHOICES_DEG,
    TRAIN_WALKING_SPEED_CHOICES,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


DEFAULT_CHECKPOINT_ROOT = Path(cfg_path("paths", "checkpoint_root"))
DEFAULT_OUTPUT_ROOT = Path(cfg_path("paths", "legacy_eval_root"))


def _latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(
        root.glob("**/model_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under {root}")
    return checkpoints[0]


def _mean(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu())


def _sum(value: torch.Tensor) -> float:
    return float(value.detach().sum().cpu())


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


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate knee-exo mjlab policy and log rollout metrics.")
    parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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


def _empty_speed_bucket() -> dict[str, float]:
    return {
        "samples": 0.0,
        "reward_sum": 0.0,
        "root_z_sum": 0.0,
        "root_z_min": float("inf"),
        "knee_pd_torque_l2_sum": 0.0,
        "knee_pd_torque_abs_sum": 0.0,
        "knee_exo_torque_abs_sum": 0.0,
        "fallen_sum": 0.0,
        "done_sum": 0.0,
    }


def _finalize_speed_bucket(bucket: dict[str, float]) -> dict[str, float]:
    samples = bucket["samples"]
    if samples <= 0.0:
        return {}
    return {
        "samples": samples,
        "mean_reward": bucket["reward_sum"] / samples,
        "mean_root_z": bucket["root_z_sum"] / samples,
        "min_root_z": bucket["root_z_min"],
        "mean_knee_pd_torque_l2": bucket["knee_pd_torque_l2_sum"] / samples,
        "mean_knee_pd_torque_abs": bucket["knee_pd_torque_abs_sum"] / (2.0 * samples),
        "mean_knee_exo_torque_abs": bucket["knee_exo_torque_abs_sum"] / (2.0 * samples),
        "fallen_count": bucket["fallen_sum"],
        "done_count": bucket["done_sum"],
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
    _configure_walk_command(env_cfg, args)
    agent_cfg = load_rl_cfg(TASK_ID)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    obs, _ = env.reset()

    policy = _build_policy(args.agent, env, agent_cfg, checkpoint_file, device)

    robot = raw_env.scene["robot"]
    knee_joint_ids, knee_joint_names = robot.find_joints(("L_Knee_x", "R_Knee_x"), preserve_order=True)
    knee_joint_ids = torch.as_tensor(knee_joint_ids, dtype=torch.long, device=raw_env.device)
    knee_exo = raw_env.action_manager.get_term("knee_exo")
    speed_command = raw_env.command_manager.get_term("walk_speed")

    rows: list[dict[str, float]] = []
    speed_buckets: dict[str, dict[str, float]] = {}
    total_fallen = 0.0
    total_time_out = 0.0
    total_done = 0.0

    with torch.inference_mode():
        for step in range(args.steps):
            action = policy(obs)
            obs, reward, dones, _ = env.step(action)

            fallen = raw_env.termination_manager.get_term("fallen")
            time_out = raw_env.termination_manager.get_term("time_out")
            pd_torque = robot.data.qfrc_actuator[:, knee_joint_ids]
            exo_action = knee_exo.raw_action
            exo_torque = getattr(knee_exo, "_processed_actions", exo_action)
            root_z = robot.data.root_link_pos_w[:, 2]
            walk_speed = speed_command.command[:, 0]
            walk_direction = speed_command.command[:, 1]
            pd_l2 = torch.sum(pd_torque.square(), dim=1)

            done_count = _sum(dones)
            fallen_count = _sum(fallen)
            time_out_count = _sum(time_out)
            total_done += done_count
            total_fallen += fallen_count
            total_time_out += time_out_count

            rows.append(
                {
                    "step": float(step + 1),
                    "reward_mean": _mean(reward),
                    "done_count": done_count,
                    "fallen_count": fallen_count,
                    "time_out_count": time_out_count,
                    "walk_speed_mean": _mean(walk_speed),
                    "walk_direction_mean": _mean(walk_direction),
                    "root_z_mean": _mean(root_z),
                    "root_z_min": float(root_z.detach().min().cpu()),
                    "knee_pd_torque_l2_mean": _mean(pd_l2),
                    "knee_pd_torque_abs_mean": _mean(pd_torque.abs()),
                    "knee_exo_action_abs_mean": _mean(exo_action.abs()),
                    "knee_exo_torque_abs_mean": _mean(exo_torque.abs()),
                }
            )

            speed_direction = torch.unique(torch.stack([walk_speed, walk_direction], dim=1), dim=0)
            for speed, direction in speed_direction.detach().cpu().tolist():
                key = f"{speed:g}@{direction:g}"
                bucket = speed_buckets.setdefault(key, _empty_speed_bucket())
                mask = (walk_speed == speed) & (walk_direction == direction)
                sample_count = float(mask.sum().detach().cpu())
                bucket["samples"] += sample_count
                bucket["reward_sum"] += float(reward[mask].detach().sum().cpu())
                bucket["root_z_sum"] += float(root_z[mask].detach().sum().cpu())
                bucket["root_z_min"] = min(bucket["root_z_min"], float(root_z[mask].detach().min().cpu()))
                bucket["knee_pd_torque_l2_sum"] += float(pd_l2[mask].detach().sum().cpu())
                bucket["knee_pd_torque_abs_sum"] += float(pd_torque[mask].abs().detach().sum().cpu())
                bucket["knee_exo_torque_abs_sum"] += float(exo_torque[mask].abs().detach().sum().cpu())
                bucket["fallen_sum"] += float(fallen[mask].detach().sum().cpu())
                bucket["done_sum"] += float(dones[mask].detach().sum().cpu())

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = checkpoint_file.parent.name if checkpoint_file is not None else args.agent
    output_dir = args.output_root / f"{timestamp}_{run_name}_{args.agent}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "task_id": TASK_ID,
        "agent": args.agent,
        "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else None,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "device": device,
        "seed": args.seed,
        "random_walk_speed": args.random_walk_speed,
        "walk_speed": args.walk_speed,
        "walk_speed_choices": args.walk_speed_choices,
        "random_walk_direction": args.random_walk_direction,
        "walk_direction": args.walk_direction,
        "walk_direction_choices": args.walk_direction_choices,
        "speed_resampling_time_range": args.speed_resampling_time_range,
        "knee_joint_names": list(knee_joint_names),
        "total_done": total_done,
        "total_fallen": total_fallen,
        "total_time_out": total_time_out,
        "mean_reward": sum(row["reward_mean"] for row in rows) / len(rows),
        "mean_root_z": sum(row["root_z_mean"] for row in rows) / len(rows),
        "min_root_z": min(row["root_z_min"] for row in rows),
        "mean_knee_pd_torque_l2": sum(row["knee_pd_torque_l2_mean"] for row in rows) / len(rows),
        "mean_knee_pd_torque_abs": sum(row["knee_pd_torque_abs_mean"] for row in rows) / len(rows),
        "mean_knee_exo_action_abs": sum(row["knee_exo_action_abs_mean"] for row in rows) / len(rows),
        "mean_knee_exo_torque_abs": sum(row["knee_exo_torque_abs_mean"] for row in rows) / len(rows),
        "per_speed_direction": {
            key: _finalize_speed_bucket(bucket)
            for key, bucket in sorted(
                speed_buckets.items(),
                key=lambda item: tuple(float(part) for part in item[0].split("@")),
            )
        },
    }

    _write_csv(output_dir / "steps.csv", rows)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote eval logs to: {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
