from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

import hit_exo_humenv.mjlab  # noqa: F401
from hit_exo_humenv.latent_z_config import cfg_path
from hit_exo_humenv.mjlab.walking_env_cfg import TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


DEFAULT_MOTION = Path("/home/yhzhu/AI/humenv/data_preparation/humenv_from_protomotions/0009_walking_medium01_poses.hdf5")
DEFAULT_OUTPUT_ROOT = Path(cfg_path("paths", "s1_latent_selection_root"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select S-1 walking latents by matching a HumEnv mocap gait.")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direction", type=float, default=0.0)
    parser.add_argument("--candidate-speeds", type=float, nargs="+", default=cfg_path("human_s1", "candidate_speeds"))
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--score-start-step", type=int, default=30)
    parser.add_argument("--s1-samples", type=int, default=cfg_path("human_s1", "num_samples_per_inference"))
    parser.add_argument("--s1-workers", type=int, default=cfg_path("human_s1", "max_workers"))
    parser.add_argument("--mean-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--root-z-weight", type=float, default=2.0)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument("--joint-weight", type=float, default=0.35)
    parser.add_argument("--knee-weight", type=float, default=0.75)
    parser.add_argument("--smooth-weight", type=float, default=0.05)
    parser.add_argument("--fallen-weight", type=float, default=25.0)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_motion(path: Path, episode: str) -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float32)
        qvel = np.asarray(ep["qvel"][:], dtype=np.float32)
        dt = float(ep.attrs.get("dt", hf.attrs.get("dt", 1.0 / 30.0)))
        metadata = {
            "motion": str(path),
            "episode": episode,
            "frames": int(qpos.shape[0]),
            "dt": dt,
            "attrs": {key: _jsonable(value) for key, value in hf.attrs.items()},
            "episode_attrs": {key: _jsonable(value) for key, value in ep.attrs.items()},
        }
    return qpos, qvel, dt, metadata


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return _jsonable(value)


def resample_reference(qpos: np.ndarray, source_dt: float, target_dt: float, steps: int) -> np.ndarray:
    source_t = np.arange(qpos.shape[0], dtype=np.float32) * source_dt
    target_t = np.arange(steps, dtype=np.float32) * target_dt
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    out = np.empty((steps, qpos.shape[1]), dtype=np.float32)
    for col in range(qpos.shape[1]):
        out[:, col] = np.interp(target_t, source_t, qpos[:, col]).astype(np.float32)
    out[:, :2] -= out[0:1, :2]
    return out


def scalar_speed_xy(qpos: torch.Tensor, dt: float) -> torch.Tensor:
    delta = qpos[1:, :, :2] - qpos[:-1, :, :2]
    speed = torch.linalg.norm(delta, dim=-1) / dt
    return torch.cat([speed[0:1], speed], dim=0)


def reference_speed_xy(qpos: torch.Tensor, dt: float) -> torch.Tensor:
    delta = qpos[1:, :2] - qpos[:-1, :2]
    speed = torch.linalg.norm(delta, dim=-1) / dt
    return torch.cat([speed[0:1], speed], dim=0)


def build_env(args: argparse.Namespace, device: str, candidates: list[tuple[float, float]]) -> RslRlVecEnvWrapper:
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = len(candidates)
    env_cfg.seed = args.seed

    speeds = tuple(sorted({speed for _, speed in candidates}))
    directions = tuple(sorted({direction for direction, _ in candidates}))
    command_cfg = env_cfg.commands["walk_speed"]
    command_cfg.include_direction = True
    command_cfg.speed_choices = speeds
    command_cfg.speed_range = (min(speeds), max(speeds))
    command_cfg.direction_choices_deg = directions
    command_cfg.direction_range_deg = (min(directions), max(directions))
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.reset_on_resample = False

    human_cfg = env_cfg.actions["human_s1"]
    human_cfg.speed_bins = speeds
    human_cfg.direction_bins_deg = directions
    human_cfg.num_samples_per_inference = args.s1_samples
    human_cfg.max_workers = args.s1_workers
    human_cfg.mean_action = args.mean_action

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(TASK_ID)
    return RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)


def set_candidate_commands(env: RslRlVecEnvWrapper, candidates: list[tuple[float, float]]) -> None:
    command = env.unwrapped.command_manager.get_term("walk_speed")
    values = torch.tensor([[speed, direction] for direction, speed in candidates], device=env.unwrapped.device)
    command.command[:] = values
    command.time_left[:] = 1.0e9


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, float | str]], dict[str, object], Path]:
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    qpos, _qvel, motion_dt, motion_metadata = load_motion(args.motion, args.episode)
    candidates = [(args.direction, speed) for speed in args.candidate_speeds]
    env = build_env(args, device, candidates)
    set_candidate_commands(env, candidates)
    env.reset()
    set_candidate_commands(env, candidates)

    target_dt = env.unwrapped.step_dt
    steps = min(args.steps, int(np.ceil((qpos.shape[0] - 1) * motion_dt / target_dt)) + 1)
    ref_qpos_np = resample_reference(qpos, motion_dt, target_dt, steps)
    ref_qpos = torch.as_tensor(ref_qpos_np, device=env.unwrapped.device)
    ref_speed = reference_speed_xy(ref_qpos, target_dt)

    zero_exo_action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    rollout_qpos = []
    human_actions = []
    fallen_counts = torch.zeros(len(candidates), device=env.unwrapped.device)
    obs, _ = env.reset()
    set_candidate_commands(env, candidates)

    with torch.inference_mode():
        for _ in range(steps):
            obs, _reward, dones, _extras = env.step(zero_exo_action)
            del obs
            rollout_qpos.append(env.unwrapped.sim.data.qpos.detach().clone())
            human_actions.append(env.unwrapped.action_manager.get_term("human_s1")._processed_actions.detach().clone())
            fallen_counts += dones.to(dtype=torch.float32)
            set_candidate_commands(env, candidates)

    rollout = torch.stack(rollout_qpos, dim=0)
    actions = torch.stack(human_actions, dim=0)
    rollout_xy = rollout[:, :, :2] - rollout[0:1, :, :2]
    rollout_aligned = rollout.clone()
    rollout_aligned[:, :, :2] = rollout_xy

    start = min(max(args.score_start_step, 0), steps - 1)
    scored_rollout = rollout_aligned[start:]
    scored_ref = ref_qpos[start:, None, :]

    root_z_rmse = torch.sqrt(torch.mean(torch.square(scored_rollout[:, :, 2] - scored_ref[:, :, 2]), dim=0))
    speed_rmse = torch.sqrt(torch.mean(torch.square(scalar_speed_xy(rollout_aligned, target_dt)[start:] - ref_speed[start:, None]), dim=0))
    joint_rmse = torch.sqrt(torch.mean(torch.square(scored_rollout[:, :, 7:] - scored_ref[:, :, 7:]), dim=(0, 2)))

    knee_cols = torch.tensor([7 + 3, 7 + 15], device=env.unwrapped.device)
    knee_rmse = torch.sqrt(torch.mean(torch.square(scored_rollout[:, :, knee_cols] - scored_ref[:, :, knee_cols]), dim=(0, 2)))
    action_rate = torch.mean(torch.abs(actions[1:] - actions[:-1]), dim=(0, 2)) if steps > 1 else torch.zeros(len(candidates), device=env.unwrapped.device)

    score = (
        args.root_z_weight * root_z_rmse
        + args.speed_weight * speed_rmse
        + args.joint_weight * joint_rmse
        + args.knee_weight * knee_rmse
        + args.smooth_weight * action_rate
        + args.fallen_weight * fallen_counts
    )

    human_term = env.unwrapped.action_manager.get_term("human_s1")
    rows = []
    for idx, (direction, speed) in enumerate(candidates):
        task = f"move-ego-{direction:g}-{speed:g}"
        rows.append(
            {
                "rank": 0.0,
                "task": task,
                "direction_deg": direction,
                "speed": speed,
                "score": float(score[idx].detach().cpu()),
                "root_z_rmse": float(root_z_rmse[idx].detach().cpu()),
                "speed_rmse": float(speed_rmse[idx].detach().cpu()),
                "joint_rmse": float(joint_rmse[idx].detach().cpu()),
                "knee_rmse": float(knee_rmse[idx].detach().cpu()),
                "action_rate": float(action_rate[idx].detach().cpu()),
                "fallen_count": float(fallen_counts[idx].detach().cpu()),
            }
        )
        human_term._policy.z_for_task(task)

    rows.sort(key=lambda row: float(row["score"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = float(rank)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_root / f"{timestamp}_{args.motion.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "motion": motion_metadata,
        "device": device,
        "step_dt": target_dt,
        "steps": steps,
        "score_start_step": args.score_start_step,
        "candidates": [{"direction_deg": d, "speed": s, "task": f"move-ego-{d:g}-{s:g}"} for d, s in candidates],
        "weights": {
            "root_z": args.root_z_weight,
            "speed": args.speed_weight,
            "joint": args.joint_weight,
            "knee": args.knee_weight,
            "smooth": args.smooth_weight,
            "fallen": args.fallen_weight,
        },
        "best": rows[0],
        "top_k": rows[: args.top_k],
        "args": {key: _jsonable_arg(value) for key, value in vars(args).items()},
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows[: args.top_k]:
        latent = human_term._policy.z_for_task(str(row["task"]))
        torch.save(latent.detach().cpu(), output_dir / f"rank_{int(row['rank']):02d}_{row['task']}.pt")

    env.close()
    return rows, summary, output_dir


def main() -> None:
    rows, summary, output_dir = evaluate(parse_args())
    print(json.dumps({"best": summary["best"], "top_k": summary["top_k"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
