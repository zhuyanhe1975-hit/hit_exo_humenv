from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_ROOT = Path("logs/profile/latent_z_collection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile latent-z collection hot path with PyTorch profiler.")
    parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-envs", type=int, default=cfg_path("train", "num_envs"))
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--active-steps", type=int, default=30)
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
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--instrument-hot-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--row-limit", type=int, default=40)
    return parser.parse_args()


def _latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(root.glob("**/model_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under {root}")
    return checkpoints[0]


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
        raise ValueError("trained profiling requires a checkpoint")

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _run_steps(env: RslRlVecEnvWrapper, policy, obs: torch.Tensor, steps: int) -> torch.Tensor:
    with torch.inference_mode():
        for _ in range(steps):
            with torch.profiler.record_function("latent_z/policy"):
                action = policy(obs)
            with torch.profiler.record_function("latent_z/env_step"):
                obs, _, _, _ = env.step(action)
    return obs


def _wrap_method(cls, name: str, label: str) -> None:
    original = getattr(cls, name)
    if getattr(original, "_latent_z_profile_wrapped", False):
        return

    def wrapped(self, *args, **kwargs):
        with torch.profiler.record_function(label):
            return original(self, *args, **kwargs)

    wrapped._latent_z_profile_wrapped = True
    setattr(cls, name, wrapped)


def _wrap_function(module, name: str, label: str) -> None:
    original = getattr(module, name)
    if getattr(original, "_latent_z_profile_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    wrapped._latent_z_profile_wrapped = True
    setattr(module, name, wrapped)


def _install_hot_path_labels() -> None:
    import hit_exo_humenv.mjlab.actions as action_mod
    from hit_exo_humenv.mjlab.actions import KneeExoTorqueAction, S1HumanAction
    from hit_exo_humenv.mjlab.commands import WalkingSpeedCommand
    from hit_exo_humenv.s1_policy import MetamotivoS1Policy
    from mjlab.managers.action_manager import ActionManager
    from mjlab.managers.observation_manager import ObservationManager
    from mjlab.managers.reward_manager import RewardManager
    from mjlab.managers.termination_manager import TerminationManager

    _wrap_method(ActionManager, "process_action", "latent_z/action_manager/process_action")
    _wrap_method(ActionManager, "apply_action", "latent_z/action_manager/apply_action")
    _wrap_method(ObservationManager, "compute", "latent_z/observation_manager/compute")
    _wrap_method(ObservationManager, "compute_group", "latent_z/observation_manager/compute_group")
    _wrap_method(RewardManager, "compute", "latent_z/reward_manager/compute")
    _wrap_method(TerminationManager, "compute", "latent_z/termination_manager/compute")
    _wrap_method(WalkingSpeedCommand, "compute", "latent_z/walk_command/compute")
    _wrap_method(S1HumanAction, "process_actions", "latent_z/s1_action/process_actions")
    _wrap_method(S1HumanAction, "_commanded_latents", "latent_z/s1_action/commanded_latents")
    _wrap_method(S1HumanAction, "_build_command_latent_table", "latent_z/s1_action/build_latent_table")
    _wrap_method(KneeExoTorqueAction, "process_actions", "latent_z/knee_exo/process_actions")
    _wrap_method(KneeExoTorqueAction, "apply_actions", "latent_z/knee_exo/apply_actions")
    _wrap_method(MetamotivoS1Policy, "act_tensor", "latent_z/s1_policy/act_tensor")
    _wrap_function(action_mod, "_compute_humenv_proprio_obs", "latent_z/s1_action/proprio_obs")


class _ProfiledTerm:
    def __init__(self, func, label: str) -> None:
        self._func = func
        self._label = label

    def __call__(self, *args, **kwargs):
        with torch.profiler.record_function(self._label):
            return self._func(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._func.reset(*args, **kwargs)

    def debug_vis(self, *args, **kwargs):
        return self._func.debug_vis(*args, **kwargs)


def _wrap_manager_terms(raw_env: ManagerBasedRlEnv) -> None:
    reward_manager = raw_env.reward_manager
    for name, term_cfg in zip(reward_manager._term_names, reward_manager._term_cfgs, strict=False):
        if getattr(term_cfg.func, "_latent_z_profile_term_wrapped", False):
            continue
        wrapped = _ProfiledTerm(term_cfg.func, f"latent_z/reward_term/{name}")
        wrapped._latent_z_profile_term_wrapped = True
        term_cfg.func = wrapped

    observation_manager = raw_env.observation_manager
    for group_name, term_names in observation_manager._group_obs_term_names.items():
        term_cfgs = observation_manager._group_obs_term_cfgs[group_name]
        for term_name, term_cfg in zip(term_names, term_cfgs, strict=False):
            if getattr(term_cfg.func, "_latent_z_profile_term_wrapped", False):
                continue
            label = f"latent_z/obs_term/{group_name}/{term_name}"
            wrapped = _ProfiledTerm(term_cfg.func, label)
            wrapped._latent_z_profile_term_wrapped = True
            term_cfg.func = wrapped


def _write_profiler_outputs(prof: torch.profiler.profile, output_dir: Path, row_limit: int) -> dict[str, str]:
    chrome_trace = output_dir / "trace.json"
    table_path = output_dir / "key_averages_cuda.txt"
    cpu_table_path = output_dir / "key_averages_cpu.txt"
    stacks_path = output_dir / "stacks_cuda.txt"

    prof.export_chrome_trace(str(chrome_trace))
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=row_limit)
    table_path.write_text(table)
    cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit)
    cpu_table_path.write_text(cpu_table)
    try:
        prof.export_stacks(str(stacks_path), metric="self_cuda_time_total")
    except Exception as exc:
        stacks_path.write_text(f"stack export unavailable: {exc}\n")
    return {
        "chrome_trace": str(chrome_trace),
        "key_averages_cuda": str(table_path),
        "key_averages_cpu": str(cpu_table_path),
        "stacks": str(stacks_path),
    }


def main() -> None:
    args = parse_args()
    configure_torch_backends()
    if args.instrument_hot_paths:
        _install_hot_path_labels()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_file = args.checkpoint_file
    if args.agent == "trained" and checkpoint_file is None:
        checkpoint_file = _latest_checkpoint(args.checkpoint_root)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = args.output_root / f"{timestamp}_{args.agent}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(TASK_ID, play=False)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    _configure_walk_command(env_cfg, args)
    _configure_human_gait(env_cfg, args)
    agent_cfg = load_rl_cfg(TASK_ID)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if args.instrument_hot_paths:
        _wrap_manager_terms(raw_env)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    obs, _ = env.reset()
    policy = _build_policy(args.agent, env, agent_cfg, checkpoint_file, device)

    obs = _run_steps(env, policy, obs, args.warmup_steps)
    _sync_if_cuda(device)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.profiler._utils._init_for_cuda_graphs()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    start_event = end_event = None
    if device.startswith("cuda") and torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_modules=True,
    ) as prof:
        with torch.profiler.record_function("latent_z/profile_window"):
            obs = _run_steps(env, policy, obs, args.active_steps)
        prof.step()

    if end_event is not None:
        end_event.record()
    _sync_if_cuda(device)

    wall_time_s = None
    if start_event is not None and end_event is not None:
        wall_time_s = float(start_event.elapsed_time(end_event) / 1000.0)

    paths = _write_profiler_outputs(prof, output_dir, args.row_limit)
    samples = args.num_envs * args.active_steps
    summary = {
        "task_id": TASK_ID,
        "agent": args.agent,
        "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else None,
        "device": device,
        "num_envs": args.num_envs,
        "warmup_steps": args.warmup_steps,
        "active_steps": args.active_steps,
        "samples": samples,
        "cuda_timed_window_s": wall_time_s,
        "samples_per_second": samples / wall_time_s if wall_time_s else None,
        "steps_per_second": args.active_steps / wall_time_s if wall_time_s else None,
        "profile_memory": args.profile_memory,
        "record_shapes": args.record_shapes,
        "with_stack": args.with_stack,
        "instrument_hot_paths": args.instrument_hot_paths,
        "outputs": paths,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    env.close()

    print(f"[INFO] Wrote profile outputs to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
