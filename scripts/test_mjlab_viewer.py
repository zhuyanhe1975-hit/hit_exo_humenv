#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

import hit_exo_humenv.mjlab  # noqa: F401
from hit_exo_humenv.latent_z_config import cfg_path
from hit_exo_humenv.mjlab.walking_env_cfg import TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer.native.viewer import NativeMujocoViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the mjlab RL env with frozen S-1 human control and random knee-assist actions.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=0.1)
    parser.add_argument("--s1-samples", type=int, default=cfg_path("human_s1", "num_samples_per_inference"))
    parser.add_argument("--s1-workers", type=int, default=cfg_path("human_s1", "max_workers"))
    return parser.parse_args()


class RandomKneeAssistPolicy:
    def __init__(self, action_shape: tuple[int, ...], device: str, scale: float) -> None:
        self.action_shape = action_shape
        self.device = device
        self.scale = float(scale)

    def __call__(self, obs) -> torch.Tensor:
        del obs
        return self.scale * (2.0 * torch.rand(self.action_shape, device=self.device) - 1.0)


def main() -> None:
    args = parse_args()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.actions["human_s1"].num_samples_per_inference = args.s1_samples
    env_cfg.actions["human_s1"].max_workers = args.s1_workers

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = RandomKneeAssistPolicy(env.unwrapped.action_space.shape, env.unwrapped.device, args.action_scale)
    NativeMujocoViewer(env, policy).run()
    env.close()


if __name__ == "__main__":
    main()
