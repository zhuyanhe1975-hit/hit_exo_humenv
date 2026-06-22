#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from humenv import make_humenv

from show_mocap_kinematic import (
    build_mimic_matched_stair_xml,
    build_stair_xml,
    build_support_xml,
    configure_camera,
    frame_bounds,
    load_motion,
    print_summary,
    repeat_motion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay multiple HumEnv mocap clips in one persistent MuJoCo viewer."
    )
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--pause-between", type=float, default=0.75)
    parser.add_argument("--terrain", choices=("flat", "stairs", "supports", "mimic-stairs"), default="stairs")
    parser.add_argument("--stair-width", type=float, default=1.4)
    parser.add_argument("--stair-height", type=float, default=0.135)
    parser.add_argument("--stair-steps", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(".omx/mocap_playlist_kinematic"))
    return parser.parse_args()


def playable_clip(args: argparse.Namespace, path: Path) -> tuple[dict[str, np.ndarray], dict[str, object], float, int, int]:
    motion, attrs = load_motion(path, args.episode)
    motion = repeat_motion(motion, args.repeat)
    fps = args.fps or 1.0 / float(attrs.get("dt", attrs.get("file:dt", 1.0 / 30.0)))
    start, end = frame_bounds(args, len(motion["qpos"]), fps)
    return motion, attrs, fps, start, end


def build_env_xml(args: argparse.Namespace, first_motion: dict[str, np.ndarray], first_path: Path) -> Path | None:
    if args.terrain == "flat":
        return None
    if args.terrain == "stairs":
        return build_stair_xml(
            first_motion,
            args.out_dir,
            stair_width=args.stair_width,
            nominal_step_height=args.stair_height,
            requested_steps=args.stair_steps,
        )
    if args.terrain == "supports":
        return build_support_xml(first_path, args.out_dir)
    if args.terrain == "mimic-stairs":
        return build_mimic_matched_stair_xml(
            first_motion,
            args.out_dir,
            stair_width=args.stair_width,
        )
    raise ValueError(f"Unsupported terrain: {args.terrain}")


def main() -> None:
    args = parse_args()
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if args.pause_between < 0:
        raise ValueError("--pause-between must be non-negative")

    labels = list(args.label)
    while len(labels) < len(args.motion):
        labels.append(args.motion[len(labels)].stem)

    clips = []
    for path, label in zip(args.motion, labels):
        if not path.exists():
            raise FileNotFoundError(f"Motion file not found: {path}")
        motion, attrs, fps, start, end = playable_clip(args, path)
        clips.append((path, label, motion, attrs, fps, start, end))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env_xml = build_env_xml(args, clips[0][2], clips[0][0])
    env, _ = make_humenv(
        num_envs=1,
        task="zero",
        state_init="Default",
        render_mode=None,
        max_episode_steps=100000,
        **({"xml": str(env_xml)} if env_xml is not None else {}),
    )

    viewer = None
    try:
        first_motion = clips[0][2]
        first_start = clips[0][5]
        env.unwrapped.set_physics(qpos=first_motion["qpos"][first_start], qvel=first_motion["qvel"][first_start])
        if not args.headless:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
            all_qpos = np.concatenate([clip[2]["qpos"][clip[5] : clip[6]] for clip in clips], axis=0)
            configure_camera(viewer, all_qpos)

        print(f"[INFO] Persistent mocap playlist viewer: clips={len(clips)} terrain={args.terrain}")
        while True:
            for clip_idx, (path, label, motion, attrs, fps, start, end) in enumerate(clips, start=1):
                frame_delay = 1.0 / (fps * args.speed)
                print(f"\n[INFO] Clip {clip_idx}/{len(clips)}: {label}")
                print_summary(motion, attrs, path, args.episode, start, end, fps, args.speed)

                for frame in range(start, end):
                    env.unwrapped.set_physics(qpos=motion["qpos"][frame], qvel=motion["qvel"][frame])
                    if args.print_every > 0 and (frame == start or (frame - start) % args.print_every == 0):
                        root = env.unwrapped.data.qpos[:3]
                        print(f"[INFO] {label} frame={frame} root=({root[0]:.3f}, {root[1]:.3f}, {root[2]:.3f})")
                    if viewer is not None:
                        viewer.sync()
                        if not viewer.is_running():
                            os._exit(0)
                    time.sleep(frame_delay)

                if args.pause_between > 0:
                    deadline = time.time() + args.pause_between
                    while time.time() < deadline:
                        if viewer is not None:
                            viewer.sync()
                            if not viewer.is_running():
                                os._exit(0)
                        time.sleep(min(0.05, deadline - time.time()))
            if not args.loop:
                break
        if viewer is not None:
            os._exit(0)
    finally:
        # Avoid explicit passive-viewer close; on some GLFW/MuJoCo stacks that
        # native shutdown path segfaults after the final frame.
        env.close()


if __name__ == "__main__":
    main()
