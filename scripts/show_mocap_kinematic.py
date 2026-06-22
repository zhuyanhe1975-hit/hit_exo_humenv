#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from humenv import make_humenv

from run_s1_mocap_track_visual import build_mimic_matched_stair_xml, build_stair_xml, build_support_xml
from train_mjlab_knee_exo_mocap_track import DEFAULT_MOTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay HumEnv mocap qpos kinematically in the MuJoCo viewer. "
            "This does not use S-1, RL, actions, or dynamics stepping."
        )
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
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
    parser.add_argument("--terrain", choices=("flat", "stairs", "supports", "mimic-stairs"), default="flat")
    parser.add_argument("--stair-width", type=float, default=1.4)
    parser.add_argument("--stair-height", type=float, default=0.135)
    parser.add_argument("--stair-steps", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(".omx/mocap_kinematic_visual"))
    return parser.parse_args()


def load_motion(path: Path, episode: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with h5py.File(path, "r") as hf:
        if episode not in hf:
            raise KeyError(f"{path} does not contain episode '{episode}'")
        ep = hf[episode]
        missing = [key for key in ("qpos", "qvel") if key not in ep]
        if missing:
            raise KeyError(f"{path}:{episode} is missing required datasets: {', '.join(missing)}")
        motion = {
            "qpos": np.asarray(ep["qpos"][:], dtype=np.float64),
            "qvel": np.asarray(ep["qvel"][:], dtype=np.float64),
        }
        attrs = {key: ep.attrs[key] for key in ep.attrs.keys()}
        attrs.update({f"file:{key}": hf.attrs[key] for key in hf.attrs.keys()})
    return motion, attrs


def repeat_motion(motion: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    if count <= 0:
        raise ValueError("--repeat must be positive")
    if count == 1:
        return motion

    qpos_parts = []
    qvel_parts = []
    root_delta = motion["qpos"][-1, :3] - motion["qpos"][0, :3]
    root_delta[2] = 0.0
    for cycle in range(count):
        qpos = motion["qpos"].copy()
        qpos[:, :3] += cycle * root_delta
        qpos_parts.append(qpos)
        qvel_parts.append(motion["qvel"])
    return {
        "qpos": np.concatenate(qpos_parts, axis=0),
        "qvel": np.concatenate(qvel_parts, axis=0),
    }


def frame_bounds(args: argparse.Namespace, frame_count: int, fps: float) -> tuple[int, int]:
    start = max(0, min(args.start_frame, frame_count - 1))
    if args.duration > 0:
        duration_end = start + max(1, int(args.duration * fps))
    else:
        duration_end = frame_count
    requested_end = duration_end if args.end_frame is None else args.end_frame
    end = max(start + 1, min(requested_end, frame_count))
    return start, end


def configure_camera(viewer, qpos: np.ndarray) -> None:
    root = qpos[:, :3]
    lo = root.min(axis=0)
    hi = root.max(axis=0)
    center = (lo + hi) / 2.0
    span_xy = float(np.linalg.norm(hi[:2] - lo[:2]))
    span_z = float(max(0.2, hi[2] - lo[2]))

    viewer.cam.lookat[:] = center
    viewer.cam.lookat[2] += 0.25 * span_z
    viewer.cam.distance = max(3.0, span_xy * 1.4 + span_z * 2.0)
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 135


def print_summary(
    motion: dict[str, np.ndarray],
    attrs: dict[str, object],
    path: Path,
    episode: str,
    start: int,
    end: int,
    fps: float,
    speed: float,
) -> None:
    qpos = motion["qpos"]
    root = qpos[start:end, :3]
    delta = root[-1] - root[0]
    duration = max((end - start - 1) / fps, 1.0 / fps)
    xy_speed = float(np.linalg.norm(delta[:2]) / duration)
    print(f"[INFO] Kinematic mocap replay: {path}:{episode}")
    if "file:source_motion" in attrs:
        print(f"[INFO] Source motion: {attrs['file:source_motion']}")
    if "file:speedup" in attrs:
        print(f"[INFO] Source speedup: {attrs['file:speedup']}")
    print(
        "[INFO] Motion summary: "
        f"frames={len(qpos)} playing={start}:{end} fps={fps:g} speed={speed:g} "
        f"root_delta=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f})m "
        f"xy_speed={xy_speed:.3f}m/s z_range=({root[:, 2].min():.3f}, {root[:, 2].max():.3f})"
    )
    print("[INFO] No S-1, no RL policy, no actions, no physics stepping; directly setting qpos/qvel per frame.")


def main() -> None:
    args = parse_args()
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if not args.motion.exists():
        raise FileNotFoundError(f"Motion file not found: {args.motion}")

    motion, attrs = load_motion(args.motion, args.episode)
    motion = repeat_motion(motion, args.repeat)
    fps = args.fps or 1.0 / float(attrs.get("dt", attrs.get("file:dt", 1.0 / 30.0)))
    start, end = frame_bounds(args, len(motion["qpos"]), fps)
    frame_delay = 1.0 / (fps * args.speed)
    print_summary(motion, attrs, args.motion, args.episode, start, end, fps, args.speed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env_xml = None
    if args.terrain == "stairs":
        env_xml = build_stair_xml(
            motion,
            args.out_dir,
            stair_width=args.stair_width,
            nominal_step_height=args.stair_height,
            requested_steps=args.stair_steps,
        )
    elif args.terrain == "supports":
        env_xml = build_support_xml(args.motion, args.out_dir)
    elif args.terrain == "mimic-stairs":
        env_xml = build_mimic_matched_stair_xml(
            motion,
            args.out_dir,
            stair_width=args.stair_width,
        )

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
        env.unwrapped.set_physics(qpos=motion["qpos"][start], qvel=motion["qvel"][start])
        if not args.headless:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
            configure_camera(viewer, motion["qpos"][start:end])

        while True:
            for frame in range(start, end):
                env.unwrapped.set_physics(qpos=motion["qpos"][frame], qvel=motion["qvel"][frame])
                if args.print_every > 0 and (frame == start or (frame - start) % args.print_every == 0):
                    root = env.unwrapped.data.qpos[:3]
                    print(f"[INFO] frame={frame} root=({root[0]:.3f}, {root[1]:.3f}, {root[2]:.3f})")
                if viewer is not None:
                    viewer.sync()
                    if not viewer.is_running():
                        os._exit(0)
                time.sleep(frame_delay)
            if not args.loop:
                break
        if viewer is not None:
            os._exit(0)
    finally:
        # MuJoCo's passive GLFW viewer can segfault in native shutdown on some
        # driver stacks. Let process teardown close it after releasing HumEnv.
        env.close()


if __name__ == "__main__":
    main()
