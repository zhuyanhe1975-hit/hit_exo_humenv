#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_KIMODO_CSV = Path(".omx/kimodo_stair_pseudomocap/kimodo_g1_stairs_up_d20.csv")
DEFAULT_TEMPLATE = Path(
    "/home/yhzhu/AI/humenv/data_preparation/humenv_from_protomotions/"
    "KIT_167_upstairs_downstairs01_poses_upx_upstairs.hdf5"
)
DEFAULT_OUTPUT = Path(".omx/kimodo_stair_pseudomocap/kimodo_g1_root_humenv_stairs_up_d20.hdf5")
DEFAULT_ROBOT_XML = Path("/home/yhzhu/AI/humenv/humenv/assets/robot.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a HumEnv pseudo-mocap HDF5 by combining Kimodo-G1 stair root timing "
            "with an existing HumEnv humanoid pose template."
        )
    )
    parser.add_argument("--kimodo-csv", type=Path, default=DEFAULT_KIMODO_CSV)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direction", choices=("up", "down"), default="up")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--step-height", type=float, default=0.135)
    parser.add_argument("--tread-depth", type=float, default=0.22)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--motion-id", type=int, default=-1)
    parser.add_argument("--bob-scale", type=float, default=0.35)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    return parser.parse_args()


def load_kimodo_root(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected a qpos CSV with at least 3 columns, got {data.shape}")
    return data[:, :3]


def load_template(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import h5py

    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as hf:
        ep = hf["ep_0"]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float64)
        qvel = np.asarray(ep["qvel"][:], dtype=np.float64)
        attrs = {key: hf.attrs[key] for key in hf.attrs}
    return qpos, qvel, attrs


def _linear_resample(values: np.ndarray, count: int) -> np.ndarray:
    src_t = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, count, dtype=np.float64)
    out = np.empty((count, values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(dst_t, src_t, values[:, col])
    return out


def _resample_quat_xyzw_to_wxyz(quat_wxyz: np.ndarray, count: int) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation, Slerp

        src_t = np.linspace(0.0, 1.0, len(quat_wxyz), dtype=np.float64)
        dst_t = np.linspace(0.0, 1.0, count, dtype=np.float64)
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        rotations = Rotation.from_quat(quat_xyzw)
        resampled = Slerp(src_t, rotations)(dst_t).as_quat()
        return resampled[:, [3, 0, 1, 2]]
    except Exception:
        out = _linear_resample(quat_wxyz, count)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-8)


def resample_qpos(qpos: np.ndarray, count: int) -> np.ndarray:
    out = _linear_resample(qpos, count)
    out[:, 3:7] = _resample_quat_xyzw_to_wxyz(qpos[:, 3:7], count)
    return out


def progress_from_root(root: np.ndarray) -> np.ndarray:
    xy = root[:, :2]
    delta = xy[-1] - xy[0]
    distance = float(np.linalg.norm(delta))
    if distance < 1e-8:
        return np.linspace(0.0, 1.0, len(root), dtype=np.float64)
    direction = delta / distance
    progress = ((xy - xy[0]) @ direction) / distance
    progress -= float(progress[0])
    denom = float(progress[-1] - progress[0])
    if abs(denom) < 1e-8:
        return np.linspace(0.0, 1.0, len(root), dtype=np.float64)
    return np.clip(progress / denom, 0.0, 1.0)


def build_pseudo_qpos(
    kimodo_root: np.ndarray,
    template_qpos: np.ndarray,
    direction: str,
    steps: int,
    step_height: float,
    tread_depth: float,
    bob_scale: float,
) -> np.ndarray:
    count = len(kimodo_root)
    qpos = resample_qpos(template_qpos, count)
    progress = progress_from_root(kimodo_root)
    if direction == "down":
        stair_progress = 1.0 - progress
    else:
        stair_progress = progress

    run = steps * tread_depth
    total_height = steps * step_height
    qpos[:, 0] = progress * run
    qpos[:, 1] = 0.0

    original_rise = np.linspace(qpos[0, 2], qpos[-1, 2], count, dtype=np.float64)
    vertical_bob = qpos[:, 2] - original_rise
    qpos[:, 2] = template_qpos[0, 2] + stair_progress * total_height + bob_scale * vertical_bob
    return qpos


def differentiate_qpos(qpos: np.ndarray, dt: float, robot_xml: Path) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    for idx in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(model, qvel[idx], dt, qpos[idx], qpos[idx + 1])
    if len(qvel) > 1:
        qvel[-1] = qvel[-2]
    return qvel


def compute_observations(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    from gymnasium.wrappers import FlattenObservation
    from humenv import make_humenv

    env, _ = make_humenv(
        num_envs=1,
        task="zero",
        state_init="Default",
        wrappers=[FlattenObservation],
        render_mode=None,
        max_episode_steps=100000,
    )
    observations = []
    try:
        for pos, vel in zip(qpos, qvel):
            env.unwrapped.set_physics(qpos=pos, qvel=vel)
            observations.append(env.unwrapped.get_obs()["proprio"].copy())
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32)


def write_hdf5(
    output: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    obs: np.ndarray,
    args: argparse.Namespace,
    template_attrs: dict[str, object],
) -> None:
    import h5py

    output.parent.mkdir(parents=True, exist_ok=True)
    motion_id = np.full((len(qpos), 1), args.motion_id, dtype=np.int64)
    terminated = np.zeros((len(qpos), 1), dtype=bool)
    truncated = np.zeros((len(qpos), 1), dtype=bool)
    truncated[-1] = True

    with h5py.File(output, "w") as hf:
        hf.attrs["num_episodes"] = 1
        hf.attrs["dt"] = 1.0 / args.fps
        hf.attrs["source_format"] = "Kimodo-G1 root + HumEnv pose-template pseudo mocap"
        hf.attrs["conversion"] = "kimodo_g1_root_to_humenv_template"
        hf.attrs["kimodo_csv"] = str(args.kimodo_csv)
        hf.attrs["template_hdf5"] = str(args.template)
        hf.attrs["template_conversion"] = str(template_attrs.get("conversion", "unknown"))
        hf.attrs["direction"] = args.direction
        hf.attrs["steps"] = args.steps
        hf.attrs["step_height"] = args.step_height
        hf.attrs["tread_depth"] = args.tread_depth
        ep = hf.create_group("ep_0")
        ep.attrs["length"] = len(qpos)
        ep.attrs["motion_id"] = args.motion_id
        ep.create_dataset("qpos", data=qpos.astype(np.float32), compression="gzip")
        ep.create_dataset("qvel", data=qvel.astype(np.float32), compression="gzip")
        ep.create_dataset("observation", data=obs.astype(np.float32), compression="gzip")
        ep.create_dataset("motion_id", data=motion_id, compression="gzip")
        ep.create_dataset("terminated", data=terminated, compression="gzip")
        ep.create_dataset("truncated", data=truncated, compression="gzip")


def write_manifest(args: argparse.Namespace, qpos: np.ndarray) -> None:
    manifest_path = args.output.with_suffix(".manifest.json")
    meta = {
        "hdf5": str(args.output),
        "kimodo_csv": str(args.kimodo_csv),
        "template": str(args.template),
        "frames": int(len(qpos)),
        "fps": args.fps,
        "direction": args.direction,
        "steps": args.steps,
        "step_height": args.step_height,
        "tread_depth": args.tread_depth,
        "root_start": qpos[0, :3].tolist(),
        "root_end": qpos[-1, :3].tolist(),
        "format": "humenv episode HDF5",
        "conversion": "kimodo_g1_root_to_humenv_template",
    }
    manifest_path.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.step_height <= 0.0 or args.tread_depth <= 0.0 or args.fps <= 0.0:
        raise ValueError("--step-height, --tread-depth, and --fps must be positive")

    kimodo_root = load_kimodo_root(args.kimodo_csv)
    template_qpos, _template_qvel, template_attrs = load_template(args.template)
    qpos = build_pseudo_qpos(
        kimodo_root=kimodo_root,
        template_qpos=template_qpos,
        direction=args.direction,
        steps=args.steps,
        step_height=args.step_height,
        tread_depth=args.tread_depth,
        bob_scale=args.bob_scale,
    )
    qvel = differentiate_qpos(qpos, 1.0 / args.fps, args.robot_xml)
    obs = compute_observations(qpos, qvel)
    write_hdf5(args.output, qpos, qvel, obs, args, template_attrs)
    write_manifest(args, qpos)

    print(f"wrote: {args.output}")
    print(f"frames: {len(qpos)} fps={args.fps:g} dt={1.0 / args.fps:.6f}s")
    print(f"root_start: {qpos[0, :3].tolist()}")
    print(f"root_end: {qpos[-1, :3].tolist()}")
    print(f"root_min: {qpos[:, :3].min(axis=0).tolist()}")
    print(f"root_max: {qpos[:, :3].max(axis=0).tolist()}")
    print(
        "preview: "
        f"MOCAP_MOTION={args.output} STAIR_STEPS={args.steps} "
        f"STAIR_HEIGHT={args.step_height:g} STAIR_WIDTH=1.4 "
        "./show_mocap_track_updown_stairs.sh --headless --duration 0.1"
    )


if __name__ == "__main__":
    main()
