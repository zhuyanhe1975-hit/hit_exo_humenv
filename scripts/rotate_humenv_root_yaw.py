#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DEFAULT_INPUT = Path("/home/yhzhu/AI/humenv/data_preparation/humenv_amass_terrain_fixed/stairs_up.hdf5")
DEFAULT_OUTPUT = Path(".omx/mimic_stairs_up_heading_fixed.hdf5")
DEFAULT_CONVERT = Path("/home/yhzhu/AI/humenv/scripts/convert_amass_smplsim_motion.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a constant world-yaw correction to HumEnv root quaternions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument("--yaw-deg", type=float, default=90.0)
    parser.add_argument("--convert-script", type=Path, default=DEFAULT_CONVERT)
    return parser.parse_args()


def load_motion(path: Path, episode: str) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    import h5py

    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float64)
        file_attrs = {key: hf.attrs[key] for key in hf.attrs}
        ep_attrs = {key: ep.attrs[key] for key in ep.attrs}
    return qpos, file_attrs, ep_attrs


def import_converter(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("humenv_amass_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotate_root_quats(qpos: np.ndarray, yaw_deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    out = qpos.copy()
    yaw = R.from_euler("z", yaw_deg, degrees=True)
    quat_wxyz = out[:, 3:7]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rotated = (yaw * R.from_quat(quat_xyzw)).as_quat()
    out[:, 3:7] = rotated[:, [3, 0, 1, 2]]
    return out


def write_hdf5(
    output: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    obs: np.ndarray,
    file_attrs: dict[str, object],
    ep_attrs: dict[str, object],
    source: Path,
    yaw_deg: float,
) -> None:
    import h5py

    output.parent.mkdir(parents=True, exist_ok=True)
    motion_id_value = int(ep_attrs.get("motion_id", -1))
    motion_id = np.full((len(qpos), 1), motion_id_value, dtype=np.int64)
    terminated = np.zeros((len(qpos), 1), dtype=bool)
    truncated = np.zeros((len(qpos), 1), dtype=bool)
    truncated[-1] = True
    with h5py.File(output, "w") as hf:
        for key, value in file_attrs.items():
            hf.attrs[key] = value
        hf.attrs["source_motion"] = str(source)
        hf.attrs["conversion"] = "root_yaw_corrected"
        hf.attrs["root_yaw_correction_deg"] = yaw_deg
        ep = hf.create_group("ep_0")
        ep.attrs["length"] = len(qpos)
        ep.attrs["motion_id"] = motion_id_value
        ep.create_dataset("qpos", data=qpos.astype(np.float32), compression="gzip")
        ep.create_dataset("qvel", data=qvel.astype(np.float32), compression="gzip")
        ep.create_dataset("observation", data=obs.astype(np.float32), compression="gzip")
        ep.create_dataset("motion_id", data=motion_id, compression="gzip")
        ep.create_dataset("terminated", data=terminated, compression="gzip")
        ep.create_dataset("truncated", data=truncated, compression="gzip")


def main() -> None:
    args = parse_args()
    qpos, file_attrs, ep_attrs = load_motion(args.input, args.episode)
    qpos = rotate_root_quats(qpos, args.yaw_deg)
    converter = import_converter(args.convert_script)
    dt = float(file_attrs.get("dt", 1.0 / 30.0))
    qvel = converter.differentiate_qpos(qpos, dt)
    obs = converter.compute_observations(qpos, qvel)
    write_hdf5(args.output, qpos, qvel, obs, file_attrs, ep_attrs, args.input, args.yaw_deg)
    print(f"wrote: {args.output}")
    print(f"frames: {len(qpos)} yaw_correction={args.yaw_deg:g}deg")
    print(f"root_start: {qpos[0, :3].tolist()}")
    print(f"root_end: {qpos[-1, :3].tolist()}")


if __name__ == "__main__":
    main()
