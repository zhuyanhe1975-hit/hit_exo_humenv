#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(".omx/kimodo_stair_pseudomocap/kimodo_smplx_stairs_up_d20.hdf5")
DEFAULT_OUTPUT = Path(".omx/kimodo_stair_pseudomocap/kimodo_smplx_stairs_up_d20_footlocked.hdf5")
DEFAULT_ROBOT_XML = Path("/home/yhzhu/AI/humenv/humenv/assets/robot.xml")
DEFAULT_CONVERT = Path("/home/yhzhu/AI/humenv/scripts/convert_amass_smplsim_motion.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce support-foot sliding in a HumEnv HDF5 by root XY correction.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--convert-script", type=Path, default=DEFAULT_CONVERT)
    parser.add_argument("--min-segment-frames", type=int, default=6)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--max-correction-step", type=float, default=0.025)
    return parser.parse_args()


def load_motion(path: Path, episode: str) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    import h5py

    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float64)
        qvel = np.asarray(ep["qvel"][:], dtype=np.float64)
        file_attrs = {key: hf.attrs[key] for key in hf.attrs}
        ep_attrs = {key: ep.attrs[key] for key in ep.attrs}
    return qpos, qvel, file_attrs, ep_attrs


def foot_positions(qpos: np.ndarray, robot_xml: Path) -> dict[str, np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    body_ids = {
        "L_Toe": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Toe"),
        "R_Toe": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_Toe"),
    }
    out = {name: np.zeros((len(qpos), 3), dtype=np.float64) for name in body_ids}
    for idx, pos in enumerate(qpos):
        data.qpos[:] = pos
        mujoco.mj_forward(model, data)
        for name, body_id in body_ids.items():
            out[name][idx] = data.xpos[body_id]
    return out


def smooth_labels(labels: np.ndarray, min_segment_frames: int) -> np.ndarray:
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels) and labels[end] == labels[start]:
                end += 1
            if end - start < min_segment_frames:
                left = labels[start - 1] if start > 0 else None
                right = labels[end] if end < len(labels) else None
                if left is not None and right is not None and left == right:
                    labels[start:end] = left
                    changed = True
                elif left is not None:
                    labels[start:end] = left
                    changed = True
                elif right is not None:
                    labels[start:end] = right
                    changed = True
            start = end
    return labels


def stance_labels(feet: dict[str, np.ndarray], min_segment_frames: int) -> np.ndarray:
    left = feet["L_Toe"]
    right = feet["R_Toe"]
    left_speed = np.r_[0.0, np.linalg.norm(np.diff(left[:, :2], axis=0), axis=1)]
    right_speed = np.r_[0.0, np.linalg.norm(np.diff(right[:, :2], axis=0), axis=1)]
    left_score = left_speed + 0.35 * (left[:, 2] - np.minimum(left[:, 2], right[:, 2]))
    right_score = right_speed + 0.35 * (right[:, 2] - np.minimum(left[:, 2], right[:, 2]))
    labels = np.where(left_score <= right_score, 0, 1).astype(np.int32)
    return smooth_labels(labels, min_segment_frames)


def segment_bounds(labels: np.ndarray) -> list[tuple[int, int, int]]:
    segments = []
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        segments.append((start, end, int(labels[start])))
        start = end
    return segments


def smooth_offsets(offset: np.ndarray, window: int, max_step: float) -> np.ndarray:
    if window > 1 and len(offset) >= 3:
        window = min(window, len(offset) if len(offset) % 2 == 1 else len(offset) - 1)
        if window >= 3:
            kernel = np.ones(window, dtype=np.float64) / window
            pad = window // 2
            for col in range(offset.shape[1]):
                padded = np.pad(offset[:, col], (pad, pad), mode="edge")
                offset[:, col] = np.convolve(padded, kernel, mode="valid")
    if max_step > 0.0:
        limited = offset.copy()
        for idx in range(1, len(limited)):
            delta = limited[idx] - limited[idx - 1]
            norm = float(np.linalg.norm(delta))
            if norm > max_step:
                limited[idx] = limited[idx - 1] + delta * (max_step / norm)
        return limited
    return offset


def lock_support_feet(qpos: np.ndarray, feet: dict[str, np.ndarray], labels: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    qpos_locked = qpos.copy()
    offset = np.zeros((len(qpos), 2), dtype=np.float64)
    segments = segment_bounds(labels)
    names = {0: "L_Toe", 1: "R_Toe"}
    for start, end, label in segments:
        foot_xy = feet[names[label]][start:end, :2]
        target = np.median(foot_xy, axis=0)
        offset[start:end] = target - foot_xy
    offset = smooth_offsets(offset, args.smooth_window, args.max_correction_step)
    qpos_locked[:, :2] += offset
    meta = {
        "segments": [
            {"start": start, "end": end, "foot": names[label]}
            for start, end, label in segments
        ],
        "max_xy_correction": float(np.linalg.norm(offset, axis=1).max()),
        "mean_xy_correction": float(np.linalg.norm(offset, axis=1).mean()),
    }
    return qpos_locked, meta


def import_converter(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("humenv_amass_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_hdf5(
    output: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    obs: np.ndarray,
    file_attrs: dict[str, object],
    ep_attrs: dict[str, object],
    meta: dict[str, object],
    source: Path,
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
        hf.attrs["conversion"] = "support_foot_root_xy_lock"
        hf.attrs["support_lock_source_conversion"] = str(file_attrs.get("conversion", "unknown"))
        hf.attrs["support_lock_max_xy_correction"] = meta["max_xy_correction"]
        hf.attrs["support_lock_mean_xy_correction"] = meta["mean_xy_correction"]
        ep = hf.create_group("ep_0")
        ep.attrs["length"] = len(qpos)
        ep.attrs["motion_id"] = motion_id_value
        ep.create_dataset("qpos", data=qpos.astype(np.float32), compression="gzip")
        ep.create_dataset("qvel", data=qvel.astype(np.float32), compression="gzip")
        ep.create_dataset("observation", data=obs.astype(np.float32), compression="gzip")
        ep.create_dataset("motion_id", data=motion_id, compression="gzip")
        ep.create_dataset("terminated", data=terminated, compression="gzip")
        ep.create_dataset("truncated", data=truncated, compression="gzip")


def sliding_metric(feet: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    total = []
    for start, end, label in segment_bounds(labels):
        name = "L_Toe" if label == 0 else "R_Toe"
        xy = feet[name][start:end, :2]
        if len(xy) > 1:
            total.append(float(np.linalg.norm(xy[-1] - xy[0])))
    values = np.asarray(total, dtype=np.float64)
    if len(values) == 0:
        return {"mean_segment_slide": 0.0, "max_segment_slide": 0.0}
    return {"mean_segment_slide": float(values.mean()), "max_segment_slide": float(values.max())}


def main() -> None:
    args = parse_args()
    qpos, _qvel, file_attrs, ep_attrs = load_motion(args.input, args.episode)
    before_feet = foot_positions(qpos, args.robot_xml)
    labels = stance_labels(before_feet, args.min_segment_frames)
    before = sliding_metric(before_feet, labels)
    qpos_locked, meta = lock_support_feet(qpos, before_feet, labels, args)

    converter = import_converter(args.convert_script)
    dt = float(file_attrs.get("dt", 1.0 / 30.0))
    qvel = converter.differentiate_qpos(qpos_locked, dt)
    obs = converter.compute_observations(qpos_locked, qvel)
    after_feet = foot_positions(qpos_locked, args.robot_xml)
    after = sliding_metric(after_feet, labels)
    meta.update({"before": before, "after": after})
    write_hdf5(args.output, qpos_locked, qvel, obs, file_attrs, ep_attrs, meta, args.input)
    args.output.with_suffix(".support_lock.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote: {args.output}")
    print(f"segments: {len(meta['segments'])}")
    print(f"support slide before mean/max: {before['mean_segment_slide']:.4f}/{before['max_segment_slide']:.4f} m")
    print(f"support slide after  mean/max: {after['mean_segment_slide']:.4f}/{after['max_segment_slide']:.4f} m")
    print(f"root correction mean/max: {meta['mean_xy_correction']:.4f}/{meta['max_xy_correction']:.4f} m")
    print(
        "preview: "
        f"MOCAP_MOTION={args.output} STAIR_STEPS=6 STAIR_HEIGHT=0.135 STAIR_WIDTH=1.4 "
        "./show_mocap_track_updown_stairs.sh"
    )


if __name__ == "__main__":
    main()
