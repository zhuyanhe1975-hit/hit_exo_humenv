#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path(".omx/kimodo_stair_pseudomocap/kimodo_smplx_stairs_up_5s_d40.npz")
DEFAULT_OUTPUT = Path(".omx/kimodo_stair_pseudomocap/kimodo_smplx_stairs_up_5s_foot_constraints.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Kimodo root + foot end-effector stair constraints from a first-pass Kimodo motion.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--step-height", type=float, default=0.135)
    parser.add_argument("--tread-depth", type=float, default=0.22)
    parser.add_argument("--min-segment-frames", type=int, default=5)
    parser.add_argument("--max-frames-per-segment", type=int, default=18)
    return parser.parse_args()


def matrix_to_axis_angle_np(rot_mats: np.ndarray) -> np.ndarray:
    from kimodo.geometry import matrix_to_axis_angle
    import torch

    tensor = torch.as_tensor(rot_mats, dtype=torch.float32)
    return matrix_to_axis_angle(tensor).cpu().numpy()


def smooth_labels(labels: np.ndarray, min_segment_frames: int) -> np.ndarray:
    labels = labels.copy()
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        if end - start < min_segment_frames:
            left = labels[start - 1] if start > 0 else None
            right = labels[end] if end < len(labels) else None
            fill = left if left is not None else right
            if fill is not None:
                labels[start:end] = fill
        start = end
    return labels


def stance_labels(foot_contacts: np.ndarray, joints: np.ndarray) -> np.ndarray:
    left_contact = foot_contacts[:, 0] | foot_contacts[:, 1]
    right_contact = foot_contacts[:, 2] | foot_contacts[:, 3]
    left = joints[:, 10]
    right = joints[:, 11]
    left_speed = np.r_[0.0, np.linalg.norm(np.diff(left[:, [0, 2]], axis=0), axis=1)]
    right_speed = np.r_[0.0, np.linalg.norm(np.diff(right[:, [0, 2]], axis=0), axis=1)]
    labels = np.where(left_speed <= right_speed, 0, 1).astype(np.int32)
    labels[left_contact & ~right_contact] = 0
    labels[right_contact & ~left_contact] = 1
    return labels


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


def sample_segment_frames(start: int, end: int, max_frames: int) -> np.ndarray:
    count = end - start
    if count <= max_frames:
        return np.arange(start, end, dtype=np.int64)
    return np.unique(np.linspace(start, end - 1, max_frames, dtype=np.int64))


def stair_height_for_forward(forward: float, steps: int, step_height: float, tread_depth: float) -> float:
    idx = int(np.clip(np.floor(forward / max(tread_depth, 1e-6) + 0.5), 0, steps))
    return idx * step_height


def build_constraints(args: argparse.Namespace) -> tuple[list[dict], dict]:
    with np.load(args.source, allow_pickle=True) as data:
        local_rot = np.asarray(data["local_rot_mats"], dtype=np.float32)
        root = np.asarray(data["root_positions"], dtype=np.float32)
        smooth_root = np.asarray(data["smooth_root_pos"], dtype=np.float32)
        joints = np.asarray(data["posed_joints"], dtype=np.float32)
        contacts = np.asarray(data["foot_contacts"], dtype=bool)

    frame_count = len(root)
    run = args.steps * args.tread_depth
    forward = np.linspace(0.0, run, frame_count, dtype=np.float32)
    root_path = np.column_stack([np.zeros(frame_count, dtype=np.float32), forward])
    heading = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (frame_count, 1))
    constraints = [
        {
            "type": "root2d",
            "frame_indices": list(range(frame_count)),
            "smooth_root_2d": root_path.tolist(),
            "global_root_heading": heading.tolist(),
        }
    ]

    labels = smooth_labels(stance_labels(contacts, joints), args.min_segment_frames)
    local_axis = matrix_to_axis_angle_np(local_rot)
    foot_joint_idx = {0: 10, 1: 11}
    foot_type = {0: "left-foot", 1: "right-foot"}
    by_foot: dict[int, list[int]] = {0: [], 1: []}
    root_adjusted = {0: [], 1: []}
    local_adjusted = {0: [], 1: []}
    smooth_adjusted = {0: [], 1: []}

    for start, end, label in segment_bounds(labels):
        frames = sample_segment_frames(start, end, args.max_frames_per_segment)
        foot = joints[start:end, foot_joint_idx[label]]
        foot_target = np.median(foot, axis=0)
        foot_target[2] = float(np.median(forward[start:end]))
        foot_target[1] = root[0, 1] - 0.9 + stair_height_for_forward(
            float(foot_target[2]), args.steps, args.step_height, args.tread_depth
        )
        for frame in frames:
            current = joints[frame, foot_joint_idx[label]]
            delta = foot_target - current
            shifted_root = root[frame] + delta
            shifted_root[2] = forward[frame]
            by_foot[label].append(int(frame))
            root_adjusted[label].append(shifted_root.tolist())
            smooth_adjusted[label].append([float(shifted_root[0]), float(shifted_root[2])])
            local_adjusted[label].append(local_axis[frame].tolist())

    for label in (0, 1):
        if by_foot[label]:
            constraints.append(
                {
                    "type": foot_type[label],
                    "frame_indices": by_foot[label],
                    "local_joints_rot": local_adjusted[label],
                    "root_positions": root_adjusted[label],
                    "smooth_root_2d": smooth_adjusted[label],
                }
            )

    meta = {
        "segments": [
            {"start": start, "end": end, "foot": "left" if label == 0 else "right"}
            for start, end, label in segment_bounds(labels)
        ],
        "left_constraint_frames": len(by_foot[0]),
        "right_constraint_frames": len(by_foot[1]),
    }
    return constraints, meta


def main() -> None:
    args = parse_args()
    constraints, meta = build_constraints(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(constraints, indent=2) + "\n")
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote: {args.output}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
