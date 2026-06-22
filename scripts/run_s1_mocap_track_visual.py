#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import types
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten

from humenv import make_humenv

from hit_exo_humenv.latent_z_config import cfg_path
from train_mjlab_knee_exo_mocap_track import DEFAULT_MOTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize frozen S-1 mocap tracking in native HumEnv without training or exo."
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--episode", default="ep_0")
    parser.add_argument("--model-id", default="facebook/metamotivo-S-1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path(".omx/s1_mocap_track_visual"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--terrain", choices=("flat", "stairs", "supports", "mimic-stairs"), default="flat")
    parser.add_argument("--stair-width", type=float, default=1.4)
    parser.add_argument("--stair-height", type=float, default=0.135)
    parser.add_argument("--stair-steps", type=int, default=0)
    parser.add_argument(
        "--track-root-xy",
        action="store_true",
        help="Project the simulated floating root XY toward the mocap root XY after each dynamics step.",
    )
    parser.add_argument("--root-xy-track-gain", type=float, default=1.0)
    parser.add_argument("--root-xy-velocity-gain", type=float, default=1.0)
    parser.add_argument(
        "--track-root-z",
        action="store_true",
        help="Project the simulated floating root height toward the mocap root height after each dynamics step.",
    )
    parser.add_argument("--root-z-track-gain", type=float, default=1.0)
    parser.add_argument("--root-z-velocity-gain", type=float, default=1.0)
    parser.add_argument(
        "--track-root-orientation",
        action="store_true",
        help="Blend the simulated floating root quaternion toward the mocap root quaternion after each dynamics step.",
    )
    parser.add_argument("--root-orientation-track-gain", type=float, default=0.5)
    parser.add_argument("--root-angular-velocity-gain", type=float, default=0.5)
    parser.add_argument(
        "--joint-pose-track-gain",
        type=float,
        default=0.0,
        help="Blend non-root joint qpos toward mocap after each step; 0 keeps the frozen S-1 dynamics unassisted.",
    )
    parser.add_argument(
        "--joint-velocity-track-gain",
        type=float,
        default=0.0,
        help="Blend non-root joint qvel toward mocap after each step; usually paired with a small joint pose gain.",
    )
    parser.add_argument(
        "--foot-stance-speed-threshold",
        type=float,
        default=0.18,
        help="Mocap foot XY speed below which a foot is considered planted, in m/s.",
    )
    parser.add_argument(
        "--landing-min-stance-frames",
        type=int,
        default=4,
        help="Minimum contiguous mocap stance frames before a foot segment is treated as a landing/support.",
    )
    return parser.parse_args()


def load_motion(path: Path, episode: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        missing = [key for key in ("qpos", "qvel", "observation") if key not in ep]
        if missing:
            raise KeyError(f"{path}:{episode} is missing required datasets: {', '.join(missing)}")
        return {
            "qpos": np.asarray(ep["qpos"][:], dtype=np.float32),
            "qvel": np.asarray(ep["qvel"][:], dtype=np.float32),
            "observation": np.asarray(ep["observation"][:], dtype=np.float32),
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_status(status_path: Path, status: dict[str, object], **updates: object) -> None:
    status.update(updates)
    status["updated_at"] = _now()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8")


def configure_viewer_camera(viewer, motion: dict[str, np.ndarray]) -> None:
    root = motion["qpos"][:, :3]
    viewer.cam.lookat[:] = (root[0] + root[-1]) / 2
    viewer.cam.lookat[2] += 0.35
    viewer.cam.distance = max(2.5, float(np.linalg.norm(root[-1, :2] - root[0, :2])) * 1.4)
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 135


MOCAP_REFERENCE_RGBA = np.array([0.0, 0.85, 0.15, 0.58], dtype=np.float32)


def _descendant_body_ids(model, root_name: str) -> set[int]:
    import mujoco

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root_id < 0:
        raise ValueError(f"Body '{root_name}' was not found in the MuJoCo model")

    body_ids = {root_id}
    for body_id in range(model.nbody):
        parent = int(model.body_parentid[body_id])
        while parent > 0:
            if parent == root_id:
                body_ids.add(body_id)
                break
            parent = int(model.body_parentid[parent])
    return body_ids


def _humanoid_geom_ids(model) -> list[int]:
    body_ids = _descendant_body_ids(model, "Pelvis")
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    ]


def make_mocap_reference_renderer(xml_path: Path, qpos: np.ndarray):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if len(qpos) != model.nq:
        raise ValueError(
            "Mocap qpos length does not match the reference visual model: "
            f"qpos={len(qpos)} model.nq={model.nq} xml={xml_path}"
        )
    data = mujoco.MjData(model)
    geom_ids = _humanoid_geom_ids(model)
    if not geom_ids:
        raise ValueError(f"No humanoid geoms found under Pelvis in {xml_path}")
    return model, data, geom_ids


def update_mocap_reference_scene(viewer, ref_model, ref_data, geom_ids: list[int], qpos: np.ndarray) -> None:
    import mujoco

    scene = viewer.user_scn
    if scene is None:
        return

    ref_data.qpos[:] = qpos
    ref_data.qvel[:] = 0.0
    mujoco.mj_forward(ref_model, ref_data)

    with viewer.lock():
        scene.ngeom = 0
        for geom_id in geom_ids:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                int(ref_model.geom_type[geom_id]),
                ref_model.geom_size[geom_id],
                ref_data.geom_xpos[geom_id],
                ref_data.geom_xmat[geom_id],
                MOCAP_REFERENCE_RGBA,
            )
            scene.ngeom += 1


def print_motion_summary(motion: dict[str, np.ndarray], source: Path, episode: str) -> None:
    root = motion["qpos"][:, :3]
    delta = root[-1] - root[0]
    duration = max((len(root) - 1) / 30.0, 1.0 / 30.0)
    speed = float(np.linalg.norm(delta[:2]) / duration)
    print(f"[INFO] Motion source: {source}:{episode}")
    print(
        "[INFO] Motion summary: "
        f"frames={len(root)} duration={duration:.2f}s "
        f"root_delta=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f})m "
        f"xy_speed={speed:.3f}m/s"
    )


def _flatten_current_obs(env) -> np.ndarray:
    raw_obs = env.unwrapped.get_obs()
    if hasattr(env, "observation"):
        return env.observation(raw_obs)
    return flatten(env.unwrapped.observation_space, raw_obs)


def _blend_quat_toward(current: np.ndarray, target: np.ndarray, gain: float) -> np.ndarray:
    if np.dot(current, target) < 0.0:
        target = -target
    blended = current + gain * (target - current)
    norm = float(np.linalg.norm(blended))
    if norm < 1e-8:
        return current
    return blended / norm


def apply_mocap_tracking_assist(
    env,
    target_qpos: np.ndarray,
    target_qvel: np.ndarray,
    *,
    track_root_xy: bool,
    root_xy_position_gain: float,
    root_xy_velocity_gain: float,
    track_root_z: bool,
    root_z_position_gain: float,
    root_z_velocity_gain: float,
    track_root_orientation: bool,
    root_orientation_gain: float,
    root_angular_velocity_gain: float,
    joint_pose_gain: float,
    joint_velocity_gain: float,
) -> tuple[np.ndarray, bool]:
    import mujoco

    data = env.unwrapped.data
    corrected = False

    if track_root_xy:
        data.qpos[:2] += root_xy_position_gain * (target_qpos[:2] - data.qpos[:2])
        if len(data.qvel) >= 2 and len(target_qvel) >= 2:
            data.qvel[:2] += root_xy_velocity_gain * (target_qvel[:2] - data.qvel[:2])
        corrected = True

    if track_root_z:
        data.qpos[2] += root_z_position_gain * (target_qpos[2] - data.qpos[2])
        if len(data.qvel) >= 3 and len(target_qvel) >= 3:
            data.qvel[2] += root_z_velocity_gain * (target_qvel[2] - data.qvel[2])
        corrected = True

    if track_root_orientation and len(data.qpos) >= 7 and len(target_qpos) >= 7:
        data.qpos[3:7] = _blend_quat_toward(data.qpos[3:7].copy(), target_qpos[3:7].copy(), root_orientation_gain)
        if len(data.qvel) >= 6 and len(target_qvel) >= 6:
            data.qvel[3:6] += root_angular_velocity_gain * (target_qvel[3:6] - data.qvel[3:6])
        corrected = True

    if joint_pose_gain > 0.0 and len(data.qpos) > 7 and len(target_qpos) > 7:
        joint_count = min(len(data.qpos) - 7, len(target_qpos) - 7)
        data.qpos[7 : 7 + joint_count] += joint_pose_gain * (
            target_qpos[7 : 7 + joint_count] - data.qpos[7 : 7 + joint_count]
        )
        corrected = True

    if joint_velocity_gain > 0.0 and len(data.qvel) > 6 and len(target_qvel) > 6:
        joint_count = min(len(data.qvel) - 6, len(target_qvel) - 6)
        data.qvel[6 : 6 + joint_count] += joint_velocity_gain * (
            target_qvel[6 : 6 + joint_count] - data.qvel[6 : 6 + joint_count]
        )
        corrected = True

    if corrected:
        mujoco.mj_forward(env.unwrapped.model, data)
    return data.qpos[:3].copy(), corrected


def _mjcf_vec(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


UNITREE_TERRAIN_GENERATOR = Path("/home/yhzhu/myWorks_vips/unitree_mujoco/terrain_tool/terrain_generator.py")
MIMIC_MATCHED_RISERS = np.array([0.2630, 0.5820, 0.8330, 1.1080], dtype=np.float64)
MIMIC_MATCHED_LEVEL_HEIGHTS = np.array([0.0, 0.2045, 0.4125, 0.6055, 0.8060], dtype=np.float64)
FOOT_BODY_PAIRS = {"L": ("L_Ankle", "L_Toe"), "R": ("R_Ankle", "R_Toe")}
TERRAIN_GEOM_PREFIXES = (
    "mocap_stair",
    "mocap_support",
    "mimic_amass_stair",
)


def _load_unitree_terrain_generator():
    if not UNITREE_TERRAIN_GENERATOR.exists():
        raise FileNotFoundError(
            "Unitree terrain_generator.py not found. Expected "
            f"{UNITREE_TERRAIN_GENERATOR}. Clone unitreerobotics/unitree_mujoco there first."
        )

    # terrain_generator imports noise for Perlin hfields. Stairs only need boxes, so
    # keep this path dependency-free when noise is not installed in the HumEnv env.
    if "noise" not in sys.modules:
        noise_stub = types.ModuleType("noise")

        def _missing_noise(*_args, **_kwargs):
            raise RuntimeError("noise is required only for Unitree Perlin heightfields")

        noise_stub.pnoise2 = _missing_noise
        sys.modules["noise"] = noise_stub

    spec = importlib.util.spec_from_file_location("unitree_terrain_generator", UNITREE_TERRAIN_GENERATOR)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {UNITREE_TERRAIN_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contiguous_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append((start, idx))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


def _foot_supports_for_stairs(
    qpos: np.ndarray,
    xml_path: Path,
    direction_xy: np.ndarray,
    lateral_xy: np.ndarray,
    origin_xy: np.ndarray,
) -> list[dict[str, float]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for pair in FOOT_BODY_PAIRS.values()
        for name in pair
    }
    if any(body_id < 0 for body_id in body_ids.values()):
        return []

    tracks = {name: [] for name in body_ids}
    for pos in qpos:
        data.qpos[:] = pos
        mujoco.mj_forward(model, data)
        for name, body_id in body_ids.items():
            tracks[name].append(data.xpos[body_id].copy())
    tracks = {name: np.asarray(values, dtype=np.float64) for name, values in tracks.items()}

    supports: list[dict[str, float]] = []
    for side, names in FOOT_BODY_PAIRS.items():
        foot = 0.5 * (tracks[names[0]] + tracks[names[1]])
        speed = np.linalg.norm(np.gradient(foot[:, :2], axis=0), axis=1) * 30.0
        stance = speed < 0.18
        for start, end in _contiguous_ranges(stance):
            if end - start < 4:
                continue
            center = np.median(foot[start:end], axis=0)
            progress = float((center[:2] - origin_xy) @ direction_xy)
            lateral = float((center[:2] - origin_xy) @ lateral_xy)
            if any(
                abs(progress - item["progress"]) < 0.16
                and abs(float(center[2]) - item["foot_z"]) < 0.10
                for item in supports
            ):
                continue
            supports.append(
                {
                    "side": side,
                    "start": float(start),
                    "end": float(end),
                    "progress": progress,
                    "lateral": lateral,
                    "foot_z": float(center[2]),
                }
            )
    supports.sort(key=lambda item: (item["progress"], item["foot_z"], item["side"]))
    return supports


def _foot_matched_stair_spec(
    motion: dict[str, np.ndarray],
    xml_path: Path,
    direction_xy: np.ndarray,
    origin_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    lateral_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float64)
    supports = _foot_supports_for_stairs(motion["qpos"], xml_path, direction_xy, lateral_xy, origin_xy)
    if len(supports) < 4:
        return None

    ground_candidates = [item for item in supports if item["progress"] <= supports[0]["progress"] + 0.18]
    foot_to_ground = float(np.median([item["foot_z"] for item in ground_candidates]))
    levels: list[dict[str, float]] = []
    for support in supports:
        height = max(0.0, support["foot_z"] - foot_to_ground)
        if height < 0.05:
            height = 0.0
        if not levels or abs(height - levels[-1]["height"]) > 0.08 or support["progress"] - levels[-1]["progress"] > 0.30:
            levels.append({"progress": support["progress"], "height": height})
        else:
            levels[-1]["progress"] = 0.5 * (levels[-1]["progress"] + support["progress"])
            levels[-1]["height"] = max(levels[-1]["height"], height)

    elevated = [level for level in levels if level["height"] > 0.05]
    if len(elevated) < 2:
        return None

    first_ground_progress = min(level["progress"] for level in levels if level["height"] == 0.0)
    level_points = [{"progress": first_ground_progress, "height": 0.0}, *elevated]
    level_points.sort(key=lambda item: item["progress"])
    risers = np.asarray(
        [
            0.5 * (level_points[idx]["progress"] + level_points[idx + 1]["progress"])
            for idx in range(len(level_points) - 1)
        ],
        dtype=np.float64,
    )
    heights = np.asarray([item["height"] for item in level_points], dtype=np.float64)
    if np.any(np.diff(risers) < 0.12):
        return None
    return risers, heights


def _add_foot_matched_stairs(
    worldbody,
    *,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    yaw: float,
    risers: np.ndarray,
    heights: np.ndarray,
    run: float,
    stair_width: float,
) -> None:
    terrain_generator = _load_unitree_terrain_generator()
    quat = _mjcf_vec(terrain_generator.euler_to_quat(0.0, 0.0, yaw))
    ends = np.concatenate([risers[1:], [max(run + 0.55, risers[-1] + 0.55)]])
    for idx, (start_s, end_s, height) in enumerate(zip(risers, ends, heights[1:]), start=1):
        length = max(0.05, float(end_s - start_s))
        center_s = float(start_s) + 0.5 * length
        center_xy = origin_xy + direction_xy * center_s
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"mocap_stair_footmatched_{idx:02d}",
                "type": "box",
                "pos": _mjcf_vec((center_xy[0], center_xy[1], 0.5 * float(height))),
                "quat": quat,
                "size": _mjcf_vec((0.5 * length, stair_width / 2.0, 0.5 * float(height))),
                "condim": "3",
                "friction": "1.0 0.005 0.0001",
                "solimp": "0.98 0.98 0.001",
                "solref": "0.015 1",
                "rgba": "0.42 0.42 0.40 1",
            },
        )


def build_stair_xml(
    motion: dict[str, np.ndarray],
    out_dir: Path,
    *,
    stair_width: float,
    nominal_step_height: float,
    requested_steps: int,
) -> Path:
    root_xy = motion["qpos"][:, :2].astype(np.float64)
    root_z = motion["qpos"][:, 2].astype(np.float64)
    delta_xy = root_xy[-1] - root_xy[0]
    horizontal_run = float(np.linalg.norm(delta_xy))
    if horizontal_run < 1e-6:
        direction_xy = np.array([1.0, 0.0], dtype=np.float64)
        horizontal_run = 1.2
    else:
        direction_xy = delta_xy / horizontal_run

    total_rise = float(abs(root_z[-1] - root_z[0]))
    if total_rise < 0.05:
        total_rise = float(max(root_z) - min(root_z))
    if total_rise < 0.05:
        total_rise = nominal_step_height * 6

    steps = requested_steps if requested_steps > 0 else max(1, int(round(total_rise / nominal_step_height)))
    step_height = total_rise / steps
    tread_depth = max(0.18, horizontal_run / steps)
    descending = root_z[-1] < root_z[0]

    xml_path = Path(cfg_path("environment", "humenv_xml"))
    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"{xml_path} does not contain a <worldbody>")

    start_xy = root_xy[0]
    yaw = math.atan2(float(direction_xy[1]), float(direction_xy[0]))

    stair_xml = out_dir / "humenv_mocap_stairs.xml"
    if not descending:
        matched = _foot_matched_stair_spec(motion, xml_path, direction_xy, start_xy)
        if matched is not None:
            risers, heights = matched
            _add_foot_matched_stairs(
                worldbody,
                origin_xy=start_xy,
                direction_xy=direction_xy,
                yaw=yaw,
                risers=risers,
                heights=heights,
                run=horizontal_run,
                stair_width=stair_width,
            )
            stair_xml.parent.mkdir(parents=True, exist_ok=True)
            tree.write(stair_xml, encoding="utf-8", xml_declaration=True)
            print(
                "[INFO] Generated foot-matched stair terrain: "
                f"risers={np.round(risers, 3).tolist()} heights={np.round(heights, 3).tolist()} "
                f"width={stair_width:.3f}m xml={stair_xml}"
            )
            return stair_xml

    terrain_generator = _load_unitree_terrain_generator()
    terrain_generator.INPUT_SCENE_PATH = str(xml_path)
    terrain_generator.OUTPUT_SCENE_PATH = str(stair_xml)
    terrain = terrain_generator.TerrainGenerator()
    existing_geom_count = len(terrain.worldbody.findall("geom"))

    if descending:
        low_end_xy = root_xy[-1]
        stair_init_xy = low_end_xy
        stair_yaw = yaw + math.pi
    else:
        stair_init_xy = start_xy
        stair_yaw = yaw
    terrain.AddStairs(
        init_pos=[float(stair_init_xy[0]), float(stair_init_xy[1]), 0.0],
        yaw=stair_yaw,
        width=tread_depth,
        height=step_height,
        length=stair_width,
        stair_nums=steps,
    )

    new_geoms = terrain.worldbody.findall("geom")[existing_geom_count:]
    for idx, geom in enumerate(new_geoms, start=1):
        geom.attrib["name"] = f"mocap_stair_{idx:02d}"
        geom.attrib["condim"] = "3"
        geom.attrib["friction"] = "1.0 0.005 0.0001"
        geom.attrib["solimp"] = "0.98 0.98 0.001"
        geom.attrib["solref"] = "0.015 1"
        geom.attrib["rgba"] = "0.42 0.42 0.40 1"

    top_landing_direction = -direction_xy if descending else direction_xy
    high_side_xy = start_xy if descending else start_xy + direction_xy * ((steps + 1.0) * tread_depth)
    high_center_xy = high_side_xy + top_landing_direction * (0.75 * tread_depth)
    ET.SubElement(
        terrain.worldbody,
        "geom",
        {
            "name": "mocap_stair_top_landing",
            "type": "box",
            "pos": _mjcf_vec((high_center_xy[0], high_center_xy[1], total_rise / 2.0)),
            "quat": _mjcf_vec(terrain_generator.euler_to_quat(0.0, 0.0, stair_yaw)),
            "size": _mjcf_vec((0.75 * tread_depth, stair_width / 2.0, total_rise / 2.0)),
            "condim": "3",
            "friction": "1.0 0.005 0.0001",
            "solimp": "0.98 0.98 0.001",
            "solref": "0.015 1",
            "rgba": "0.36 0.36 0.34 1",
        },
    )

    terrain.Save()
    print(
        "[INFO] Generated 3D stair terrain with Unitree terrain_generator.py: "
        f"steps={steps} rise={step_height:.3f}m tread={tread_depth:.3f}m "
        f"total_height={total_rise:.3f}m width={stair_width:.3f}m xml={stair_xml}"
    )
    return stair_xml


def support_sidecar_path(motion_path: Path) -> Path:
    if motion_path.name.endswith("_upx_upstairs.hdf5"):
        return motion_path.with_name(motion_path.stem + "_pillars.supports.json")
    return motion_path.with_suffix(".supports.json")


def build_support_xml(motion_path: Path, out_dir: Path) -> Path:
    supports_path = support_sidecar_path(motion_path)
    if not supports_path.exists():
        raise FileNotFoundError(
            f"Support sidecar not found for {motion_path}: {supports_path}. "
            "Use --terrain stairs for synthetic regular stairs, or provide a mocap with a .supports.json sidecar."
        )

    supports = json.loads(supports_path.read_text())
    xml_path = Path(cfg_path("environment", "humenv_xml"))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{xml_path} does not contain a <worldbody>")

    for idx, support in enumerate(supports, start=1):
        center = np.asarray(support["center"], dtype=np.float64)
        top_z = float(support.get("top_z", center[2]))
        half_height = max(0.025, 0.5 * top_z)
        geom_center = center.copy()
        geom_center[2] = half_height
        plate_width = float(support.get("plate_width", 0.30))
        plate_length = float(support.get("plate_length", 0.30))
        side = support.get("side", "?")
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"mocap_support_{idx:02d}_{side}",
                "type": "box",
                "pos": _mjcf_vec(geom_center),
                "size": _mjcf_vec((plate_width / 2.0, plate_length / 2.0, half_height)),
                "condim": "3",
                "friction": "1.0 0.005 0.0001",
                "solimp": "0.98 0.98 0.001",
                "solref": "0.015 1",
                "rgba": "0.38 0.42 0.36 1",
            },
        )

    support_xml = out_dir / "humenv_mocap_supports.xml"
    support_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(support_xml, encoding="utf-8", xml_declaration=True)
    print(
        "[INFO] Generated mocap support terrain from sidecar: "
        f"supports={len(supports)} sidecar={supports_path} xml={support_xml}"
    )
    return support_xml


def _motion_direction_xy(motion: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    root_xy = motion["qpos"][:, :2].astype(np.float64)
    delta_xy = root_xy[-1] - root_xy[0]
    run = float(np.linalg.norm(delta_xy))
    if run < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0
    return delta_xy / run, run


def build_mimic_matched_stair_xml(
    motion: dict[str, np.ndarray],
    out_dir: Path,
    *,
    stair_width: float,
) -> Path:
    """Build MuJoCo boxes matching hit_exo_mimic's AMASS-matched stairs heightfield.

    The source terrain is /home/yhzhu/myWorks_vips/hit_exo_mimic/tools/
    build_amass_matched_stairs_terrain.py. Its x_scene coordinate starts at the
    AMASS root origin, so here we align x_scene=0 to the HumEnv mocap root at
    frame 0 and rotate the risers along the mocap's XY displacement.
    """
    xml_path = Path(cfg_path("environment", "humenv_xml"))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{xml_path} does not contain a <worldbody>")

    root_xy0 = motion["qpos"][0, :2].astype(np.float64)
    direction_xy, run = _motion_direction_xy(motion)
    lateral_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float64)
    yaw = math.atan2(float(direction_xy[1]), float(direction_xy[0]))
    quat = _mjcf_vec(_load_unitree_terrain_generator().euler_to_quat(0.0, 0.0, yaw))

    # One elevated box per constant-height level. The ground before first riser
    # is the base plane from HumEnv.
    level_starts = MIMIC_MATCHED_RISERS
    level_ends = np.concatenate([MIMIC_MATCHED_RISERS[1:], [max(run + 0.45, MIMIC_MATCHED_RISERS[-1] + 0.80)]])
    for idx, (start, end, height) in enumerate(
        zip(level_starts, level_ends, MIMIC_MATCHED_LEVEL_HEIGHTS[1:]),
        start=1,
    ):
        center_s = 0.5 * (float(start) + float(end))
        length_s = max(0.02, float(end) - float(start))
        center_xy = root_xy0 + direction_xy * center_s
        center = np.array([center_xy[0], center_xy[1], 0.5 * float(height)], dtype=np.float64)
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"mimic_amass_stair_level_{idx:02d}",
                "type": "box",
                "pos": _mjcf_vec(center),
                "quat": quat,
                "size": _mjcf_vec((0.5 * length_s, 0.5 * stair_width, 0.5 * float(height))),
                "condim": "3",
                "friction": "1.0 0.005 0.0001",
                "solimp": "0.98 0.98 0.001",
                "solref": "0.015 1",
                "rgba": "0.40 0.40 0.38 1",
            },
        )

    stair_xml = out_dir / "humenv_mimic_amass_matched_stairs.xml"
    stair_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(stair_xml, encoding="utf-8", xml_declaration=True)
    riser_world = [root_xy0 + direction_xy * s for s in MIMIC_MATCHED_RISERS]
    print(
        "[INFO] Generated hit_exo_mimic AMASS-matched stairs: "
        f"risers={MIMIC_MATCHED_RISERS.tolist()} heights={MIMIC_MATCHED_LEVEL_HEIGHTS.tolist()} "
        f"yaw={math.degrees(yaw):.3f}deg width={stair_width:.3f}m xml={stair_xml}"
    )
    print(
        "[INFO] Mimic stair riser XY: "
        + ", ".join(f"({xy[0]:.3f},{xy[1]:.3f})" for xy in riser_world)
    )
    return stair_xml


def as_numpy_action(action) -> np.ndarray:
    if hasattr(action, "detach"):
        return action.detach().cpu().numpy().ravel()
    return np.asarray(action, dtype=np.float64).ravel()


def qpos_error_summary(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "root": float(np.linalg.norm(actual[:3] - target[:3])),
        "root_xy": float(np.linalg.norm(actual[:2] - target[:2])),
        "root_z": float(abs(actual[2] - target[2])),
        "joints": float(np.linalg.norm(actual[7:] - target[7:]) / math.sqrt(max(1, len(target[7:])))),
    }


def _foot_body_ids(model) -> dict[str, tuple[int, int]]:
    import mujoco

    ids: dict[str, tuple[int, int]] = {}
    for side, names in FOOT_BODY_PAIRS.items():
        body_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names)
        if any(body_id < 0 for body_id in body_ids):
            raise ValueError(f"Cannot find foot bodies for side {side}: {names}")
        ids[side] = body_ids
    return ids


def _foot_positions_from_data(data, body_ids: dict[str, tuple[int, int]]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for side, (ankle_id, toe_id) in body_ids.items():
        ankle = data.xpos[ankle_id].copy()
        toe = data.xpos[toe_id].copy()
        out[side] = {
            "ankle": ankle,
            "toe": toe,
            "center": 0.5 * (ankle + toe),
        }
    return out


class TerrainHeightLookup:
    def __init__(self, xml_path: Path):
        import mujoco

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._geoms = []
        for geom_id in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not name.startswith(TERRAIN_GEOM_PREFIXES):
                continue
            if int(self._model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            self._geoms.append(
                {
                    "name": name,
                    "xpos": self._data.geom_xpos[geom_id].copy(),
                    "xmat": self._data.geom_xmat[geom_id].reshape(3, 3).copy(),
                    "size": self._model.geom_size[geom_id].copy(),
                }
            )

    def height_at(self, xy: np.ndarray, *, tolerance: float = 1e-5) -> dict[str, object]:
        best_z = 0.0
        best_name = "ground"
        for geom in self._geoms:
            center = geom["xpos"]
            point = np.array([float(xy[0]), float(xy[1]), float(center[2])], dtype=np.float64)
            local = geom["xmat"].T @ (point - center)
            size = geom["size"]
            if abs(float(local[0])) <= float(size[0]) + tolerance and abs(float(local[1])) <= float(size[1]) + tolerance:
                top_z = float(center[2] + size[2])
                if top_z >= best_z:
                    best_z = top_z
                    best_name = str(geom["name"])
        return {"z": best_z, "name": best_name}


def compute_mocap_foot_tracks(
    qpos: np.ndarray,
    xml_path: Path,
    *,
    stance_speed_threshold: float,
    min_stance_frames: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    body_ids = _foot_body_ids(model)
    tracks: dict[str, dict[str, list[np.ndarray]]] = {
        side: {"ankle": [], "toe": [], "center": []}
        for side in FOOT_BODY_PAIRS
    }
    for pos in qpos:
        data.qpos[:] = pos
        mujoco.mj_forward(model, data)
        foot_positions = _foot_positions_from_data(data, body_ids)
        for side, positions in foot_positions.items():
            for name, value in positions.items():
                tracks[side][name].append(value)

    np_tracks: dict[str, dict[str, np.ndarray]] = {}
    landing_events: list[dict[str, object]] = []
    for side, positions in tracks.items():
        center = np.asarray(positions["center"], dtype=np.float64)
        speed_xy = np.linalg.norm(np.gradient(center[:, :2], axis=0), axis=1) * 30.0
        stance = speed_xy < stance_speed_threshold
        np_tracks[side] = {
            "ankle": np.asarray(positions["ankle"], dtype=np.float64),
            "toe": np.asarray(positions["toe"], dtype=np.float64),
            "center": center,
            "speed_xy": speed_xy,
            "stance": stance,
        }
        support_idx = 0
        for start, end in _contiguous_ranges(stance):
            if end - start < min_stance_frames:
                continue
            support_idx += 1
            center_median = np.median(center[start:end], axis=0)
            event_type = "initial_stance" if start <= 1 else "touchdown"
            landing_events.append(
                {
                    "side": side,
                    "support_index": support_idx,
                    "event_type": event_type,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "duration_frames": int(end - start),
                    "mocap_x": float(center_median[0]),
                    "mocap_y": float(center_median[1]),
                    "mocap_z": float(center_median[2]),
                    "mocap_speed_xy": float(speed_xy[start]),
                }
            )
    landing_events.sort(key=lambda item: (int(item["start_frame"]), str(item["side"])))
    return np_tracks, landing_events


def first_step_landing(landing_events: list[dict[str, object]]) -> dict[str, object] | None:
    touchdown = [event for event in landing_events if event["event_type"] == "touchdown"]
    if touchdown:
        return touchdown[0]
    return landing_events[0] if landing_events else None


def _vec_components(prefix: str, value: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_x": float(value[0]),
        f"{prefix}_y": float(value[1]),
        f"{prefix}_z": float(value[2]),
    }


def main() -> None:
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from metamotivo.wrappers.humenvbench import TrackingWrapper

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / "status.json"
    status: dict[str, object] = {
        "state": "starting",
        "started_at": _now(),
        "args": vars(args),
        "status_path": status_path,
    }
    write_status(status_path, status)

    previous_excepthook = sys.excepthook

    def _status_excepthook(exc_type, exc, tb):
        write_status(
            status_path,
            status,
            state="failed",
            failed_at=_now(),
            error=f"{exc_type.__name__}: {exc}",
        )
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = _status_excepthook

    if not args.motion.exists():
        raise FileNotFoundError(f"Motion file not found: {args.motion}")
    gain_values = {
        "--root-xy-track-gain": args.root_xy_track_gain,
        "--root-xy-velocity-gain": args.root_xy_velocity_gain,
        "--root-z-track-gain": args.root_z_track_gain,
        "--root-z-velocity-gain": args.root_z_velocity_gain,
        "--root-orientation-track-gain": args.root_orientation_track_gain,
        "--root-angular-velocity-gain": args.root_angular_velocity_gain,
        "--joint-pose-track-gain": args.joint_pose_track_gain,
        "--joint-velocity-track-gain": args.joint_velocity_track_gain,
    }
    invalid_gains = [name for name, value in gain_values.items() if value < 0.0 or value > 1.0]
    if invalid_gains:
        raise ValueError(f"Tracking assist gains must be in [0, 1]: {', '.join(invalid_gains)}")
    if args.foot_stance_speed_threshold <= 0.0:
        raise ValueError("--foot-stance-speed-threshold must be positive")
    if args.landing_min_stance_frames <= 0:
        raise ValueError("--landing-min-stance-frames must be positive")

    write_status(status_path, status, state="loading_motion")
    motion = load_motion(args.motion, args.episode)
    print_motion_summary(motion, args.motion, args.episode)
    if args.start_frame < 0 or args.start_frame >= len(motion["qpos"]) - 1:
        raise ValueError(f"--start-frame must be in [0, {len(motion['qpos']) - 2}]")
    motion = {key: value[args.start_frame :] for key, value in motion.items()}
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    if args.duration <= 0:
        rollout_steps = len(motion["observation"]) - 1
    else:
        rollout_steps = min(len(motion["observation"]) - 1, max(1, int(args.duration * 30)))

    write_status(
        status_path,
        status,
        state="building_terrain",
        rollout_steps=rollout_steps,
        motion_frames=len(motion["qpos"]),
    )
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
    reference_xml = env_xml if env_xml is not None else Path(cfg_path("environment", "humenv_xml"))
    write_status(status_path, status, terrain_xml=reference_xml)

    write_status(status_path, status, state="analyzing_mocap_feet")
    mocap_foot_tracks, landing_events = compute_mocap_foot_tracks(
        motion["qpos"][: rollout_steps + 1],
        reference_xml,
        stance_speed_threshold=args.foot_stance_speed_threshold,
        min_stance_frames=args.landing_min_stance_frames,
    )
    supports_path = args.out_dir / "mocap_foot_supports.csv"
    support_fields = [
        "side",
        "support_index",
        "event_type",
        "start_frame",
        "end_frame",
        "duration_frames",
        "mocap_x",
        "mocap_y",
        "mocap_z",
        "mocap_speed_xy",
    ]
    with supports_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=support_fields)
        writer.writeheader()
        writer.writerows(landing_events)
    landing_events_by_frame: dict[int, list[dict[str, object]]] = {}
    for event in landing_events:
        landing_events_by_frame.setdefault(int(event["start_frame"]), []).append(event)
    first_landing_event = first_step_landing(landing_events)
    if first_landing_event is not None:
        print(
            "[INFO] First mocap foot support/landing: "
            f"side={first_landing_event['side']} type={first_landing_event['event_type']} "
            f"frame={first_landing_event['start_frame']} "
            f"pos=({first_landing_event['mocap_x']:.3f}, "
            f"{first_landing_event['mocap_y']:.3f}, {first_landing_event['mocap_z']:.3f})"
        )
    print(f"[INFO] Wrote mocap foot support log: {supports_path}")
    write_status(
        status_path,
        status,
        mocap_foot_supports_csv=supports_path,
        landing_event_count=len(landing_events),
        first_landing_event=first_landing_event,
    )

    write_status(status_path, status, state="loading_model")
    print(f"[INFO] Loading frozen S-1 model: {args.model_id} on {args.device}")
    base_model = FBcprModel.from_pretrained(args.model_id, device=args.device)
    model = TrackingWrapper(model=base_model)

    next_obs = torch.as_tensor(
        motion["observation"][1 : rollout_steps + 1],
        dtype=torch.float32,
        device=args.device,
    )
    print(f"[INFO] Inferring tracking latents from mocap observation: {tuple(next_obs.shape)}")
    with torch.no_grad():
        z_seq = model.tracking_inference(next_obs=next_obs)
    latents_path = args.out_dir / "tracking_latents.pt"
    torch.save(z_seq.detach().cpu(), latents_path)
    write_status(status_path, status, state="creating_env", tracking_latents_path=latents_path, z_shape=list(z_seq.shape))

    env, _ = make_humenv(
        num_envs=1,
        task="zero",
        state_init="Default",
        wrappers=[FlattenObservation],
        max_episode_steps=max(rollout_steps + 5, 300),
        **({"xml": str(env_xml)} if env_xml is not None else {}),
    )
    sim_foot_body_ids = _foot_body_ids(env.unwrapped.model)
    terrain_heights = TerrainHeightLookup(reference_xml)
    initial_obs, _ = env.reset(options={"qpos": motion["qpos"][0], "qvel": motion["qvel"][0]})
    reset_error = qpos_error_summary(env.unwrapped.data.qpos.copy(), motion["qpos"][0])
    initial_obs_mse = float(np.mean((np.asarray(initial_obs).ravel() - motion["observation"][0]) ** 2))
    print(
        "[INFO] Initial dynamic reset error vs mocap: "
        f"root={reset_error['root']:.6f}m root_xy={reset_error['root_xy']:.6f}m "
        f"root_z={reset_error['root_z']:.6f}m joint_rms={reset_error['joints']:.6f}rad "
        f"obs_mse={initial_obs_mse:.8f}"
    )
    write_status(
        status_path,
        status,
        state="ready_for_rollout",
        initial_reset_error=reset_error,
        initial_obs_mse=initial_obs_mse,
    )

    viewer = None
    mocap_reference = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
        configure_viewer_camera(viewer, motion)
        mocap_reference = make_mocap_reference_renderer(reference_xml, motion["qpos"][0])
        update_mocap_reference_scene(viewer, *mocap_reference, motion["qpos"][0])
        print("[INFO] Viewer overlay: mocap reference humanoid is green; dynamic S-1 keeps the XML color.")
    csv_path = args.out_dir / "rollout.csv"
    foot_csv_path = args.out_dir / "foot_tracking.csv"
    landing_csv_path = args.out_dir / "foot_landings.csv"
    rows = []
    foot_rows = []
    landing_rows = []
    print(f"[INFO] Visualizing native HumEnv S-1 mocap tracking only: {args.motion}:{args.episode}")
    print("[INFO] No training, no mjlab wrapper, no knee-exo action.")
    print(f"[INFO] Repeat mode: reset to the initial mocap state before each of {args.repeat} cycle(s).")
    if args.track_root_xy:
        print(
            "[INFO] Root XY tracking enabled: "
            f"position_gain={args.root_xy_track_gain:g} velocity_gain={args.root_xy_velocity_gain:g}"
        )
    if args.track_root_z:
        print(
            "[INFO] Root Z tracking enabled: "
            f"position_gain={args.root_z_track_gain:g} velocity_gain={args.root_z_velocity_gain:g}"
        )
    if args.track_root_orientation:
        print(
            "[INFO] Root orientation tracking enabled: "
            f"quat_gain={args.root_orientation_track_gain:g} angular_velocity_gain={args.root_angular_velocity_gain:g}"
        )
    if args.joint_pose_track_gain > 0.0 or args.joint_velocity_track_gain > 0.0:
        print(
            "[INFO] Joint tracking assist enabled: "
            f"pose_gain={args.joint_pose_track_gain:g} velocity_gain={args.joint_velocity_track_gain:g}"
        )

    def append_foot_diagnostics(cycle: int, frame: int, target_frame: int, phase: str):
        sim_feet = _foot_positions_from_data(env.unwrapped.data, sim_foot_body_ids)
        for side in FOOT_BODY_PAIRS:
            sim_center = sim_feet[side]["center"]
            sim_ankle = sim_feet[side]["ankle"]
            sim_toe = sim_feet[side]["toe"]
            mocap_center = mocap_foot_tracks[side]["center"][target_frame]
            mocap_ankle = mocap_foot_tracks[side]["ankle"][target_frame]
            mocap_toe = mocap_foot_tracks[side]["toe"][target_frame]
            delta = sim_center - mocap_center
            sim_terrain = terrain_heights.height_at(sim_center[:2])
            mocap_terrain = terrain_heights.height_at(mocap_center[:2])
            foot_rows.append(
                {
                    "cycle": cycle,
                    "phase": phase,
                    "frame": frame,
                    "target_frame": target_frame,
                    "side": side,
                    **_vec_components("sim_center", sim_center),
                    **_vec_components("mocap_center", mocap_center),
                    **_vec_components("sim_ankle", sim_ankle),
                    **_vec_components("mocap_ankle", mocap_ankle),
                    **_vec_components("sim_toe", sim_toe),
                    **_vec_components("mocap_toe", mocap_toe),
                    "sim_terrain_name": sim_terrain["name"],
                    "sim_terrain_z": sim_terrain["z"],
                    "mocap_terrain_name": mocap_terrain["name"],
                    "mocap_terrain_z": mocap_terrain["z"],
                    "sim_center_clearance": float(sim_center[2] - float(sim_terrain["z"])),
                    "mocap_center_clearance": float(mocap_center[2] - float(mocap_terrain["z"])),
                    "center_error": float(np.linalg.norm(delta)),
                    "center_xy_error": float(np.linalg.norm(delta[:2])),
                    "center_z_error": float(abs(delta[2])),
                    "mocap_speed_xy": float(mocap_foot_tracks[side]["speed_xy"][target_frame]),
                    "mocap_stance": bool(mocap_foot_tracks[side]["stance"][target_frame]),
                }
            )
        for event in landing_events_by_frame.get(target_frame, []):
            side = str(event["side"])
            sim_center = sim_feet[side]["center"]
            mocap_center = np.array(
                [
                    float(event["mocap_x"]),
                    float(event["mocap_y"]),
                    float(event["mocap_z"]),
                ],
                dtype=np.float64,
            )
            delta = sim_center - mocap_center
            sim_terrain = terrain_heights.height_at(sim_center[:2])
            mocap_terrain = terrain_heights.height_at(mocap_center[:2])
            landing_rows.append(
                {
                    "cycle": cycle,
                    "phase": phase,
                    "frame": frame,
                    "target_frame": target_frame,
                    "side": side,
                    "event_type": event["event_type"],
                    "support_index": event["support_index"],
                    "support_start_frame": event["start_frame"],
                    "support_end_frame": event["end_frame"],
                    "support_duration_frames": event["duration_frames"],
                    **_vec_components("sim_center", sim_center),
                    **_vec_components("mocap_landing", mocap_center),
                    "sim_terrain_name": sim_terrain["name"],
                    "sim_terrain_z": sim_terrain["z"],
                    "mocap_terrain_name": mocap_terrain["name"],
                    "mocap_terrain_z": mocap_terrain["z"],
                    "sim_center_clearance": float(sim_center[2] - float(sim_terrain["z"])),
                    "mocap_center_clearance": float(mocap_center[2] - float(mocap_terrain["z"])),
                    "center_clearance_error": float(
                        abs((sim_center[2] - float(sim_terrain["z"])) - (mocap_center[2] - float(mocap_terrain["z"])))
                    ),
                    "landing_error": float(np.linalg.norm(delta)),
                    "landing_xy_error": float(np.linalg.norm(delta[:2])),
                    "landing_z_error": float(abs(delta[2])),
                    "mocap_speed_xy": event["mocap_speed_xy"],
                }
            )
        return sim_feet

    write_status(status_path, status, state="rolling_out")
    try:
        for cycle in range(args.repeat):
            obs, _ = env.reset(options={"qpos": motion["qpos"][0], "qvel": motion["qvel"][0]})
            append_foot_diagnostics(cycle=cycle, frame=-1, target_frame=0, phase="reset")
            if viewer is not None and mocap_reference is not None:
                update_mocap_reference_scene(viewer, *mocap_reference, motion["qpos"][0])
                viewer.sync()

            for frame in range(rollout_steps):
                obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=args.device)
                with torch.no_grad():
                    action = as_numpy_action(model.act(obs=obs_t, z=z_seq[frame : frame + 1]))
                obs, _reward, terminated, truncated, _info = env.step(action)

                target_root = motion["qpos"][frame + 1, :3]
                target_qvel = motion["qvel"][frame + 1]
                target_qpos = motion["qpos"][frame + 1]
                root, corrected_state = apply_mocap_tracking_assist(
                    env,
                    target_qpos,
                    target_qvel,
                    track_root_xy=args.track_root_xy,
                    root_xy_position_gain=args.root_xy_track_gain,
                    root_xy_velocity_gain=args.root_xy_velocity_gain,
                    track_root_z=args.track_root_z,
                    root_z_position_gain=args.root_z_track_gain,
                    root_z_velocity_gain=args.root_z_velocity_gain,
                    track_root_orientation=args.track_root_orientation,
                    root_orientation_gain=args.root_orientation_track_gain,
                    root_angular_velocity_gain=args.root_angular_velocity_gain,
                    joint_pose_gain=args.joint_pose_track_gain,
                    joint_velocity_gain=args.joint_velocity_track_gain,
                )
                if corrected_state:
                    obs = _flatten_current_obs(env)
                target_frame = frame + 1
                append_foot_diagnostics(cycle=cycle, frame=frame, target_frame=target_frame, phase="rollout")
                root_xy_error = float(np.linalg.norm(root[:2] - target_root[:2]))
                root_z_error = float(abs(root[2] - target_root[2]))
                rows.append(
                    {
                        "cycle": cycle,
                        "frame": frame,
                        "root_x": root[0],
                        "root_y": root[1],
                        "root_z": root[2],
                        "target_x": target_root[0],
                        "target_y": target_root[1],
                        "target_z": target_root[2],
                        "root_xy_error": root_xy_error,
                        "root_z_error": root_z_error,
                        "root_error": float(np.linalg.norm(root - target_root)),
                        "obs_mse": float(np.mean((np.asarray(obs).ravel() - motion["observation"][frame + 1]) ** 2)),
                    }
                )
                if terminated or truncated:
                    break
                if viewer is not None and not viewer.is_running():
                    break
                if viewer is not None:
                    if mocap_reference is not None:
                        update_mocap_reference_scene(viewer, *mocap_reference, motion["qpos"][frame + 1])
                    viewer.sync()
                    time.sleep(1.0 / 30.0)
            if viewer is not None and not viewer.is_running():
                break
    finally:
        env.close()

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cycle",
                "frame",
                "root_x",
                "root_y",
                "root_z",
                "target_x",
                "target_y",
                "target_z",
                "root_xy_error",
                "root_z_error",
                "root_error",
                "obs_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    foot_fields = [
        "cycle",
        "phase",
        "frame",
        "target_frame",
        "side",
        "sim_center_x",
        "sim_center_y",
        "sim_center_z",
        "mocap_center_x",
        "mocap_center_y",
        "mocap_center_z",
        "sim_ankle_x",
        "sim_ankle_y",
        "sim_ankle_z",
        "mocap_ankle_x",
        "mocap_ankle_y",
        "mocap_ankle_z",
        "sim_toe_x",
        "sim_toe_y",
        "sim_toe_z",
        "mocap_toe_x",
        "mocap_toe_y",
        "mocap_toe_z",
        "sim_terrain_name",
        "sim_terrain_z",
        "mocap_terrain_name",
        "mocap_terrain_z",
        "sim_center_clearance",
        "mocap_center_clearance",
        "center_error",
        "center_xy_error",
        "center_z_error",
        "mocap_speed_xy",
        "mocap_stance",
    ]
    with foot_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=foot_fields)
        writer.writeheader()
        writer.writerows(foot_rows)
    landing_fields = [
        "cycle",
        "phase",
        "frame",
        "target_frame",
        "side",
        "event_type",
        "support_index",
        "support_start_frame",
        "support_end_frame",
        "support_duration_frames",
        "sim_center_x",
        "sim_center_y",
        "sim_center_z",
        "mocap_landing_x",
        "mocap_landing_y",
        "mocap_landing_z",
        "sim_terrain_name",
        "sim_terrain_z",
        "mocap_terrain_name",
        "mocap_terrain_z",
        "sim_center_clearance",
        "mocap_center_clearance",
        "center_clearance_error",
        "landing_error",
        "landing_xy_error",
        "landing_z_error",
        "mocap_speed_xy",
    ]
    with landing_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=landing_fields)
        writer.writeheader()
        writer.writerows(landing_rows)
    if rows:
        root_errors = [row["root_error"] for row in rows]
        root_xy_errors = [row["root_xy_error"] for row in rows]
        root_z_errors = [row["root_z_error"] for row in rows]
        print(
            "[INFO] root_error "
            f"mean={np.mean(root_errors):.4f}m max={np.max(root_errors):.4f}m final={root_errors[-1]:.4f}m"
        )
        print(
            "[INFO] root_xy_error "
            f"mean={np.mean(root_xy_errors):.4f}m max={np.max(root_xy_errors):.4f}m "
            f"final={root_xy_errors[-1]:.4f}m"
        )
        print(
            "[INFO] root_z_error "
            f"mean={np.mean(root_z_errors):.4f}m max={np.max(root_z_errors):.4f}m "
            f"final={root_z_errors[-1]:.4f}m"
        )
        if root_errors[-1] > 0.5:
            print(
                "[WARN] Frozen S-1 dynamics drifted far from the mocap target. "
                "Use show_mocap_track_updown_stairs.sh for kinematic mocap/terrain alignment; "
                "this script tests whether S-1 can dynamically track the clip."
            )
    if foot_rows:
        foot_errors = [row["center_error"] for row in foot_rows if row["phase"] == "rollout"]
        foot_xy_errors = [row["center_xy_error"] for row in foot_rows if row["phase"] == "rollout"]
        if foot_errors:
            print(
                "[INFO] foot_center_error "
                f"mean={np.mean(foot_errors):.4f}m max={np.max(foot_errors):.4f}m"
            )
            print(
                "[INFO] foot_center_xy_error "
                f"mean={np.mean(foot_xy_errors):.4f}m max={np.max(foot_xy_errors):.4f}m"
            )
    first_landing_row = None
    touchdown_rows = [row for row in landing_rows if row["event_type"] == "touchdown" and row["cycle"] == 0]
    if touchdown_rows:
        first_landing_row = touchdown_rows[0]
    else:
        initial_rows = [row for row in landing_rows if row["cycle"] == 0]
        first_landing_row = initial_rows[0] if initial_rows else None
    if first_landing_row is not None:
        print(
            "[INFO] first logged foot landing/support error: "
            f"side={first_landing_row['side']} type={first_landing_row['event_type']} "
            f"target_frame={first_landing_row['target_frame']} "
            f"error={first_landing_row['landing_error']:.4f}m "
            f"xy={first_landing_row['landing_xy_error']:.4f}m "
            f"z={first_landing_row['landing_z_error']:.4f}m"
        )
        print(
            "[INFO] first logged foot landing terrain clearance: "
            f"target_terrain={first_landing_row['mocap_terrain_name']} "
            f"target_terrain_z={first_landing_row['mocap_terrain_z']:.4f}m "
            f"mocap_clearance={first_landing_row['mocap_center_clearance']:.4f}m "
            f"sim_terrain={first_landing_row['sim_terrain_name']} "
            f"sim_terrain_z={first_landing_row['sim_terrain_z']:.4f}m "
            f"sim_clearance={first_landing_row['sim_center_clearance']:.4f}m "
            f"clearance_error={first_landing_row['center_clearance_error']:.4f}m"
        )
    print(f"[INFO] Wrote rollout log: {csv_path}")
    print(f"[INFO] Wrote foot tracking log: {foot_csv_path}")
    print(f"[INFO] Wrote foot landing log: {landing_csv_path}")
    write_status(
        status_path,
        status,
        state="completed",
        completed_at=_now(),
        rollout_csv=csv_path,
        foot_tracking_csv=foot_csv_path,
        foot_landings_csv=landing_csv_path,
        row_count=len(rows),
        foot_row_count=len(foot_rows),
        landing_row_count=len(landing_rows),
        first_logged_landing=first_landing_row,
    )
    print(f"[INFO] Wrote status log: {status_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    if viewer is not None:
        os._exit(0)


if __name__ == "__main__":
    main()
