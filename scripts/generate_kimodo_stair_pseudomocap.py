#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DEFAULT_OUT_DIR = Path(".omx/kimodo_stair_pseudomocap")
DEFAULT_CONVERT = Path("/home/yhzhu/AI/humenv/scripts/convert_amass_smplsim_motion.py")
DEFAULT_SMPLSIM_PYTHON = Path("/home/yhzhu/miniconda3/envs/isaacgym/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a regular-stair pseudo mocap clip with Kimodo-SMPLX and convert it to HumEnv HDF5."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default="kimodo_stairs_up")
    parser.add_argument("--direction", choices=("up", "down"), default="up")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--step-height", type=float, default=0.135)
    parser.add_argument("--tread-depth", type=float, default=0.22)
    parser.add_argument("--duration", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--model", default="Kimodo-SMPLX-RP-v1")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--constraints", type=Path, default=None)
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--use-dummy-text-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--convert-script", type=Path, default=DEFAULT_CONVERT)
    parser.add_argument("--smplsim-python", type=Path, default=DEFAULT_SMPLSIM_PYTHON)
    return parser.parse_args()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def stair_prompt(direction: str) -> str:
    if direction == "up":
        return "A person walks up a short flight of stairs with steady alternating steps."
    return "A person walks down a short flight of stairs with steady alternating steps."


def build_root_constraints(args: argparse.Namespace, path: Path) -> None:
    frame_count = max(2, int(round(args.duration * 30.0)))
    run = args.steps * args.tread_depth
    if args.direction == "down":
        root_2d = [[0.0, run * (1.0 - i / (frame_count - 1))] for i in range(frame_count)]
        heading = [[-1.0, 0.0] for _ in range(frame_count)]
    else:
        root_2d = [[0.0, run * i / (frame_count - 1)] for i in range(frame_count)]
        heading = [[1.0, 0.0] for _ in range(frame_count)]

    constraints = [
        {
            "type": "root2d",
            "frame_indices": list(range(frame_count)),
            "smooth_root_2d": root_2d,
            "global_root_heading": heading,
        }
    ]
    write_json(path, constraints)


class DummyTextEncoder:
    def __init__(self, token_count: int = 50, dim: int = 4096) -> None:
        self.token_count = token_count
        self.dim = dim

    def __call__(self, texts: list[str]):
        import torch

        return torch.zeros((len(texts), self.token_count, self.dim), dtype=torch.float32), [0] * len(texts)


def run_generation_direct(args: argparse.Namespace, constraints_path: Path, output_stem: Path) -> tuple[str, Path]:
    import torch
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.model.registry import get_model_info

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    text_encoder = DummyTextEncoder() if args.use_dummy_text_encoder else None
    model, resolved_model = load_model(
        args.model,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
        text_encoder=text_encoder,
    )
    constraints = load_constraints_lst(str(constraints_path), model.skeleton)
    output = model(
        [args.prompt or stair_prompt(args.direction)],
        [int(round(args.duration * model.fps))],
        constraint_lst=constraints,
        num_denoising_steps=args.diffusion_steps,
        num_samples=1,
        multi_prompt=True,
        post_processing=False if "g1" in resolved_model else True,
        return_numpy=True,
        cfg_type="separated",
        cfg_weight=[0.0 if args.use_dummy_text_encoder else 2.0, 2.0],
    )
    single = {
        key: (value[0] if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == 1 else value)
        for key, value in output.items()
    }
    npz_path = output_stem.with_suffix(".npz")
    save_kimodo_npz(str(npz_path), single)

    info = get_model_info(resolved_model)
    skeleton_name = (info.skeleton if info else resolved_model).lower()
    if skeleton_name == "smplx":
        from kimodo.exports.smplx import AMASSConverter

        amass_path = output_stem.with_name(output_stem.name + "_amass").with_suffix(".npz")
        AMASSConverter(fps=model.fps, skeleton=model.skeleton).convert_save_npz(output, amass_path)
        return "amass", amass_path
    if skeleton_name == "g1":
        from kimodo.exports.mujoco import MujocoQposConverter

        csv_path = output_stem.with_suffix(".csv")
        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(output, device)
        converter.save_csv(qpos, csv_path)
        return "g1-csv", csv_path
    return "kimodo-npz", npz_path


def run_generation_cli(args: argparse.Namespace, constraints_path: Path, output_stem: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "kimodo.scripts.generate",
        args.prompt or stair_prompt(args.direction),
        "--model",
        args.model,
        "--duration",
        f"{args.duration:g}",
        "--num_samples",
        "1",
        "--diffusion_steps",
        str(args.diffusion_steps),
        "--seed",
        str(args.seed),
        "--constraints",
        str(constraints_path),
        "--output",
        str(output_stem),
    ]
    env = os.environ.copy()
    env.setdefault("TEXT_ENCODER_DEVICE", args.text_encoder_device)
    subprocess.run(cmd, check=True, env=env)
    amass_path = output_stem.with_name(output_stem.name + "_amass").with_suffix(".npz")
    if not amass_path.exists():
        raise FileNotFoundError(f"Kimodo did not create expected AMASS file: {amass_path}")
    return amass_path


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def warp_amass_to_regular_stairs(args: argparse.Namespace, source: Path, target: Path) -> Path:
    with np.load(source, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    trans = np.asarray(arrays["trans"], dtype=np.float64).copy()
    xy = trans[:, :2]
    delta = xy[-1] - xy[0]
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        direction = np.array([0.0, 1.0], dtype=np.float64)
    else:
        direction = delta / norm
    progress = ((xy - xy[0]) @ direction) / max(norm, 1e-6)
    progress = np.clip(progress, 0.0, 1.0)
    if args.direction == "down":
        progress = 1.0 - progress

    total_height = args.steps * args.step_height
    terrain_height = progress * total_height
    original_rise = np.linspace(trans[0, 2], trans[-1, 2], len(trans))
    vertical_bob = _smooth(trans[:, 2] - original_rise)
    if args.direction == "down":
        terrain_height = total_height - terrain_height
    trans[:, 2] = trans[0, 2] + terrain_height + 0.35 * vertical_bob

    arrays["trans"] = trans.astype(np.float32)
    if "mocap_framerate" not in arrays and "mocap_frame_rate" in arrays:
        arrays["mocap_framerate"] = np.asarray(arrays["mocap_frame_rate"])
    if "poses" not in arrays and "root_orient" in arrays and "pose_body" in arrays:
        arrays["poses"] = np.concatenate(
            [
                np.asarray(arrays["root_orient"], dtype=np.float32),
                np.asarray(arrays["pose_body"], dtype=np.float32),
            ],
            axis=1,
        )
    arrays["kimodo_stair_direction"] = np.asarray(args.direction)
    arrays["kimodo_stair_steps"] = np.asarray(args.steps, dtype=np.int32)
    arrays["kimodo_stair_step_height"] = np.asarray(args.step_height, dtype=np.float32)
    arrays["kimodo_stair_tread_depth"] = np.asarray(args.tread_depth, dtype=np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    return target


def convert_to_humenv(args: argparse.Namespace, amass_path: Path, output_hdf5: Path) -> None:
    temp_npz = output_hdf5.with_suffix(".smplsim_qpos.npz")
    export_cmd = [
        str(args.smplsim_python),
        str(args.convert_script),
        "--stage",
        "export-qpos",
        "--amass-source",
        str(amass_path),
        "--temp-npz",
        str(temp_npz),
    ]
    subprocess.run(export_cmd, check=True)

    import importlib.util

    spec = importlib.util.spec_from_file_location("humenv_amass_converter", args.convert_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {args.convert_script}")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    data = np.load(temp_npz, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    fps = float(data["fps"])
    qvel = converter.differentiate_qpos(qpos, 1.0 / fps)
    obs = converter.compute_observations(qpos, qvel)
    writer_args = SimpleNamespace(output=output_hdf5, motion_id=-1)
    converter.write_hdf5(writer_args, qpos, qvel, obs, fps, str(amass_path))
    converter.update_manifest(SimpleNamespace(output=output_hdf5, amass_source=amass_path, motion_id=-1), len(qpos))


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.step_height <= 0 or args.tread_depth <= 0 or args.duration <= 0:
        raise ValueError("--step-height, --tread-depth, and --duration must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    constraints_path = args.constraints or (args.out_dir / f"{args.name}_constraints.json")
    output_stem = args.out_dir / args.name
    generated_amass = output_stem.with_name(output_stem.name + "_amass").with_suffix(".npz")
    warped_amass = args.out_dir / f"{args.name}_regular_stairs_amass.npz"
    output_hdf5 = args.out_dir / f"{args.name}.hdf5"
    generated_kind = "amass"
    generated_path: Path = generated_amass

    if args.constraints is None:
        build_root_constraints(args, constraints_path)
    elif not constraints_path.exists():
        raise FileNotFoundError(f"--constraints file not found: {constraints_path}")
    if not args.skip_generate:
        generated_kind, generated_path = run_generation_direct(args, constraints_path, output_stem)
        if generated_kind == "amass":
            generated_amass = generated_path
    elif not generated_amass.exists():
        raise FileNotFoundError(f"--skip-generate requested but missing {generated_amass}")

    if generated_kind == "amass":
        warp_amass_to_regular_stairs(args, generated_amass, warped_amass)
        if not args.skip_convert:
            convert_to_humenv(args, warped_amass, output_hdf5)
    else:
        print(f"[WARN] Generated {generated_kind}; HumEnv conversion requires SMPLX/AMASS output.")

    meta = {
        "constraints": str(constraints_path),
        "generated_kind": generated_kind,
        "generated_path": str(generated_path),
        "generated_amass": str(generated_amass) if generated_kind == "amass" else None,
        "warped_amass": str(warped_amass) if generated_kind == "amass" else None,
        "humenv_hdf5": str(output_hdf5) if generated_kind == "amass" else None,
        "direction": args.direction,
        "steps": args.steps,
        "step_height": args.step_height,
        "tread_depth": args.tread_depth,
        "duration": args.duration,
        "prompt": args.prompt or stair_prompt(args.direction),
    }
    write_json(args.out_dir / f"{args.name}_manifest.json", meta)
    print(f"[INFO] constraints: {constraints_path}")
    print(f"[INFO] generated {generated_kind}: {generated_path}")
    if generated_kind == "amass":
        print(f"[INFO] generated AMASS: {generated_amass}")
        print(f"[INFO] stair-warped AMASS: {warped_amass}")
        print(f"[INFO] HumEnv HDF5: {output_hdf5}")
    if output_hdf5.exists():
        print(
            "[INFO] Preview with: "
            f"MOCAP_MOTION={output_hdf5} STAIR_STEPS={args.steps} "
            f"STAIR_HEIGHT={args.step_height:g} ./show_mocap_track_updown_stairs.sh"
        )


if __name__ == "__main__":
    main()
