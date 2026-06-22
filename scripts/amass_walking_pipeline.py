#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DEFAULT_HUMENV_ROOT = Path("/home/yhzhu/AI/humenv")
DEFAULT_DATA_PREP = DEFAULT_HUMENV_ROOT / "data_preparation"
DEFAULT_AMASS_ROOT = DEFAULT_DATA_PREP / "AMASS"
DEFAULT_DATASETS = DEFAULT_AMASS_ROOT / "datasets"
DEFAULT_SUBSET = DEFAULT_AMASS_ROOT / "walking_forward_subset"
DEFAULT_OUTPUT = DEFAULT_DATA_PREP / "humenv_amass_walking_forward"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "walking_manifest.csv"

RECOMMENDED_ARCHIVES = (
    "KIT.tar.bz2",
    "CMU.tar.bz2",
    "BMLmovi.tar.bz2",
    "BMLrub.tar.bz2",
    "MPI_HDM05.tar.bz2",
    "Transitions.tar.bz2",
)

DATASET_NAME_ALIASES = {
    "KIT": ("KIT",),
    "CMU": ("CMU",),
    "BMLmovi": ("BMLmovi",),
    "BMLrub": ("BMLrub",),
    "MPI_HDM05": ("MPI_HDM05", "MPI_HDM05"),
    "Transitions": ("Transitions_mocap", "Transitions"),
}

WALK_RE = re.compile(r"(?:^|[_\-\s])(walk|walking|locomotion|stride|stroll)(?:[_\-\s]|\d|$)", re.I)
EXCLUDE_RE = re.compile(
    r"(backward|backwards|sideway|sideways|turn|twist|run|jog|jump|hop|sit|stand|dance|"
    r"stairs|upstairs|downstairs|crawl|crouch|kick|punch|throw)",
    re.I,
)


@dataclass(frozen=True)
class MotionMetrics:
    file: Path
    frames: int
    duration_s: float
    root_dx: float
    root_dy: float
    root_dz: float
    xy_distance: float
    xy_speed: float
    z_min: float
    z_mean: float
    z_max: float
    accepted: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a curated forward-walking HumEnv HDF5 set from manually downloaded "
            "AMASS SMPL-H archives. Download/login remains manual because AMASS requires "
            "accepted license terms."
        )
    )
    parser.add_argument(
        "stage",
        choices=("check", "extract", "select", "process", "validate", "all"),
        help="Pipeline stage to run.",
    )
    parser.add_argument("--data-prep", type=Path, default=DEFAULT_DATA_PREP)
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--copy", action="store_true", help="Copy selected NPZ files instead of symlinking.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit selected files for a smoke run.")
    parser.add_argument("--min-frames", type=int, default=90)
    parser.add_argument("--min-distance", type=float, default=0.5)
    parser.add_argument("--min-speed", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.8)
    parser.add_argument("--min-root-z", type=float, default=0.65)
    return parser.parse_args()


def archive_candidates(datasets_root: Path) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {}
    for archive in RECOMMENDED_ARCHIVES:
        matches = [
            datasets_root / archive,
            datasets_root.parent / archive,
            datasets_root.parent / "archives" / archive,
        ]
        result[archive] = next((path for path in matches if path.exists()), None)
    return result


def check(args: argparse.Namespace) -> None:
    print("[INFO] Recommended AMASS archives for this project:")
    print("       KIT, CMU, BMLmovi, BMLrub, MPI_HDM05, Transitions")
    print("[INFO] Download format: SMPL-H / gender based, .tar.bz2 archives")
    print(f"[INFO] Put archives under: {args.datasets}  or  {args.datasets.parent / 'archives'}")
    print(f"[INFO] Expected SMPL models after humenv setup: {args.data_prep / 'AMASS/models'}")
    print("")
    for archive, path in archive_candidates(args.datasets).items():
        status = "OK" if path else "MISSING"
        print(f"{status:7s} {archive}" + (f" -> {path}" if path else ""))
    if not (args.data_prep / "AMASS/models").exists():
        print("")
        print("[WARN] Missing AMASS/models. Follow humenv data_preparation README to install SMPL models.")
    for required in ("PHC", "SMPLSim", "process_amass.py"):
        if not (args.data_prep / required).exists():
            print(f"[WARN] Missing {args.data_prep / required}; run humenv data_preparation/preprocess_dataset.sh setup first.")


def extract(args: argparse.Namespace) -> None:
    args.datasets.mkdir(parents=True, exist_ok=True)
    archives = archive_candidates(args.datasets)
    found = [path for path in archives.values() if path is not None]
    if not found:
        raise FileNotFoundError(f"No recommended AMASS archives found under {args.datasets} or {args.datasets.parent / 'archives'}")
    for archive in found:
        assert archive is not None
        print(f"[INFO] Extracting {archive} -> {args.datasets}")
        with tarfile.open(archive) as tf:
            tf.extractall(args.datasets)


def dataset_dirs(datasets_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for aliases in DATASET_NAME_ALIASES.values():
        for alias in aliases:
            path = datasets_root / alias
            if path.is_dir():
                dirs.append(path)
                break
    return dirs


def is_walk_candidate(path: Path) -> bool:
    text = str(path).replace(os.sep, "_")
    if path.name.endswith("shape.npz"):
        return False
    return bool(WALK_RE.search(text)) and not EXCLUDE_RE.search(text)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def select(args: argparse.Namespace) -> None:
    dirs = dataset_dirs(args.datasets)
    if not dirs:
        raise FileNotFoundError(f"No recommended AMASS dataset folders found under {args.datasets}")
    if args.subset.exists():
        shutil.rmtree(args.subset)
    args.subset.mkdir(parents=True)

    selected: list[Path] = []
    for dataset_dir in dirs:
        for src in sorted(dataset_dir.rglob("*.npz")):
            if is_walk_candidate(src):
                rel = src.relative_to(args.datasets)
                dst = args.subset / rel
                link_or_copy(src, dst, args.copy)
                selected.append(src)
                if args.max_files > 0 and len(selected) >= args.max_files:
                    break
        if args.max_files > 0 and len(selected) >= args.max_files:
            break
    print(f"[INFO] Selected {len(selected)} candidate walking NPZ files into {args.subset}")
    for path in selected[:30]:
        print(f"       {path.relative_to(args.datasets)}")
    if not selected:
        print("[WARN] No candidates found. KIT usually has descriptive walking_* names; check archive extraction.")


def process(args: argparse.Namespace) -> None:
    if not args.subset.exists():
        raise FileNotFoundError(f"Subset folder does not exist: {args.subset}")
    process_script = args.data_prep / "process_amass.py"
    if not process_script.exists():
        raise FileNotFoundError(f"Missing humenv process script: {process_script}")
    cwd = Path.cwd()
    sys.path.insert(0, str(args.data_prep))
    os.chdir(args.data_prep)
    try:
        import process_amass  # type: ignore

        seq_pkl = args.output.parent / "amass_walking_seq_data.pkl"
        mapper_json = args.output.parent / "amass_walking_motion_name_mapper.json"
        print(f"[INFO] Filtering AMASS subset: {args.subset}")
        amass_seq_data, motion_name_mapped = process_amass._filter(
            str(args.subset),
            phc_files_to_remove=process_amass.PHC_FILES_TO_REMOVE,
        )
        import joblib
        import json

        joblib.dump(amass_seq_data, seq_pkl)
        with mapper_json.open("w") as f:
            json.dump(motion_name_mapped, f)
        print(f"[INFO] Converting to HumEnv HDF5: {args.output}")
        process_amass._hdf5_step(
            str(seq_pkl),
            str(mapper_json),
            num_workers=args.num_workers,
            output_dir=str(args.output),
        )
    finally:
        os.chdir(cwd)


def load_hdf5_metrics(path: Path, args: argparse.Namespace) -> MotionMetrics:
    with h5py.File(path, "r") as hf:
        ep = hf["ep_0"]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float64)
        dt = float(ep.attrs.get("dt", hf.attrs.get("dt", 1.0 / 30.0)))
    frames = int(qpos.shape[0])
    duration_s = max((frames - 1) * dt, dt)
    root = qpos[:, :3]
    delta = root[-1] - root[0]
    xy_distance = float(np.linalg.norm(delta[:2]))
    xy_speed = xy_distance / duration_s
    z_min = float(root[:, 2].min())
    z_mean = float(root[:, 2].mean())
    z_max = float(root[:, 2].max())

    reasons = []
    if frames < args.min_frames:
        reasons.append("short")
    if xy_distance < args.min_distance:
        reasons.append("low_distance")
    if xy_speed < args.min_speed:
        reasons.append("slow")
    if xy_speed > args.max_speed:
        reasons.append("fast")
    if z_min < args.min_root_z:
        reasons.append("low_root")
    accepted = not reasons
    return MotionMetrics(
        file=path,
        frames=frames,
        duration_s=duration_s,
        root_dx=float(delta[0]),
        root_dy=float(delta[1]),
        root_dz=float(delta[2]),
        xy_distance=xy_distance,
        xy_speed=xy_speed,
        z_min=z_min,
        z_mean=z_mean,
        z_max=z_max,
        accepted=accepted,
        reason="ok" if accepted else ";".join(reasons),
    )


def validate(args: argparse.Namespace) -> None:
    files = sorted(args.output.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.output}")
    metrics = [load_hdf5_metrics(path, args) for path in files]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MotionMetrics.__dataclass_fields__.keys()))
        writer.writeheader()
        for item in sorted(metrics, key=lambda x: (not x.accepted, -x.xy_distance)):
            row = item.__dict__.copy()
            row["file"] = str(item.file)
            writer.writerow(row)
    accepted = [item for item in metrics if item.accepted]
    print(f"[INFO] Validated {len(metrics)} HDF5 motions. Accepted {len(accepted)}.")
    print(f"[INFO] Manifest: {args.manifest}")
    for item in sorted(accepted, key=lambda x: -x.xy_distance)[:20]:
        print(
            f"       {item.file.name}: frames={item.frames} "
            f"duration={item.duration_s:.2f}s xy_speed={item.xy_speed:.3f} "
            f"z=({item.z_min:.3f},{item.z_mean:.3f},{item.z_max:.3f})"
        )


def main() -> None:
    args = parse_args()
    if args.stage in ("check", "all"):
        check(args)
    if args.stage in ("extract", "all"):
        extract(args)
    if args.stage in ("select", "all"):
        select(args)
    if args.stage in ("process", "all"):
        process(args)
    if args.stage in ("validate", "all"):
        validate(args)


if __name__ == "__main__":
    main()
