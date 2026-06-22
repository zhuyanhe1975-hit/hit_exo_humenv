#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "test_mocap_track_updown_stairs.sh"


CONFIGS: dict[str, dict[str, str]] = {
    "assist060": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "1",
        "ROOT_ORIENTATION_TRACK": "1",
        "JOINT_POSE_TRACK_GAIN": "0.60",
        "JOINT_VELOCITY_TRACK_GAIN": "0.60",
    },
    "assist045": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "1",
        "ROOT_ORIENTATION_TRACK": "1",
        "JOINT_POSE_TRACK_GAIN": "0.45",
        "JOINT_VELOCITY_TRACK_GAIN": "0.45",
    },
    "assist035": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "1",
        "ROOT_ORIENTATION_TRACK": "1",
        "JOINT_POSE_TRACK_GAIN": "0.35",
        "JOINT_VELOCITY_TRACK_GAIN": "0.35",
    },
    "assist025": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "1",
        "ROOT_ORIENTATION_TRACK": "1",
        "JOINT_POSE_TRACK_GAIN": "0.25",
        "JOINT_VELOCITY_TRACK_GAIN": "0.25",
    },
    "assist015": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "1",
        "ROOT_ORIENTATION_TRACK": "1",
        "JOINT_POSE_TRACK_GAIN": "0.15",
        "JOINT_VELOCITY_TRACK_GAIN": "0.15",
    },
    "root_xyz_ori": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "1",
        "ROOT_ORIENTATION_TRACK": "1",
        "JOINT_POSE_TRACK_GAIN": "0.0",
        "JOINT_VELOCITY_TRACK_GAIN": "0.0",
    },
    "rootxy_ori_joint035_noz": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "0",
        "ROOT_ORIENTATION_TRACK": "1",
        "JOINT_POSE_TRACK_GAIN": "0.35",
        "JOINT_VELOCITY_TRACK_GAIN": "0.35",
    },
    "rootxy": {
        "ROOT_XY_TRACK": "1",
        "ROOT_Z_TRACK": "0",
        "ROOT_ORIENTATION_TRACK": "0",
        "JOINT_POSE_TRACK_GAIN": "0.0",
        "JOINT_VELOCITY_TRACK_GAIN": "0.0",
    },
    "none": {
        "ROOT_XY_TRACK": "0",
        "ROOT_Z_TRACK": "0",
        "ROOT_ORIENTATION_TRACK": "0",
        "JOINT_POSE_TRACK_GAIN": "0.0",
        "JOINT_VELOCITY_TRACK_GAIN": "0.0",
    },
}


SUMMARY_FIELDS = [
    "config",
    "state",
    "success",
    "reason",
    "touchdowns",
    "wrong_terrain",
    "first_error_m",
    "first_xy_error_m",
    "first_clearance_error_m",
    "first_target_terrain",
    "first_sim_terrain",
    "max_landing_error_m",
    "mean_landing_error_m",
    "max_clearance_error_m",
    "foot_mean_error_m",
    "foot_max_error_m",
    "foot_xy_mean_error_m",
    "root_mean_error_m",
    "root_max_error_m",
    "root_final_error_m",
    "out_dir",
    "log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep stair mocap tracking assist gains and summarize where assisted climbing breaks."
    )
    parser.add_argument("--out-root", type=Path, default=Path(".omx/stair_assist_sweep"))
    parser.add_argument("--duration", default="0")
    parser.add_argument("--repeat", default="1")
    parser.add_argument("--start-frame", default="0")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--direction", default=None)
    parser.add_argument("--terrain", default=None)
    parser.add_argument("--motion", type=Path, default=None)
    parser.add_argument("--episode", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS.keys(), default=list(CONFIGS.keys()))
    parser.add_argument("--success-max-landing-error", type=float, default=0.10)
    parser.add_argument("--success-max-clearance-error", type=float, default=0.06)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def summarize_run(name: str, out_dir: Path, returncode: int) -> dict[str, object]:
    landing_rows = _read_csv(out_dir / "foot_landings.csv")
    foot_rows = _read_csv(out_dir / "foot_tracking.csv")
    rollout_rows = _read_csv(out_dir / "rollout.csv")
    status_path = out_dir / "status.json"
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))

    touchdowns = [row for row in landing_rows if row.get("phase") == "rollout" and row.get("event_type") == "touchdown"]
    first = touchdowns[0] if touchdowns else {}
    wrong_terrain = [
        row
        for row in touchdowns
        if row.get("sim_terrain_name") != row.get("mocap_terrain_name")
    ]

    landing_errors = [_float(row, "landing_error") for row in touchdowns]
    clearance_errors = [_float(row, "center_clearance_error") for row in touchdowns]
    rollout_foot_rows = [row for row in foot_rows if row.get("phase") == "rollout"]
    foot_errors = [_float(row, "center_error") for row in rollout_foot_rows]
    foot_xy_errors = [_float(row, "center_xy_error") for row in rollout_foot_rows]
    root_errors = [_float(row, "root_error") for row in rollout_rows]

    success = (
        returncode == 0
        and bool(touchdowns)
        and not wrong_terrain
        and max(landing_errors or [999.0]) <= summarize_run.max_landing_error
        and max(clearance_errors or [999.0]) <= summarize_run.max_clearance_error
    )
    reason = "ok"
    if returncode != 0:
        reason = f"runner_exit_{returncode}"
    elif not touchdowns:
        reason = "no_touchdowns_logged"
    elif wrong_terrain:
        reason = "wrong_terrain"
    elif max(landing_errors or [0.0]) > summarize_run.max_landing_error:
        reason = "landing_error_high"
    elif max(clearance_errors or [0.0]) > summarize_run.max_clearance_error:
        reason = "clearance_error_high"

    return {
        "config": name,
        "state": status.get("state", "missing_status"),
        "success": success,
        "reason": reason,
        "touchdowns": len(touchdowns),
        "wrong_terrain": len(wrong_terrain),
        "first_error_m": _float(first, "landing_error"),
        "first_xy_error_m": _float(first, "landing_xy_error"),
        "first_clearance_error_m": _float(first, "center_clearance_error"),
        "first_target_terrain": first.get("mocap_terrain_name", ""),
        "first_sim_terrain": first.get("sim_terrain_name", ""),
        "max_landing_error_m": max(landing_errors or [0.0]),
        "mean_landing_error_m": _mean(landing_errors),
        "max_clearance_error_m": max(clearance_errors or [0.0]),
        "foot_mean_error_m": _mean(foot_errors),
        "foot_max_error_m": max(foot_errors or [0.0]),
        "foot_xy_mean_error_m": _mean(foot_xy_errors),
        "root_mean_error_m": _mean(root_errors),
        "root_max_error_m": max(root_errors or [0.0]),
        "root_final_error_m": root_errors[-1] if root_errors else 0.0,
        "out_dir": str(out_dir),
        "log": str(out_dir / "logs" / f"{name}.log"),
    }


def write_summary(out_root: Path, rows: list[dict[str, object]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_root / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    md_path = out_root / "report.md"
    lines = [
        "# Stair Assist Sweep",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "",
        "| config | success | reason | first err | max landing | max clearance | wrong terrain | foot mean | root final |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {config} | {success} | {reason} | {first:.3f} | {max_land:.3f} | {max_clear:.3f} | {wrong} | {foot:.3f} | {root:.3f} |".format(
                config=row["config"],
                success="yes" if row["success"] else "no",
                reason=row["reason"],
                first=float(row["first_error_m"]),
                max_land=float(row["max_landing_error_m"]),
                max_clear=float(row["max_clearance_error_m"]),
                wrong=row["wrong_terrain"],
                foot=float(row["foot_mean_error_m"]),
                root=float(row["root_final_error_m"]),
            )
        )
    lines.extend(["", f"- CSV: `{csv_path}`", f"- JSON: `{json_path}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote sweep summary: {csv_path}")
    print(f"[INFO] Wrote sweep report: {md_path}")


def run_config(args: argparse.Namespace, name: str, overrides: dict[str, str]) -> dict[str, object]:
    out_dir = args.out_root / name
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": name,
            "OUT_DIR": str(out_dir),
            "KINEMATIC_PREFLIGHT": "0",
            "REPEAT": str(args.repeat),
            "DURATION": str(args.duration),
            "START_FRAME": str(args.start_frame),
            "ASSIST_PRESET": "none",
        }
    )
    env.update(overrides)
    optional_env = {
        "DATASET": args.dataset,
        "DIRECTION": args.direction,
        "TERRAIN": args.terrain,
        "MOCAP_MOTION": str(args.motion) if args.motion is not None else None,
        "MOCAP_EPISODE": args.episode,
        "DEVICE": args.device,
    }
    env.update({key: value for key, value in optional_env.items() if value})

    cmd = [str(RUNNER), "--headless"]
    printable = " ".join(f"{key}={env[key]}" for key in sorted(overrides)) + " " + " ".join(cmd)
    print(f"[INFO] Running {name}: {printable}")
    if args.dry_run:
        row = {field: "" for field in SUMMARY_FIELDS}
        row.update(
            {
                "config": name,
                "state": "dry_run",
                "success": False,
                "reason": "dry_run",
                "touchdowns": 0,
                "wrong_terrain": 0,
                "out_dir": str(out_dir),
                "log": str(out_dir / "logs" / f"{name}.log"),
            }
        )
        for field in SUMMARY_FIELDS:
            if field.endswith("_m"):
                row[field] = 0.0
        return row

    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True)
    return summarize_run(name, out_dir, result.returncode)


def main() -> None:
    args = parse_args()
    if not RUNNER.exists():
        raise FileNotFoundError(f"Runner not found: {RUNNER}")
    summarize_run.max_landing_error = args.success_max_landing_error
    summarize_run.max_clearance_error = args.success_max_clearance_error

    rows = [run_config(args, name, CONFIGS[name]) for name in args.configs]
    write_summary(args.out_root, rows)
    print("[INFO] Sweep complete.")
    for row in rows:
        print(
            "[INFO] {config}: success={success} reason={reason} "
            "first={first:.3f} max_landing={max_land:.3f} max_clearance={max_clear:.3f} wrong_terrain={wrong}".format(
                config=row["config"],
                success=row.get("success"),
                reason=row.get("reason"),
                first=float(row.get("first_error_m", 0.0)),
                max_land=float(row.get("max_landing_error_m", 0.0)),
                max_clear=float(row.get("max_clearance_error_m", 0.0)),
                wrong=row.get("wrong_terrain", 0),
            )
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
