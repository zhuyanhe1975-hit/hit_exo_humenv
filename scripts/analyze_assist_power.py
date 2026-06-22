from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


POWER_COLUMNS = (
    "human_knee_abs_power_w",
    "human_knee_signed_power_w",
    "exo_knee_abs_power_w",
    "exo_knee_signed_power_w",
    "combined_knee_abs_power_w",
    "combined_knee_signed_power_w",
)
LOWER_LIMB_POWER_COLUMNS = (
    "human_lower_limb_abs_power_w",
    "human_lower_limb_signed_power_w",
    "exo_knee_abs_power_w",
    "exo_knee_signed_power_w",
    "combined_lower_limb_abs_power_w",
    "combined_lower_limb_signed_power_w",
)
METABOLIC_POWER_COLUMNS = (
    "human_lower_limb_metabolic_power_w",
    "human_lower_limb_positive_power_w",
    "human_lower_limb_negative_power_w",
    "combined_lower_limb_abs_power_w",
    "combined_lower_limb_signed_power_w",
)
RAW_COLUMNS = (
    "left_knee_pd_torque_nm",
    "right_knee_pd_torque_nm",
    "left_knee_exo_torque_nm",
    "right_knee_exo_torque_nm",
    "left_knee_joint_vel_rad_s",
    "right_knee_joint_vel_rad_s",
)


@dataclass
class PowerSummary:
    source: str
    samples: int
    duration_s: float | None
    human_knee_abs_power_mean_w: float
    human_knee_abs_work_j: float | None
    human_knee_signed_power_mean_w: float
    human_knee_positive_power_mean_w: float
    human_knee_negative_power_mean_w: float
    exo_knee_abs_power_mean_w: float
    exo_knee_abs_work_j: float | None
    exo_knee_signed_power_mean_w: float
    exo_knee_positive_power_mean_w: float
    exo_knee_negative_power_mean_w: float
    combined_knee_abs_power_mean_w: float
    combined_knee_abs_work_j: float | None
    combined_knee_signed_power_mean_w: float
    combined_knee_positive_power_mean_w: float
    combined_knee_negative_power_mean_w: float
    human_lower_limb_mechanical_abs_power_mean_w: float | None = None
    human_lower_limb_metabolic_proxy_mean_w: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze knee-assist power from run CSV logs. Logs produced by the current "
            "run_mjlab_knee_exo_viewer.py include the required knee velocity and power fields."
        )
    )
    parser.add_argument("after", type=Path, help="CSV log for the assisted/exoskeleton run.")
    parser.add_argument(
        "--before",
        type=Path,
        default=None,
        help="Optional CSV log for the unassisted baseline run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to printing JSON to stdout.",
    )
    parser.add_argument(
        "--drop-fallen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with fallen_env0=1 or done_env0=1 before summarizing.",
    )
    return parser.parse_args()


def read_rows(path: Path, *, drop_fallen: bool) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            if drop_fallen and (_truthy(raw.get("fallen_env0")) or _truthy(raw.get("done_env0"))):
                continue
            rows.append(_row_with_power(raw, path))
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return rows


def summarize(path: Path, *, drop_fallen: bool) -> PowerSummary:
    rows = read_rows(path, drop_fallen=drop_fallen)
    dt = _infer_dt(rows)

    human_signed = [row["human_knee_signed_power_w"] for row in rows]
    exo_signed = [row["exo_knee_signed_power_w"] for row in rows]
    combined_signed = [row["combined_knee_signed_power_w"] for row in rows]

    return PowerSummary(
        source=str(path),
        samples=len(rows),
        duration_s=None if dt is None else dt * len(rows),
        human_knee_abs_power_mean_w=_mean(row["human_knee_abs_power_w"] for row in rows),
        human_knee_abs_work_j=_work(rows, "human_knee_abs_power_w", dt),
        human_knee_signed_power_mean_w=_mean(human_signed),
        human_knee_positive_power_mean_w=_mean(max(value, 0.0) for value in human_signed),
        human_knee_negative_power_mean_w=_mean(min(value, 0.0) for value in human_signed),
        exo_knee_abs_power_mean_w=_mean(row["exo_knee_abs_power_w"] for row in rows),
        exo_knee_abs_work_j=_work(rows, "exo_knee_abs_power_w", dt),
        exo_knee_signed_power_mean_w=_mean(exo_signed),
        exo_knee_positive_power_mean_w=_mean(max(value, 0.0) for value in exo_signed),
        exo_knee_negative_power_mean_w=_mean(min(value, 0.0) for value in exo_signed),
        combined_knee_abs_power_mean_w=_mean(row["combined_knee_abs_power_w"] for row in rows),
        combined_knee_abs_work_j=_work(rows, "combined_knee_abs_power_w", dt),
        combined_knee_signed_power_mean_w=_mean(combined_signed),
        combined_knee_positive_power_mean_w=_mean(max(value, 0.0) for value in combined_signed),
        combined_knee_negative_power_mean_w=_mean(min(value, 0.0) for value in combined_signed),
        human_lower_limb_mechanical_abs_power_mean_w=_optional_mean(
            rows,
            "human_lower_limb_mechanical_abs_power_w",
        ),
        human_lower_limb_metabolic_proxy_mean_w=_optional_mean(
            rows,
            "human_lower_limb_metabolic_proxy_w",
        ),
    )


def compare(before: PowerSummary, after: PowerSummary) -> dict[str, float | None]:
    baseline = before.human_knee_abs_power_mean_w
    assisted = after.human_knee_abs_power_mean_w
    reduction = baseline - assisted
    baseline_work = before.human_knee_abs_work_j
    assisted_work = after.human_knee_abs_work_j
    work_reduction = (
        None if baseline_work is None or assisted_work is None else baseline_work - assisted_work
    )
    after_system_input = after.human_knee_abs_power_mean_w + after.exo_knee_abs_power_mean_w
    before_system_input = before.human_knee_abs_power_mean_w
    system_input_change = after_system_input - before_system_input
    return {
        "human_abs_power_saved_w": reduction,
        "human_abs_power_saved_percent": _safe_ratio(reduction, baseline, scale=100.0),
        "human_abs_work_saved_j": work_reduction,
        "human_abs_work_saved_percent": None
        if work_reduction is None or baseline_work is None
        else _safe_ratio(work_reduction, baseline_work, scale=100.0),
        "assist_efficiency_human_saved_per_exo_abs_power": _safe_ratio(
            reduction,
            after.exo_knee_abs_power_mean_w,
        ),
        "exo_abs_power_per_human_saved_power": _safe_ratio(after.exo_knee_abs_power_mean_w, reduction),
        "net_system_input_power_change_w": system_input_change,
        "net_system_input_power_change_percent": _safe_ratio(
            system_input_change,
            before_system_input,
            scale=100.0,
        ),
        "combined_abs_power_change_w": after.combined_knee_abs_power_mean_w
        - before.combined_knee_abs_power_mean_w,
        "combined_abs_power_change_percent": _safe_ratio(
            after.combined_knee_abs_power_mean_w - before.combined_knee_abs_power_mean_w,
            before.combined_knee_abs_power_mean_w,
            scale=100.0,
        ),
        "legacy_human_knee_abs_power_reduction_w": reduction,
        "legacy_human_knee_abs_power_reduction_ratio": _safe_ratio(reduction, baseline),
        "legacy_human_knee_abs_power_reduction_percent": _safe_ratio(
            reduction,
            baseline,
            scale=100.0,
        ),
        "legacy_exo_abs_to_human_saving_ratio": _safe_ratio(
            after.exo_knee_abs_power_mean_w,
            reduction,
        ),
        "human_knee_abs_power_reduction_w": reduction,
        "human_knee_abs_power_reduction_ratio": _safe_ratio(reduction, baseline),
        "human_knee_abs_power_reduction_percent": _safe_ratio(
            reduction,
            baseline,
            scale=100.0,
        ),
        "exo_abs_to_human_saving_ratio": _safe_ratio(
            after.exo_knee_abs_power_mean_w,
            reduction,
        ),
    }


def assisted_indicators(summary: PowerSummary) -> dict[str, float | None]:
    human = summary.human_knee_abs_power_mean_w
    exo = summary.exo_knee_abs_power_mean_w
    combined = summary.combined_knee_abs_power_mean_w
    separated_total = human + exo
    return {
        "exo_to_human_abs_power_ratio": _safe_ratio(exo, human),
        "exo_abs_power_share_of_separated_total_percent": _safe_ratio(
            exo,
            separated_total,
            scale=100.0,
        ),
        "combined_vs_human_abs_power_delta_w": combined - human,
        "combined_vs_human_abs_power_delta_percent": _safe_ratio(
            combined - human,
            human,
            scale=100.0,
        ),
        "exo_signed_to_abs_power_ratio": _safe_ratio(summary.exo_knee_signed_power_mean_w, exo),
        "exo_negative_to_positive_power_ratio": _safe_ratio(
            -summary.exo_knee_negative_power_mean_w,
            summary.exo_knee_positive_power_mean_w,
        ),
        "human_negative_to_positive_power_ratio": _safe_ratio(
            -summary.human_knee_negative_power_mean_w,
            summary.human_knee_positive_power_mean_w,
        ),
    }


def headline(after: PowerSummary, before: PowerSummary | None = None) -> dict[str, float | str | None]:
    indicators = assisted_indicators(after)
    result: dict[str, float | str | None] = {
        "assisted_human_abs_power_w": after.human_knee_abs_power_mean_w,
        "assisted_exo_abs_power_w": after.exo_knee_abs_power_mean_w,
        "assisted_combined_abs_power_w": after.combined_knee_abs_power_mean_w,
        "exo_to_human_abs_power_ratio": indicators["exo_to_human_abs_power_ratio"],
    }
    if before is None:
        result["baseline_status"] = "missing_before_log"
        result["human_abs_power_saved_percent"] = None
        result["assist_efficiency_human_saved_per_exo_abs_power"] = None
        result["note"] = "Set BEFORE_LOG to compute human power reduction and assist efficiency."
        return result

    comparison = compare(before, after)
    result.update(
        {
            "baseline_status": "available",
            "baseline_human_abs_power_w": before.human_knee_abs_power_mean_w,
            "human_abs_power_saved_w": comparison["human_abs_power_saved_w"],
            "human_abs_power_saved_percent": comparison["human_abs_power_saved_percent"],
            "assist_efficiency_human_saved_per_exo_abs_power": comparison[
                "assist_efficiency_human_saved_per_exo_abs_power"
            ],
            "net_system_input_power_change_w": comparison["net_system_input_power_change_w"],
            "net_system_input_power_change_percent": comparison[
                "net_system_input_power_change_percent"
            ],
        }
    )
    return result


def _row_with_power(raw: dict[str, str], path: Path) -> dict[str, float]:
    if all(_has_value(raw, name) for name in METABOLIC_POWER_COLUMNS):
        row = {
            "human_knee_abs_power_w": _float(
                raw["human_lower_limb_metabolic_power_w"],
                "human_lower_limb_metabolic_power_w",
                path,
            ),
            "human_knee_signed_power_w": _float(
                raw["human_lower_limb_positive_power_w"],
                "human_lower_limb_positive_power_w",
                path,
            ) - _float(
                raw["human_lower_limb_negative_power_w"],
                "human_lower_limb_negative_power_w",
                path,
            ),
            "exo_knee_abs_power_w": _exo_power_value(raw, path, signed=False),
            "exo_knee_signed_power_w": _exo_power_value(raw, path, signed=True),
            "combined_knee_abs_power_w": _float(
                raw["combined_lower_limb_abs_power_w"],
                "combined_lower_limb_abs_power_w",
                path,
            ),
            "combined_knee_signed_power_w": _float(
                raw["combined_lower_limb_signed_power_w"],
                "combined_lower_limb_signed_power_w",
                path,
            ),
            "human_lower_limb_mechanical_abs_power_w": _float(
                raw["human_lower_limb_abs_power_w"],
                "human_lower_limb_abs_power_w",
                path,
            ),
            "human_lower_limb_metabolic_proxy_w": _float(
                raw["human_lower_limb_metabolic_power_w"],
                "human_lower_limb_metabolic_power_w",
                path,
            ),
        }
    elif all(_has_value(raw, name) for name in LOWER_LIMB_POWER_COLUMNS):
        row = {
            "human_knee_abs_power_w": _float(raw["human_lower_limb_abs_power_w"], "human_lower_limb_abs_power_w", path),
            "human_knee_signed_power_w": _float(raw["human_lower_limb_signed_power_w"], "human_lower_limb_signed_power_w", path),
            "exo_knee_abs_power_w": _exo_power_value(raw, path, signed=False),
            "exo_knee_signed_power_w": _exo_power_value(raw, path, signed=True),
            "combined_knee_abs_power_w": _float(raw["combined_lower_limb_abs_power_w"], "combined_lower_limb_abs_power_w", path),
            "combined_knee_signed_power_w": _float(raw["combined_lower_limb_signed_power_w"], "combined_lower_limb_signed_power_w", path),
            "human_lower_limb_mechanical_abs_power_w": _float(
                raw["human_lower_limb_abs_power_w"],
                "human_lower_limb_abs_power_w",
                path,
            ),
        }
    elif all(_has_value(raw, name) for name in POWER_COLUMNS):
        row = {name: _float(raw[name], name, path) for name in POWER_COLUMNS}
    elif all(_has_value(raw, name) for name in RAW_COLUMNS):
        row = _compute_power_columns(raw, path)
    else:
        missing_raw = [name for name in METABOLIC_POWER_COLUMNS if not _has_value(raw, name)]
        if len(missing_raw) == len(METABOLIC_POWER_COLUMNS):
            missing_raw = [name for name in LOWER_LIMB_POWER_COLUMNS if not _has_value(raw, name)]
        if len(missing_raw) == len(LOWER_LIMB_POWER_COLUMNS):
            missing_raw = [name for name in RAW_COLUMNS if not _has_value(raw, name)]
        missing_power = [name for name in POWER_COLUMNS if not _has_value(raw, name)]
        missing = missing_raw or missing_power
        raise ValueError(
            f"{path} does not contain enough data to compute power. Missing examples: "
            f"{', '.join(missing[:6])}. Re-run with the updated run_mjlab_knee_exo_viewer.py logger."
        )

    for optional in ("step", "time_s", "fallen_env0", "done_env0"):
        if _has_value(raw, optional):
            row[optional] = _float(raw[optional], optional, path)
    return row


def _compute_power_columns(raw: dict[str, str], path: Path) -> dict[str, float]:
    left_human = _float(raw["left_knee_pd_torque_nm"], "left_knee_pd_torque_nm", path)
    right_human = _float(raw["right_knee_pd_torque_nm"], "right_knee_pd_torque_nm", path)
    left_exo = _float(raw["left_knee_exo_torque_nm"], "left_knee_exo_torque_nm", path)
    right_exo = _float(raw["right_knee_exo_torque_nm"], "right_knee_exo_torque_nm", path)
    left_vel = _float(raw["left_knee_joint_vel_rad_s"], "left_knee_joint_vel_rad_s", path)
    right_vel = _float(raw["right_knee_joint_vel_rad_s"], "right_knee_joint_vel_rad_s", path)

    human = (left_human * left_vel, right_human * right_vel)
    exo = (left_exo * left_vel, right_exo * right_vel)
    combined = ((left_human + left_exo) * left_vel, (right_human + right_exo) * right_vel)
    return {
        "human_knee_abs_power_w": abs(human[0]) + abs(human[1]),
        "human_knee_signed_power_w": human[0] + human[1],
        "exo_knee_abs_power_w": abs(exo[0]) + abs(exo[1]),
        "exo_knee_signed_power_w": exo[0] + exo[1],
        "combined_knee_abs_power_w": abs(combined[0]) + abs(combined[1]),
        "combined_knee_signed_power_w": combined[0] + combined[1],
    }


def _exo_power_value(raw: dict[str, str], path: Path, *, signed: bool) -> float:
    lower_key = "exo_lower_limb_signed_power_w" if signed else "exo_lower_limb_abs_power_w"
    legacy_key = "exo_knee_signed_power_w" if signed else "exo_knee_abs_power_w"
    key = lower_key if _has_value(raw, lower_key) else legacy_key
    return _float(raw[key], key, path)


def _infer_dt(rows: list[dict[str, float]]) -> float | None:
    times = [row.get("time_s") for row in rows if "time_s" in row]
    if len(times) < 2:
        return None
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _work(rows: list[dict[str, float]], key: str, dt: float | None) -> float | None:
    if dt is None:
        return None
    return sum(row[key] for row in rows) * dt


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _optional_mean(rows: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    return _mean(values)


def _safe_ratio(numerator: float, denominator: float, *, scale: float = 1.0) -> float | None:
    if abs(denominator) < 1.0e-12:
        return None
    return scale * numerator / denominator


def _has_value(row: dict[str, str], key: str) -> bool:
    return key in row and row[key] not in {"", None}


def _truthy(value: str | None) -> bool:
    if value is None or value == "":
        return False
    return bool(int(float(value)))


def _float(value: str, field: str, path: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float in {path}, field {field!r}: {value!r}") from exc


def main() -> None:
    args = parse_args()
    after = summarize(args.after, drop_fallen=args.drop_fallen)
    before = summarize(args.before, drop_fallen=args.drop_fallen) if args.before is not None else None
    payload: dict[str, object] = {
        "headline": headline(after, before),
        "after": asdict(after),
        "after_indicators": assisted_indicators(after),
        "drop_fallen": args.drop_fallen,
        "definitions": {
            "human_knee_power": "primary human metric: passive-adjusted lower-limb hip+knee+ankle metabolic proxy when available: positive mechanical power / 0.25 plus negative mechanical power magnitude / 1.20 after canceling XML passive joint forces used for simulation stability; older logs fall back to lower-limb absolute mechanical power or knee-only absolute power",
            "exo_knee_power": "sum over both knees of exo_torque * knee_joint_velocity",
            "combined_knee_power": "primary combined metric: lower-limb hip+knee+ankle power with exo torque added at both knees when available; legacy knee-only logs fall back to both knees",
            "abs_power": "sum of absolute per-knee mechanical powers; used as consumption proxy",
            "signed_power": "net signed mechanical power; positive means output in joint velocity direction",
            "assist_efficiency_human_saved_per_exo_abs_power": (
                "human abs-power saved divided by exoskeleton abs-power. "
                "1.0 means each 1 W of exo mechanical output saved 1 W of human knee abs-power."
            ),
            "net_system_input_power_change": (
                "after human abs-power plus exo abs-power minus before human abs-power; "
                "negative is lower separated human+exo mechanical demand."
            ),
        },
    }
    if before is not None:
        payload["before"] = asdict(before)
        payload["before_indicators"] = assisted_indicators(before)
        payload["comparison"] = compare(before, after)

    text = json.dumps(payload, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"[INFO] Wrote power analysis: {args.output}")


if __name__ == "__main__":
    main()
