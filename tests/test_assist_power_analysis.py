from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_assist_power import assisted_indicators, compare, headline, summarize


def _write_rows(path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_assist_power_summary_and_reduction_ratio(tmp_path) -> None:
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    _write_rows(
        before,
        [
            {
                "step": 1,
                "time_s": 0.1,
                "left_knee_pd_torque_nm": 10.0,
                "right_knee_pd_torque_nm": -8.0,
                "left_knee_exo_torque_nm": 0.0,
                "right_knee_exo_torque_nm": 0.0,
                "left_knee_joint_vel_rad_s": 2.0,
                "right_knee_joint_vel_rad_s": -1.0,
                "fallen_env0": 0,
                "done_env0": 0,
            },
            {
                "step": 2,
                "time_s": 0.2,
                "left_knee_pd_torque_nm": 6.0,
                "right_knee_pd_torque_nm": -4.0,
                "left_knee_exo_torque_nm": 0.0,
                "right_knee_exo_torque_nm": 0.0,
                "left_knee_joint_vel_rad_s": 1.0,
                "right_knee_joint_vel_rad_s": -2.0,
                "fallen_env0": 0,
                "done_env0": 0,
            },
        ],
    )
    _write_rows(
        after,
        [
            {
                "step": 1,
                "time_s": 0.1,
                "left_knee_pd_torque_nm": 5.0,
                "right_knee_pd_torque_nm": -4.0,
                "left_knee_exo_torque_nm": 3.0,
                "right_knee_exo_torque_nm": -2.0,
                "left_knee_joint_vel_rad_s": 2.0,
                "right_knee_joint_vel_rad_s": -1.0,
                "fallen_env0": 0,
                "done_env0": 0,
            },
            {
                "step": 2,
                "time_s": 0.2,
                "left_knee_pd_torque_nm": 3.0,
                "right_knee_pd_torque_nm": -2.0,
                "left_knee_exo_torque_nm": 1.0,
                "right_knee_exo_torque_nm": -1.0,
                "left_knee_joint_vel_rad_s": 1.0,
                "right_knee_joint_vel_rad_s": -2.0,
                "fallen_env0": 0,
                "done_env0": 0,
            },
        ],
    )

    before_summary = summarize(before, drop_fallen=True)
    after_summary = summarize(after, drop_fallen=True)
    comparison = compare(before_summary, after_summary)

    assert before_summary.human_knee_abs_power_mean_w == 21.0
    assert after_summary.human_knee_abs_power_mean_w == 10.5
    assert after_summary.exo_knee_abs_power_mean_w == 5.5
    assert comparison["human_knee_abs_power_reduction_w"] == 10.5
    assert comparison["human_knee_abs_power_reduction_ratio"] == 0.5
    assert comparison["human_knee_abs_power_reduction_percent"] == 50.0
    assert comparison["assist_efficiency_human_saved_per_exo_abs_power"] == pytest.approx(
        10.5 / 5.5
    )
    assert comparison["net_system_input_power_change_w"] == -5.0

    indicators = assisted_indicators(after_summary)
    assert indicators["exo_to_human_abs_power_ratio"] == pytest.approx(5.5 / 10.5)
    assert indicators["exo_abs_power_share_of_separated_total_percent"] == pytest.approx(100 * 5.5 / 16.0)

    report = headline(after_summary, before_summary)
    assert report["baseline_status"] == "available"
    assert report["human_abs_power_saved_percent"] == 50.0
    assert report["assist_efficiency_human_saved_per_exo_abs_power"] == pytest.approx(10.5 / 5.5)


def test_assist_power_summary_rejects_logs_without_velocity(tmp_path) -> None:
    old_log = tmp_path / "old.csv"
    _write_rows(
        old_log,
        [
            {
                "step": 1,
                "time_s": 0.1,
                "left_knee_pd_torque_nm": 10.0,
                "right_knee_pd_torque_nm": -8.0,
                "left_knee_exo_torque_nm": 0.0,
                "right_knee_exo_torque_nm": 0.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="knee_joint_vel"):
        summarize(old_log, drop_fallen=True)


def test_assist_power_summary_prefers_lower_limb_fields(tmp_path) -> None:
    log = tmp_path / "lower_limb.csv"
    _write_rows(
        log,
        [
            {
                "step": 1,
                "time_s": 0.1,
                "human_lower_limb_abs_power_w": 100.0,
                "human_lower_limb_signed_power_w": 80.0,
                "combined_lower_limb_abs_power_w": 110.0,
                "combined_lower_limb_signed_power_w": 85.0,
                "human_knee_abs_power_w": 10.0,
                "human_knee_signed_power_w": 8.0,
                "combined_knee_abs_power_w": 11.0,
                "combined_knee_signed_power_w": 8.5,
                "exo_knee_abs_power_w": 20.0,
                "exo_knee_signed_power_w": 5.0,
                "fallen_env0": 0,
                "done_env0": 0,
            }
        ],
    )

    summary = summarize(log, drop_fallen=True)

    assert summary.human_knee_abs_power_mean_w == 100.0
    assert summary.combined_knee_abs_power_mean_w == 110.0
    assert summary.exo_knee_abs_power_mean_w == 20.0
    assert summary.human_lower_limb_mechanical_abs_power_mean_w == 100.0
    assert summary.human_lower_limb_metabolic_proxy_mean_w is None


def test_assist_power_summary_prefers_metabolic_fields(tmp_path) -> None:
    log = tmp_path / "metabolic.csv"
    _write_rows(
        log,
        [
            {
                "step": 1,
                "time_s": 0.1,
                "human_lower_limb_metabolic_power_w": 300.0,
                "human_lower_limb_positive_power_w": 60.0,
                "human_lower_limb_negative_power_w": 20.0,
                "human_lower_limb_abs_power_w": 100.0,
                "human_lower_limb_signed_power_w": 40.0,
                "combined_lower_limb_abs_power_w": 110.0,
                "combined_lower_limb_signed_power_w": 45.0,
                "exo_knee_abs_power_w": 20.0,
                "exo_knee_signed_power_w": 5.0,
                "fallen_env0": 0,
                "done_env0": 0,
            }
        ],
    )

    summary = summarize(log, drop_fallen=True)

    assert summary.human_knee_abs_power_mean_w == 300.0
    assert summary.human_knee_signed_power_mean_w == 40.0
    assert summary.human_lower_limb_mechanical_abs_power_mean_w == 100.0
    assert summary.human_lower_limb_metabolic_proxy_mean_w == 300.0
