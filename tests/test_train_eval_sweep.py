import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_eval_sweep import Target, default_candidates, rank_rows, target_passed


TARGET = Target(
    min_saved_percent=20.0,
    min_efficiency=1.0,
    max_net_input_change_percent=0.0,
    max_total_fallen=0,
)


def test_target_requires_power_saving_efficiency_net_reduction_and_no_falls() -> None:
    row = {
        "human_abs_power_saved_percent": 24.0,
        "assist_efficiency_human_saved_per_exo_abs_power": 2.2,
        "net_system_input_power_change_percent": -8.0,
        "total_fallen": 0,
    }

    assert target_passed(row, TARGET) is True

    row["net_system_input_power_change_percent"] = 3.0
    assert target_passed(row, TARGET) is False


def test_rank_rows_prefers_passing_then_more_saving_and_efficiency() -> None:
    rows = [
        {
            "candidate": "unstable_high_saving",
            "human_abs_power_saved_percent": 30.0,
            "assist_efficiency_human_saved_per_exo_abs_power": 3.0,
            "net_system_input_power_change_percent": -12.0,
            "total_fallen": 1,
        },
        {
            "candidate": "good",
            "human_abs_power_saved_percent": 24.0,
            "assist_efficiency_human_saved_per_exo_abs_power": 2.0,
            "net_system_input_power_change_percent": -8.0,
            "total_fallen": 0,
        },
        {
            "candidate": "better",
            "human_abs_power_saved_percent": 27.0,
            "assist_efficiency_human_saved_per_exo_abs_power": 1.7,
            "net_system_input_power_change_percent": -7.0,
            "total_fallen": 0,
        },
    ]

    ranked = rank_rows(rows, TARGET)

    assert [row["candidate"] for row in ranked] == ["better", "good", "unstable_high_saving"]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["passed_target"] is True
    assert ranked[-1]["passed_target"] is False


def test_assist_groups_preset_compares_joint_groups() -> None:
    candidates = default_candidates("assist-groups")

    assert [candidate.exo_joint_group for candidate in candidates] == [
        "knee",
        "hip",
        "ankle",
        "lower_limb",
    ]
