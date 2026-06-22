from __future__ import annotations

from types import SimpleNamespace
import sys

import h5py
import numpy as np
import torch

from hit_exo_humenv.mjlab.mdp import (
    action_assist_replacement_l2,
    lower_limb_joint_power_cost,
    lower_limb_joint_velocity_delta_l2,
    mocap_foot_event_tracking_reward,
    mocap_foot_pos_tracking_reward,
    mocap_qpos_tracking_reward,
    not_fallen_progress_reward,
)

sys.path.insert(0, "scripts")
from train_mjlab_knee_exo_mocap_track import build_train_config, parse_args  # noqa: E402


class _DummyEnv:
    device = "cpu"
    step_dt = 0.1

    def __init__(self, qpos: torch.Tensor, episode_length: torch.Tensor) -> None:
        self.sim = SimpleNamespace(data=SimpleNamespace(qpos=qpos))
        self.episode_length_buf = episode_length


class _DummyPowerEnv:
    def __init__(self) -> None:
        asset = SimpleNamespace(
            data=SimpleNamespace(
                qfrc_actuator=torch.tensor([[2.0, -3.0, 4.0, 5.0, -6.0, 7.0]]),
                qfrc_passive=torch.tensor([[-1.0, 1.0, -2.0, -3.0, 4.0, 0.0]]),
                joint_vel=torch.tensor([[0.5, -0.25, -0.5, 2.0, -1.0, 0.0]]),
            )
        )
        self.scene = {"robot": asset}
        self.episode_length_buf = torch.tensor([2])


class _Scene(SimpleNamespace):
    def __getitem__(self, name: str):
        return self.assets[name]


def _write_motion(path) -> None:
    qpos = np.zeros((3, 76), dtype=np.float32)
    qvel = np.zeros((3, 75), dtype=np.float32)
    qpos[0, 2] = 0.94
    qpos[1, 2] = 0.96
    qpos[2, 2] = 0.98
    qpos[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    qpos[:, 10] = [0.1, 0.2, 0.3]
    qpos[:, 22] = [-0.1, -0.2, -0.3]
    with h5py.File(path, "w") as hf:
        hf.attrs["dt"] = 0.1
        ep = hf.create_group("ep_0")
        ep.create_dataset("qpos", data=qpos)
        ep.create_dataset("qvel", data=qvel)
        ep.create_dataset("observation", data=np.zeros((3, 358), dtype=np.float32))
        ep.attrs["dt"] = 0.1


def test_mocap_qpos_tracking_reward_matches_reference_phase(tmp_path) -> None:
    motion = tmp_path / "walk.hdf5"
    _write_motion(motion)
    qpos = torch.zeros(2, 30)
    qpos[0, 2] = 0.94
    qpos[1, 2] = 0.96
    env = _DummyEnv(qpos=qpos, episode_length=torch.tensor([1, 2]))
    reward = mocap_qpos_tracking_reward(None, env)

    out = reward(env, motion_file=str(motion), columns=(2,), std=0.05)

    torch.testing.assert_close(out, torch.ones(2))


def test_mocap_qpos_tracking_reward_compares_root_xy_displacement(tmp_path) -> None:
    motion = tmp_path / "offset_walk.hdf5"
    _write_motion(motion)
    with h5py.File(motion, "r+") as hf:
        qpos = hf["ep_0/qpos"][:]
        qpos[:, 0] = [5.0, 5.2, 5.4]
        qpos[:, 1] = [-3.0, -2.9, -2.8]
        hf["ep_0/qpos"][:] = qpos

    qpos = torch.zeros(1, 30)
    qpos[0, 0] = 15.4
    qpos[0, 1] = 6.2
    qpos[0, 2] = 0.98
    env = _DummyEnv(qpos=qpos, episode_length=torch.tensor([3]))
    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_root_state=torch.tensor([[5.0, -3.0, 0.94]]),
        )
    )
    env.scene = _Scene(env_origins=torch.tensor([[10.0, 9.0, 0.0]]), assets={"robot": asset})
    reward = mocap_qpos_tracking_reward(None, env)

    out = reward(env, motion_file=str(motion), columns=(0, 1, 2), std=0.05, root_xy_invariant=False)

    torch.testing.assert_close(out, torch.ones(1))


def test_mocap_qpos_tracking_reward_penalizes_pose_error(tmp_path) -> None:
    motion = tmp_path / "walk.hdf5"
    _write_motion(motion)
    qpos = torch.zeros(1, 30)
    qpos[0, 10] = 1.0
    env = _DummyEnv(qpos=qpos, episode_length=torch.tensor([1]))
    reward = mocap_qpos_tracking_reward(None, env)

    out = reward(env, motion_file=str(motion), columns=(10,), std=0.1)

    assert out.item() < 0.01


def test_mocap_foot_tracking_reward_compares_root_relative_xy(monkeypatch) -> None:
    reference = torch.tensor(
        [
            [
                [[0.2, 0.1, 0.3], [0.0, -0.1, 0.3]],
                [[0.4, 0.2, 0.5], [0.1, -0.2, 0.3]],
            ]
        ],
        dtype=torch.float32,
    ).squeeze(0)
    monkeypatch.setattr(
        "hit_exo_humenv.mjlab.mdp._load_resampled_mocap_foot_centers",
        lambda *args, **kwargs: reference,
    )
    body_pos = torch.zeros(1, 4, 3)
    body_pos[0, 0] = torch.tensor([10.3, 5.2, 0.5])
    body_pos[0, 1] = torch.tensor([10.5, 5.2, 0.5])
    body_pos[0, 2] = torch.tensor([10.1, 4.8, 0.3])
    body_pos[0, 3] = torch.tensor([10.1, 4.8, 0.3])
    asset = SimpleNamespace(
        body_names=("L_Ankle", "L_Toe", "R_Ankle", "R_Toe"),
        data=SimpleNamespace(
            body_link_pos_w=body_pos,
            default_root_state=torch.tensor([[10.0, 5.0, 0.9]]),
        ),
    )
    env = SimpleNamespace(
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.tensor([2]),
        scene=_Scene(env_origins=torch.tensor([[0.0, 0.0, 0.0]]), assets={"robot": asset}),
    )
    reward = mocap_foot_pos_tracking_reward(None, env)

    out = reward(env, motion_file="dummy.hdf5", robot_xml="dummy.xml", std=0.05)

    torch.testing.assert_close(out, torch.ones(1))


def test_mocap_foot_event_tracking_reward_targets_active_touchdown_side(monkeypatch) -> None:
    reference = torch.zeros(8, 2, 3)
    reference[:, 0, :] = torch.tensor([0.0, -0.1, 0.3])
    reference[:, 1, :] = torch.tensor([0.0, 0.1, 0.3])
    reference[3:, 1, :] = torch.tensor([0.4, 0.2, 0.5])
    monkeypatch.setattr(
        "hit_exo_humenv.mjlab.mdp._load_resampled_mocap_foot_centers",
        lambda *args, **kwargs: reference,
    )
    body_pos = torch.zeros(1, 4, 3)
    body_pos[0, 0] = torch.tensor([10.0, 4.9, 0.3])
    body_pos[0, 1] = torch.tensor([10.0, 4.9, 0.3])
    body_pos[0, 2] = torch.tensor([10.4, 5.2, 0.5])
    body_pos[0, 3] = torch.tensor([10.4, 5.2, 0.5])
    asset = SimpleNamespace(
        body_names=("L_Ankle", "L_Toe", "R_Ankle", "R_Toe"),
        data=SimpleNamespace(
            body_link_pos_w=body_pos,
            default_root_state=torch.tensor([[10.0, 5.0, 0.9]]),
        ),
    )
    env = SimpleNamespace(
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.tensor([5]),
        scene=_Scene(env_origins=torch.tensor([[0.0, 0.0, 0.0]]), assets={"robot": asset}),
    )
    reward = mocap_foot_event_tracking_reward(None, env)

    out = reward(
        env,
        motion_file="dummy.hdf5",
        robot_xml="dummy.xml",
        std=0.05,
        min_stance_frames=2,
    )

    torch.testing.assert_close(out, torch.ones(1))


def test_action_assist_replacement_l2_compares_full_action_channels() -> None:
    action_term = SimpleNamespace(
        residual_action_full=torch.tensor([[0.1, 0.2, 0.3]]),
        mocap_assist_replacement_target=torch.tensor([[0.1, -0.1, 0.5]]),
    )
    env = SimpleNamespace(
        device="cpu",
        episode_length_buf=torch.tensor([1]),
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
    )

    out = action_assist_replacement_l2(env, action_name="human_s1")

    torch.testing.assert_close(out, torch.tensor([0.13]))


def test_lower_limb_joint_power_cost_excludes_passive_joint_forces() -> None:
    env = _DummyPowerEnv()
    hip_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0, 1]))
    knee_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([2, 3]))
    ankle_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([4, 5]))

    out = lower_limb_joint_power_cost(
        env,
        hip_cfg=hip_cfg,
        knee_cfg=knee_cfg,
        ankle_cfg=ankle_cfg,
        hip_weight=1.0,
        knee_weight=2.0,
        ankle_weight=3.0,
    )

    expected_power = 1.0 * (1.0 / 0.25) + 2.0 * (4.0 / 0.25 + 1.0 / 1.20) + 3.0 * (
        2.0 / 0.25
    )
    torch.testing.assert_close(out, torch.tensor([expected_power]))


def test_lower_limb_joint_velocity_delta_l2_uses_consecutive_velocity_changes() -> None:
    env = _DummyPowerEnv()
    hip_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0, 1]))
    knee_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([2, 3]))
    ankle_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([4, 5]))
    cost = lower_limb_joint_velocity_delta_l2(None, env)

    first = cost(
        env,
        hip_cfg=hip_cfg,
        knee_cfg=knee_cfg,
        ankle_cfg=ankle_cfg,
        hip_weight=1.0,
        knee_weight=2.0,
        ankle_weight=3.0,
    )
    env.scene["robot"].data.joint_vel += torch.tensor([[1.0, 0.0, 0.5, 0.0, 2.0, 0.0]])
    second = cost(
        env,
        hip_cfg=hip_cfg,
        knee_cfg=knee_cfg,
        ankle_cfg=ankle_cfg,
        hip_weight=1.0,
        knee_weight=2.0,
        ankle_weight=3.0,
    )

    expected_cost = 1.0 * 1.0 + 2.0 * 0.25 + 3.0 * 4.0
    torch.testing.assert_close(first, torch.zeros(1))
    torch.testing.assert_close(second, torch.tensor([expected_cost]))


def test_not_fallen_progress_reward_is_one_until_root_drops() -> None:
    asset = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.7]])))
    env = SimpleNamespace(scene={"robot": asset})

    out = not_fallen_progress_reward(env, min_height=0.75, asset_cfg=SimpleNamespace(name="robot"))

    torch.testing.assert_close(out, torch.tensor([1.0, 0.0]))


def test_mocap_track_config_removes_speed_command(monkeypatch, tmp_path) -> None:
    motion = tmp_path / "walk.hdf5"
    _write_motion(motion)
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--motion", str(motion), "--num-envs", "2", "--max-iterations", "1", "--cpu"],
    )

    cfg = build_train_config(parse_args())

    assert cfg.env.commands == {}
    assert "walk_speed" not in cfg.env.observations["actor"].terms
    assert "commanded_velocity" not in cfg.env.rewards
    assert cfg.env.actions["human_s1"].speed_command_name is None
    assert cfg.env.actions["human_s1"].task == "tracking"
    assert cfg.env.actions["human_s1"].tracking_motion_file == str(motion)
    assert cfg.env.actions["human_s1"].tracking_episode == "ep_0"
    np.testing.assert_allclose(cfg.env.scene.entities["robot"].init_state.pos, (0.0, 0.0, 0.94))
    np.testing.assert_allclose(cfg.env.scene.entities["robot"].init_state.rot, (0.0, 0.0, 0.0, 1.0))
    assert cfg.env.actions["knee_exo"].max_torque == 25.0
    assert set(cfg.env.rewards) == {
        "not_fallen_progress",
        "fallen_penalty",
        "lower_limb_joint_power_cost",
        "lower_limb_joint_velocity_delta_l2",
    }
    assert cfg.env.rewards["not_fallen_progress"].weight == 0.02
    assert cfg.env.rewards["fallen_penalty"].weight == -5.0
    assert cfg.env.terminations["fallen"].params["min_height"] == 0.5
    assert cfg.env.rewards["lower_limb_joint_power_cost"].weight == -1.0e-3
    assert cfg.env.rewards["lower_limb_joint_velocity_delta_l2"].weight == -0.02
    smooth_params = cfg.env.rewards["lower_limb_joint_velocity_delta_l2"].params
    assert smooth_params["hip_weight"] == 1.0
    assert smooth_params["knee_weight"] == 1.0
    assert smooth_params["ankle_weight"] == 3.0


def test_mocap_track_config_can_add_tracking_rewards(monkeypatch, tmp_path) -> None:
    motion = tmp_path / "stairs.hdf5"
    _write_motion(motion)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--motion",
            str(motion),
            "--num-envs",
            "2",
            "--max-iterations",
            "1",
            "--cpu",
            "--mocap-root-xyz-weight",
            "0.4",
            "--mocap-root-orientation-weight",
            "0.3",
            "--mocap-lower-body-weight",
            "0.8",
            "--mocap-knee-weight",
            "1.2",
            "--mocap-foot-weight",
            "0.7",
            "--mocap-first-foot-weight",
            "1.5",
            "--mocap-foot-event-weight",
            "2.5",
        ],
    )

    cfg = build_train_config(parse_args())

    assert cfg.env.rewards["mocap_root_xyz_tracking"].weight == 0.4
    assert cfg.env.rewards["mocap_root_xyz_tracking"].params["columns"] == (0, 1, 2)
    assert cfg.env.rewards["mocap_root_xyz_tracking"].params["root_xy_invariant"] is False
    assert cfg.env.rewards["mocap_root_orientation_tracking"].weight == 0.3
    assert cfg.env.rewards["mocap_root_orientation_tracking"].params["columns"] == (3, 4, 5, 6)
    assert cfg.env.rewards["mocap_lower_body_tracking"].weight == 0.8
    assert cfg.env.rewards["mocap_lower_body_tracking"].params["columns"] == tuple(range(7, 30))
    assert cfg.env.rewards["mocap_knee_tracking"].weight == 1.2
    assert cfg.env.rewards["mocap_knee_tracking"].params["columns"] == (10, 22)
    assert cfg.env.rewards["mocap_foot_tracking"].weight == 0.7
    assert cfg.env.rewards["mocap_foot_tracking"].params["end_frame"] == -1
    assert cfg.env.rewards["mocap_first_foot_tracking"].weight == 1.5
    assert cfg.env.rewards["mocap_first_foot_tracking"].params["end_frame"] == 90
    assert cfg.env.rewards["mocap_foot_event_tracking"].weight == 2.5
    assert cfg.env.rewards["mocap_foot_event_tracking"].params["min_stance_frames"] == 4


def test_mocap_track_config_can_enable_human_residual(monkeypatch, tmp_path) -> None:
    motion = tmp_path / "stairs.hdf5"
    _write_motion(motion)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--motion",
            str(motion),
            "--num-envs",
            "2",
            "--max-iterations",
            "1",
            "--cpu",
            "--human-residual-mode",
            "lower-body",
            "--human-residual-scale",
            "0.35",
            "--human-residual-action-weight",
            "-0.002",
            "--human-residual-action-rate-weight",
            "-0.004",
            "--mocap-assist-replacement-weight",
            "-0.25",
            "--mocap-assist-start",
            "1.0",
            "--mocap-assist-end",
            "0.0",
            "--mocap-assist-decay-fraction",
            "0.5",
        ],
    )

    cfg = build_train_config(parse_args())
    human_cfg = cfg.env.actions["human_s1"]

    assert human_cfg.residual_scale == 0.35
    assert human_cfg.residual_joint_names == (
        "L_Hip_.*",
        "R_Hip_.*",
        "L_Knee_.*",
        "R_Knee_.*",
        "L_Ankle_.*",
        "R_Ankle_.*",
        "L_Toe_.*",
        "R_Toe_.*",
    )
    assert cfg.env.rewards["human_residual_action_l2"].weight == -0.002
    assert cfg.env.rewards["human_residual_action_l2"].params["action_name"] == "human_s1"
    assert cfg.env.rewards["human_residual_action_rate_l2"].weight == -0.004
    assert cfg.env.rewards["mocap_assist_replacement_l2"].weight == -0.25
    assert human_cfg.mocap_assist_start == 1.0
    assert human_cfg.mocap_assist_end == 0.0
    assert human_cfg.mocap_assist_decay_steps == 8


def test_stair_compensation_stage_trains_only_full_body_residual(monkeypatch, tmp_path) -> None:
    motion = tmp_path / "stairs.hdf5"
    _write_motion(motion)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--motion",
            str(motion),
            "--training-stage",
            "stair-compensation",
            "--num-envs",
            "2",
            "--max-iterations",
            "1",
            "--cpu",
        ],
    )

    cfg = build_train_config(parse_args())
    human_cfg = cfg.env.actions["human_s1"]

    assert set(cfg.env.actions) == {"human_s1"}
    assert human_cfg.residual_joint_names == (".*",)
    assert human_cfg.residual_scale == 0.35
    assert human_cfg.residual_policy_checkpoint is None
    assert human_cfg.mocap_assist_start == 0.0
    assert cfg.env.rewards["human_residual_action_l2"].weight == -0.001
    assert cfg.agent.experiment_name == "humenv_s1_stair_compensation"


def test_knee_exo_stage_can_freeze_compensation_checkpoint(monkeypatch, tmp_path) -> None:
    motion = tmp_path / "stairs.hdf5"
    checkpoint = tmp_path / "model_999.pt"
    _write_motion(motion)
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--motion",
            str(motion),
            "--training-stage",
            "knee-exo-on-compensation",
            "--base-compensation-checkpoint",
            str(checkpoint),
            "--num-envs",
            "2",
            "--max-iterations",
            "1",
            "--cpu",
        ],
    )

    cfg = build_train_config(parse_args())
    human_cfg = cfg.env.actions["human_s1"]

    assert set(cfg.env.actions) == {"human_s1", "knee_exo"}
    assert human_cfg.residual_joint_names == (".*",)
    assert human_cfg.residual_scale == 0.35
    assert human_cfg.residual_policy_checkpoint == str(checkpoint)
    assert "human_residual_action_l2" not in cfg.env.rewards
    assert cfg.agent.experiment_name == "humenv_knee_exo_on_stair_compensation"
