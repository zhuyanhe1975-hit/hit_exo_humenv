from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hit_exo_humenv.mjlab.commands import WalkingSpeedCommand, WalkingSpeedCommandCfg
from hit_exo_humenv.mjlab.actions import (
    S1HumanAction,
    S1HumanActionCfg,
    _calc_heading_quat_inv,
    _quat_to_tan_norm,
    _remove_smpl_base_rot,
)
from hit_exo_humenv.mjlab.mdp import (
    cached_lower_limb_joint_power_cost,
    commanded_forward_speed,
    lower_limb_joint_power_cost,
    lower_limb_joint_velocity_delta_l2,
)
from hit_exo_humenv.latent_z_config import cfg_path
from hit_exo_humenv.mjlab.walking_env_cfg import (
    exo_joint_names,
    exo_torque_limits,
    humenv_knee_exo_walking_env_cfg,
    humenv_knee_exo_walking_ppo_cfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul
from run_mjlab_knee_exo_viewer import _configure_walk_command, parse_args as parse_viewer_args


class _DummyEnv:
    num_envs = 32
    device = "cpu"


class _DummyCommandManager:
    def __init__(self, command: torch.Tensor) -> None:
        self._command = command

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "walk_speed"
        return self._command


class _DummyMdpEnv:
    def __init__(self, command: torch.Tensor) -> None:
        self.device = str(command.device)
        self.command_manager = _DummyCommandManager(command)


class _DummyLatentPolicy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def z_for_task(self, task: str) -> torch.Tensor:
        self.calls.append(task)
        direction, speed = task.removeprefix("move-ego-").rsplit("-", 1)
        return torch.tensor([float(speed), float(direction)])


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def reset(self, env_ids: torch.Tensor) -> None:
        self.calls.append(tuple(env_ids.tolist()))

    def write_data_to_sim(self) -> None:
        self.calls.append(("write",))

    def forward(self) -> None:
        self.calls.append(("forward",))


class _ResetScene(_Recorder):
    entities = {}


class _ResetEnv:
    num_envs = 4
    device = "cpu"

    def __init__(self) -> None:
        self.sim = _Recorder()
        self.scene = _ResetScene()
        self.action_manager = _Recorder()


def test_training_walk_command_resamples_speed_and_direction() -> None:
    cfg = WalkingSpeedCommandCfg(
        resampling_time_range=(1.0, 1.0),
        speed_choices=(0.8, 1.25),
        include_direction=True,
        direction_choices_deg=(-90.0, 0.0, 90.0),
    )
    command = WalkingSpeedCommand(cfg, _DummyEnv())

    env_ids = torch.arange(_DummyEnv.num_envs)
    command.reset(env_ids)

    assert command.command.shape == (32, 2)
    assert {round(value, 2) for value in command.command[:, 0].tolist()}.issubset({0.8, 1.25})
    assert {round(value, 1) for value in command.command[:, 1].tolist()}.issubset({-90.0, 0.0, 90.0})


def test_command_observation_encodes_direction_with_unit_circle() -> None:
    command = torch.tensor([[1.25, 90.0], [0.8, 0.0]])

    obs = commanded_forward_speed(_DummyMdpEnv(command))

    torch.testing.assert_close(obs, torch.tensor([[1.25, 1.0, 0.0], [0.8, 0.0, 1.0]]))


def test_s1_commanded_latents_use_cached_table_without_hot_cpu_loop() -> None:
    command = torch.tensor([[0.51, 10.0], [0.9, 80.0]])
    policy = _DummyLatentPolicy()
    action = S1HumanAction.__new__(S1HumanAction)
    action.cfg = S1HumanActionCfg(
        entity_name="robot",
        speed_command_name="walk_speed",
        speed_bins=(0.5, 1.0),
        direction_bins_deg=(0.0, 90.0),
        latent_speed_scale=2.0,
    )
    action._env = _DummyMdpEnv(command)
    action._policy = policy
    action._speed_bins = torch.tensor(action.cfg.speed_bins)
    action._direction_bins = torch.tensor(action.cfg.direction_bins_deg)
    action._latent_table = None

    latents = action._commanded_latents()
    cached_latents = action._commanded_latents()

    torch.testing.assert_close(latents, torch.tensor([[1.0, 0.0], [2.0, 90.0]]))
    torch.testing.assert_close(cached_latents, latents)
    assert policy.calls == [
        "move-ego-0-1",
        "move-ego-90-1",
        "move-ego-0-2",
        "move-ego-90-2",
    ]


def test_s1_proprio_quaternion_fast_paths_match_reference_math() -> None:
    torch.manual_seed(0)
    quat = torch.randn(16, 4)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)

    base_rot = torch.tensor([0.5, -0.5, -0.5, -0.5]).expand_as(quat)
    torch.testing.assert_close(_remove_smpl_base_rot(quat), quat_mul(quat, base_rot))

    ref_dir = torch.zeros((quat.shape[0], 3))
    ref_dir[:, 0] = 1.0
    rot_dir = quat_apply(quat, ref_dir)
    heading = torch.atan2(rot_dir[:, 1], rot_dir[:, 0])
    axis = torch.zeros_like(ref_dir)
    axis[:, 2] = 1.0
    torch.testing.assert_close(_calc_heading_quat_inv(quat), quat_from_angle_axis(-heading, axis))

    ref_tan = torch.zeros((quat.shape[0], 3))
    ref_tan[:, 0] = 1.0
    ref_norm = torch.zeros_like(ref_tan)
    ref_norm[:, 2] = 1.0
    reference = torch.cat([quat_apply(quat, ref_tan), quat_apply(quat, ref_norm)], dim=-1)
    torch.testing.assert_close(_quat_to_tan_norm(quat), reference)


def test_train_and_play_cfg_both_enable_direction() -> None:
    train_cfg = humenv_knee_exo_walking_env_cfg(play=False)
    play_cfg = humenv_knee_exo_walking_env_cfg(play=True)

    assert train_cfg.commands["walk_speed"].include_direction is True
    assert play_cfg.commands["walk_speed"].include_direction is True
    assert train_cfg.commands["walk_speed"].speed_choices == play_cfg.commands["walk_speed"].speed_choices
    assert train_cfg.commands["walk_speed"].direction_choices_deg == play_cfg.commands["walk_speed"].direction_choices_deg
    assert train_cfg.commands["walk_speed"].reset_on_resample is False
    assert play_cfg.commands["walk_speed"].reset_on_resample is False
    assert train_cfg.commands["walk_speed"].resampling_time_range == (1.0e9, 1.0e9)
    assert play_cfg.commands["walk_speed"].resampling_time_range == (1.0e9, 1.0e9)


def test_latent_z_walking_samples_only_forward_speeds() -> None:
    env_cfg = humenv_knee_exo_walking_env_cfg(play=False)
    command_cfg = env_cfg.commands["walk_speed"]
    human_cfg = env_cfg.actions["human_s1"]

    assert command_cfg.speed_range == tuple(cfg_path("walking_command", "speed_range"))
    assert command_cfg.speed_choices == tuple(cfg_path("walking_command", "speed_choices"))
    assert human_cfg.speed_bins == command_cfg.speed_choices
    assert command_cfg.direction_choices_deg == tuple(cfg_path("walking_command", "direction_choices_deg"))
    assert human_cfg.direction_bins_deg == tuple(cfg_path("walking_command", "direction_choices_deg"))
    assert human_cfg.latent_speed_scale == cfg_path("human_s1", "latent_speed_scale")
    assert human_cfg.action_repeat == cfg_path("human_s1", "action_repeat")
    assert human_cfg.action_smoothing == cfg_path("human_s1", "action_smoothing")


def test_play_cfg_is_not_training_episode_length() -> None:
    train_cfg = humenv_knee_exo_walking_env_cfg(play=False)
    play_cfg = humenv_knee_exo_walking_env_cfg(play=True)

    assert train_cfg.episode_length_s == 10.0
    assert play_cfg.episode_length_s > train_cfg.episode_length_s


def test_critic_reuses_actor_observation_group() -> None:
    env_cfg = humenv_knee_exo_walking_env_cfg(play=False)
    agent_cfg = humenv_knee_exo_walking_ppo_cfg()

    assert tuple(env_cfg.observations) == ("actor",)
    assert agent_cfg.obs_groups == {"actor": ("actor",), "critic": ("actor",)}


def test_latent_z_walking_uses_latest_exo_reward_terms() -> None:
    env_cfg = humenv_knee_exo_walking_env_cfg(play=False)

    assert "walk_speed" in env_cfg.commands
    assert env_cfg.actions["human_s1"].speed_command_name == "walk_speed"
    assert env_cfg.actions["human_s1"].tracking_motion_file is None
    assert env_cfg.actions["knee_exo"].joint_names == ("L_Knee_x", "R_Knee_x")
    assert env_cfg.actions["knee_exo"].max_torque == (
        cfg_path("exo", "max_knee_torque"),
        cfg_path("exo", "max_knee_torque"),
    )
    assert set(env_cfg.rewards) == {
        "not_fallen_progress",
        "fallen_penalty",
        "lower_limb_joint_power_cost",
        "lower_limb_joint_velocity_delta_l2",
    }
    assert env_cfg.rewards["not_fallen_progress"].weight == cfg_path("reward", "not_fallen_progress_weight")
    assert env_cfg.rewards["fallen_penalty"].weight == cfg_path("reward", "fallen_penalty_weight")
    assert env_cfg.terminations["fallen"].params["min_height"] == cfg_path("reward", "fall_min_height")
    assert env_cfg.rewards["lower_limb_joint_power_cost"].weight == cfg_path("reward", "lower_limb_joint_power_weight")
    power_params = env_cfg.rewards["lower_limb_joint_power_cost"].params
    power_weights = cfg_path("reward", "power_joint_weights")
    assert power_params["hip_weight"] == power_weights["hip"]
    assert power_params["knee_weight"] == power_weights["knee"]
    assert power_params["ankle_weight"] == power_weights["ankle"]
    assert power_params["positive_efficiency"] == cfg_path("metabolic", "positive_work_efficiency")
    assert power_params["negative_efficiency"] == cfg_path("metabolic", "negative_work_efficiency")
    assert env_cfg.rewards["lower_limb_joint_power_cost"].func is cached_lower_limb_joint_power_cost
    assert env_cfg.rewards["lower_limb_joint_velocity_delta_l2"].weight == cfg_path("reward", "lower_limb_joint_velocity_delta_weight")
    smooth_params = env_cfg.rewards["lower_limb_joint_velocity_delta_l2"].params
    smoothness_weights = cfg_path("reward", "smoothness_joint_weights")
    assert smooth_params["hip_weight"] == smoothness_weights["hip"]
    assert smooth_params["knee_weight"] == smoothness_weights["knee"]
    assert smooth_params["ankle_weight"] == smoothness_weights["ankle"]


def test_latent_z_exo_joint_group_can_expand_to_lower_limb(monkeypatch) -> None:
    monkeypatch.setenv("EXO_JOINT_GROUP", "lower_limb")
    env_cfg = humenv_knee_exo_walking_env_cfg(play=False)

    assert exo_joint_names() == (
        "L_Hip_x",
        "R_Hip_x",
        "L_Knee_x",
        "R_Knee_x",
        "L_Ankle_x",
        "R_Ankle_x",
    )
    assert env_cfg.actions["knee_exo"].joint_names == exo_joint_names()
    assert env_cfg.actions["knee_exo"].max_torque == exo_torque_limits()


def test_cached_lower_limb_power_cost_matches_reference_function() -> None:
    hip_cfg = SimpleNamespace(name="robot", joint_ids=(0, 1))
    knee_cfg = SimpleNamespace(name="robot", joint_ids=(2, 3))
    ankle_cfg = SimpleNamespace(name="robot", joint_ids=(4, 5))
    params = {
        "hip_cfg": hip_cfg,
        "knee_cfg": knee_cfg,
        "ankle_cfg": ankle_cfg,
        "hip_weight": 1.0,
        "knee_weight": 1.5,
        "ankle_weight": 0.25,
        "positive_efficiency": 0.25,
        "negative_efficiency": 1.2,
    }
    joint_vel = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0],
        ]
    )
    qfrc_actuator = torch.tensor(
        [
            [2.0, 3.0, -4.0, -5.0, 6.0, 7.0],
            [-2.0, -3.0, 4.0, 5.0, -6.0, -7.0],
        ]
    )
    qfrc_passive = torch.tensor(
        [
            [0.2, -0.3, 0.4, -0.5, 0.6, -0.7],
            [-0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
        ]
    )
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_vel=joint_vel,
                    qfrc_actuator=qfrc_actuator,
                    qfrc_passive=qfrc_passive,
                )
            )
        }
    )
    cached = cached_lower_limb_joint_power_cost(SimpleNamespace(params=params), env)

    expected = lower_limb_joint_power_cost(env, **params)
    actual = cached(env, **params)

    assert torch.allclose(actual, expected)


def test_cached_lower_limb_velocity_delta_matches_grouped_reference() -> None:
    hip_cfg = SimpleNamespace(name="robot", joint_ids=(0, 1))
    knee_cfg = SimpleNamespace(name="robot", joint_ids=(2, 3))
    ankle_cfg = SimpleNamespace(name="robot", joint_ids=(4, 5))
    params = {
        "hip_cfg": hip_cfg,
        "knee_cfg": knee_cfg,
        "ankle_cfg": ankle_cfg,
        "hip_weight": 1.0,
        "knee_weight": 1.5,
        "ankle_weight": 0.25,
    }
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([2, 1]),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_vel=torch.tensor(
                        [
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        ]
                    )
                )
            )
        },
    )
    reward = lower_limb_joint_velocity_delta_l2(SimpleNamespace(params=params), env)
    reward._prev_joint_vel = torch.tensor(
        [
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            [3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
        ]
    )
    delta = env.scene["robot"].data.joint_vel[0] - reward._prev_joint_vel[0]
    expected = (
        params["hip_weight"] * torch.sum(torch.square(delta[list(hip_cfg.joint_ids)]))
        + params["knee_weight"] * torch.sum(torch.square(delta[list(knee_cfg.joint_ids)]))
        + params["ankle_weight"] * torch.sum(torch.square(delta[list(ankle_cfg.joint_ids)]))
    )

    actual = reward(env, **params)

    assert torch.allclose(actual, torch.tensor([expected, 0.0]))


def test_latent_z_viewer_does_not_override_episode_locked_command(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["viewer", "--checkpoint-file", "dummy.pt"])
    args = parse_viewer_args()
    env_cfg = humenv_knee_exo_walking_env_cfg(play=True)

    _configure_walk_command(env_cfg, args)

    assert env_cfg.commands["walk_speed"].resampling_time_range == (1.0e9, 1.0e9)
    assert env_cfg.commands["walk_speed"].reset_on_resample is False


def test_walk_command_does_not_reset_when_resampled_command_is_unchanged() -> None:
    env = _ResetEnv()
    cfg = WalkingSpeedCommandCfg(
        resampling_time_range=(1.0, 1.0),
        speed_choices=(1.25,),
        include_direction=True,
        direction_choices_deg=(0.0,),
        reset_on_resample=True,
    )
    command = WalkingSpeedCommand(cfg, env)
    env_ids = torch.arange(env.num_envs)

    command.reset(env_ids)
    assert env.sim.calls == []
    assert env.scene.calls == []
    assert env.action_manager.calls == []

    command.time_left[:] = 0.0
    command.compute(dt=0.1)

    assert env.sim.calls == []
    assert env.scene.calls == []
    assert env.action_manager.calls == []


def test_walk_command_lightly_resets_state_after_running_command_change() -> None:
    env = _ResetEnv()
    cfg = WalkingSpeedCommandCfg(
        resampling_time_range=(1.0, 1.0),
        speed_choices=(1.25,),
        include_direction=True,
        direction_choices_deg=(0.0,),
        reset_on_resample=True,
    )
    command = WalkingSpeedCommand(cfg, env)
    env_ids = torch.arange(env.num_envs)
    previous_command = torch.tensor([[1.25, 0.0]] * env.num_envs)

    command.command[:, 0] = 1.5
    command.command[:, 1] = 90.0
    command.command_counter[:] = 1
    command._reset_state_after_running_resample(env_ids, previous_command)

    assert env.sim.calls == [("forward",)]
    assert env.scene.calls == [("write",)]
    assert env.action_manager.calls == [(0, 1, 2, 3)]
