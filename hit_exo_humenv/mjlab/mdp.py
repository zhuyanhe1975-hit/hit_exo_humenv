from __future__ import annotations

from pathlib import Path

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


ROBOT_CFG = SceneEntityCfg("robot")
DEFAULT_MOCAP_MOTION = Path(
    "/home/yhzhu/AI/humenv/data_preparation/humenv_from_phc_amass_transitions_kit_cmu/"
    "08_KIT_167_walking_medium_resampled_poses.hdf5"
)
FOOT_BODY_PAIRS = {
    "L": ("L_Ankle", "L_Toe"),
    "R": ("R_Ankle", "R_Toe"),
}


def forward_velocity_reward(
    env,
    target_speed: float = 2.0,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        target_speed = command[:, 0]
    speed = asset.data.root_link_lin_vel_b[:, 0]
    return torch.exp(-torch.square(speed - target_speed) / 0.25)


def commanded_velocity_reward(
    env,
    command_name: str = "walk_speed",
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward tracking the egocentric walking command in the horizontal plane."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    speed = command[:, 0]
    if command.shape[1] > 1:
        direction_rad = torch.deg2rad(command[:, 1])
        target_xy = torch.stack(
            [speed * torch.cos(direction_rad), speed * torch.sin(direction_rad)],
            dim=1,
        )
    else:
        target_xy = torch.stack([speed, torch.zeros_like(speed)], dim=1)
    velocity_xy = asset.data.root_link_lin_vel_b[:, :2]
    error = torch.sum(torch.square(velocity_xy - target_xy), dim=1)
    return torch.exp(-error / (std * std))


def commanded_forward_speed(env, command_name: str = "walk_speed") -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    if command.shape[1] == 1:
        return command
    direction_rad = torch.deg2rad(command[:, 1:2])
    return torch.cat([command[:, 0:1], torch.sin(direction_rad), torch.cos(direction_rad)], dim=-1)


def upright_reward(env, asset_cfg: SceneEntityCfg = ROBOT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 1], min=0.0)


def height_reward(
    env,
    min_height: float = 0.9,
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return (asset.data.root_link_pos_w[:, 2] > min_height).float()


def not_fallen_progress_reward(
    env,
    min_height: float = 0.5,
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
    return (~root_height_below(env, min_height=min_height, asset_cfg=asset_cfg)).float()


def action_l2(env, action_name: str = "knee_exo") -> torch.Tensor:
    action = env.action_manager.get_term(action_name).raw_action
    return torch.sum(torch.square(action), dim=1)


def action_rate_l2(env, action_name: str = "knee_exo") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    action = action_term.raw_action
    start = 0
    for name, term in env.action_manager._terms.items():
        if name == action_name:
            break
        start += term.action_dim
    previous = env.action_manager.prev_action[:, start : start + action.shape[1]]
    return torch.sum(torch.square(action - previous), dim=1)


def action_assist_replacement_l2(env, action_name: str = "human_s1") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    residual = getattr(action_term, "residual_action_full", None)
    target = getattr(action_term, "mocap_assist_replacement_target", None)
    if residual is None or target is None:
        num_envs = getattr(env, "num_envs", env.episode_length_buf.shape[0])
        return torch.zeros(num_envs, device=env.device)
    return torch.sum(torch.square(residual - target.detach()), dim=1)


def knee_exo_torque_l2(env, action_name: str = "knee_exo") -> torch.Tensor:
    return action_l2(env, action_name=action_name)


def knee_exo_action_rate_l2(env, action_name: str = "knee_exo") -> torch.Tensor:
    return action_rate_l2(env, action_name=action_name)


def knee_pd_torque_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=("L_Knee_x", "R_Knee_x")),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1)


def knee_pd_torque_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=("L_Knee_x", "R_Knee_x")),
    scale: float = 1.0e-4,
) -> torch.Tensor:
    torque_l2 = knee_pd_torque_l2(env, asset_cfg=asset_cfg)
    return torch.exp(-scale * torque_l2)


def actuator_joint_force_excluding_passive(env, asset: Entity, joint_ids) -> torch.Tensor:
    """Actuator force after canceling artificial passive joint forces.

    ``qfrc_actuator`` is the actual generalized actuator force, but in this
    model some XML joint stiffness/damping exists only to make simulation
    stable. Adding ``qfrc_passive`` removes the actuator work spent overcoming
    those artificial passive forces from the power accounting.
    """
    del env
    actuator_force = asset.data.qfrc_actuator[:, joint_ids]
    passive_field = getattr(asset.data, "_joint_dof_field", None)
    if passive_field is not None:
        return actuator_force + passive_field("qfrc_passive")[:, joint_ids]

    passive_force = getattr(asset.data, "qfrc_passive", None)
    if passive_force is None:
        return actuator_force
    return actuator_force + passive_force[:, joint_ids]


def lower_limb_joint_power_cost(
    env,
    hip_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=("L_Hip_.*", "R_Hip_.*"),
    ),
    knee_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=("L_Knee_.*", "R_Knee_.*"),
    ),
    ankle_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=("L_Ankle_.*", "R_Ankle_.*"),
    ),
    hip_weight: float = 1.0,
    knee_weight: float = 1.0,
    ankle_weight: float = 1.0,
    positive_efficiency: float = 0.25,
    negative_efficiency: float = 1.20,
) -> torch.Tensor:
    """Weighted lower-limb metabolic power proxy excluding passive damping.

    Positive and negative joint mechanical powers have different muscle costs.
    This uses the common efficiency approximation: positive mechanical work is
    divided by ``positive_efficiency`` and absorbed negative work magnitude is
    divided by ``negative_efficiency``.

    The force term intentionally cancels MuJoCo passive joint forces that exist
    for simulation stability, such as XML joint stiffness and damping.
    """
    asset: Entity = env.scene[hip_cfg.name]

    def group_metabolic_power(cfg: SceneEntityCfg) -> torch.Tensor:
        torque = actuator_joint_force_excluding_passive(env, asset, cfg.joint_ids)
        velocity = asset.data.joint_vel[:, cfg.joint_ids]
        mechanical_power = torque * velocity
        positive_power = torch.clamp(mechanical_power, min=0.0)
        negative_power = torch.clamp(-mechanical_power, min=0.0)
        metabolic_power = positive_power / positive_efficiency + negative_power / negative_efficiency
        return torch.sum(metabolic_power, dim=1)

    power = (
        hip_weight * group_metabolic_power(hip_cfg)
        + knee_weight * group_metabolic_power(knee_cfg)
        + ankle_weight * group_metabolic_power(ankle_cfg)
    )
    return power


class cached_lower_limb_joint_power_cost:
    """Weighted lower-limb metabolic power cost with cached joint ids and weights."""

    def __init__(self, cfg, env) -> None:
        params = cfg.params
        hip_cfg = params["hip_cfg"]
        knee_cfg = params["knee_cfg"]
        ankle_cfg = params["ankle_cfg"]
        del env
        self._asset_name = hip_cfg.name
        self._joint_ids = torch.cat(
            [
                torch.as_tensor(hip_cfg.joint_ids, dtype=torch.long),
                torch.as_tensor(knee_cfg.joint_ids, dtype=torch.long),
                torch.as_tensor(ankle_cfg.joint_ids, dtype=torch.long),
            ],
            dim=0,
        )
        self._weights = torch.cat(
            [
                torch.full((len(hip_cfg.joint_ids),), float(params["hip_weight"])),
                torch.full((len(knee_cfg.joint_ids),), float(params["knee_weight"])),
                torch.full((len(ankle_cfg.joint_ids),), float(params["ankle_weight"])),
            ],
            dim=0,
        )
        self._positive_efficiency = float(params["positive_efficiency"])
        self._negative_efficiency = float(params["negative_efficiency"])
        self._device: torch.device | None = None

    def __call__(self, env, **_ignored) -> torch.Tensor:
        asset: Entity = env.scene[self._asset_name]
        if self._device != asset.data.joint_vel.device:
            self._joint_ids = self._joint_ids.to(device=asset.data.joint_vel.device)
            self._weights = self._weights.to(device=asset.data.joint_vel.device)
            self._device = asset.data.joint_vel.device

        torque = actuator_joint_force_excluding_passive(env, asset, self._joint_ids)
        velocity = asset.data.joint_vel[:, self._joint_ids]
        mechanical_power = torque * velocity
        positive_power = torch.clamp(mechanical_power, min=0.0)
        negative_power = torch.clamp(-mechanical_power, min=0.0)
        metabolic_power = positive_power / self._positive_efficiency + negative_power / self._negative_efficiency
        return torch.sum(metabolic_power * self._weights, dim=1)


class lower_limb_joint_velocity_delta_l2:
    """Cost for lower-limb joint velocity changes across consecutive RL steps."""

    def __init__(self, cfg, env) -> None:
        self._asset_name: str | None = None
        self._joint_ids: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._prev_joint_vel: torch.Tensor | None = None
        if cfg is None:
            return
        self._configure_from_params(cfg.params)
        del env

    def _configure_from_params(self, params) -> None:
        hip_cfg = params["hip_cfg"]
        knee_cfg = params["knee_cfg"]
        ankle_cfg = params["ankle_cfg"]
        self._asset_name = hip_cfg.name
        self._joint_ids = torch.cat(
            [
                torch.as_tensor(hip_cfg.joint_ids, dtype=torch.long),
                torch.as_tensor(knee_cfg.joint_ids, dtype=torch.long),
                torch.as_tensor(ankle_cfg.joint_ids, dtype=torch.long),
            ],
            dim=0,
        )
        self._weights = torch.cat(
            [
                torch.full((len(hip_cfg.joint_ids),), float(params["hip_weight"])),
                torch.full((len(knee_cfg.joint_ids),), float(params["knee_weight"])),
                torch.full((len(ankle_cfg.joint_ids),), float(params["ankle_weight"])),
            ],
            dim=0,
        )

    def __call__(self, env, **params) -> torch.Tensor:
        if self._joint_ids is None or self._weights is None:
            self._configure_from_params(params)
        assert self._asset_name is not None
        assert self._joint_ids is not None
        assert self._weights is not None
        asset: Entity = env.scene[self._asset_name]
        current = asset.data.joint_vel
        if self._device != current.device:
            self._joint_ids = self._joint_ids.to(device=current.device)
            self._weights = self._weights.to(device=current.device)
            self._device = current.device
        if self._prev_joint_vel is None or self._prev_joint_vel.shape != current.shape:
            self._prev_joint_vel = current.detach().clone()

        reset_mask = env.episode_length_buf <= 1
        delta = current[:, self._joint_ids] - self._prev_joint_vel[:, self._joint_ids]
        delta = delta.masked_fill(reset_mask[:, None], 0.0)
        smoothness_cost = torch.sum(torch.square(delta) * self._weights, dim=1)
        self._prev_joint_vel.copy_(current.detach())
        return smoothness_cost


def knee_joint_vel_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=("L_Knee_x", "R_Knee_x")),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def base_ang_vel_l2(
    env,
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


class mocap_qpos_tracking_reward:
    """Track selected mocap qpos columns at the current episode phase."""

    def __init__(self, cfg, env) -> None:
        del cfg
        self._reference_cache: dict[tuple[str, str, float], torch.Tensor] = {}
        self._device = env.device
        self._target_dt = env.step_dt

    def __call__(
        self,
        env,
        motion_file: str = str(DEFAULT_MOCAP_MOTION),
        episode: str = "ep_0",
        columns: tuple[int, ...] | None = None,
        std: float = 0.5,
        root_xy_invariant: bool = True,
    ) -> torch.Tensor:
        reference = self._reference_qpos(motion_file, episode)
        phase = torch.remainder(env.episode_length_buf - 1, reference.shape[0]).long()
        target = reference[phase]
        qpos = env.sim.data.qpos
        if root_xy_invariant:
            qpos = qpos.clone()
            target = target.clone()
            qpos[:, :2] = 0.0
            target[:, :2] = 0.0
        elif qpos.shape[1] >= 2:
            qpos = qpos.clone()
            qpos[:, :2] -= _root_xy_origin(env, qpos)
        if columns is not None:
            column_ids = torch.as_tensor(columns, device=qpos.device, dtype=torch.long)
            qpos = qpos[:, column_ids]
            target = target[:, column_ids]
        error = torch.mean(torch.square(qpos - target), dim=1)
        return torch.exp(-error / (std * std))

    def _reference_qpos(self, motion_file: str, episode: str) -> torch.Tensor:
        key = (motion_file, episode, self._target_dt)
        reference = self._reference_cache.get(key)
        if reference is None:
            reference = _load_resampled_mocap_qpos(Path(motion_file), episode, self._target_dt)
            self._reference_cache[key] = reference.to(device=self._device, dtype=torch.float32)
        return self._reference_cache[key]


class mocap_foot_pos_tracking_reward:
    """Track mocap ankle/toe foot centers in world height and root-relative XY."""

    def __init__(self, cfg, env) -> None:
        del cfg
        self._reference_cache: dict[tuple[str, str, str, float], torch.Tensor] = {}
        self._body_id_cache: dict[int, dict[str, tuple[int, int]]] = {}
        self._device = env.device
        self._target_dt = env.step_dt

    def __call__(
        self,
        env,
        motion_file: str = str(DEFAULT_MOCAP_MOTION),
        episode: str = "ep_0",
        robot_xml: str = "",
        sides: tuple[str, ...] = ("L", "R"),
        std: float = 0.25,
        xy_weight: float = 1.0,
        z_weight: float = 1.0,
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> torch.Tensor:
        if std <= 0.0:
            raise ValueError("mocap foot tracking std must be positive")
        reference = self._reference_feet(motion_file, episode, robot_xml)
        phase = torch.remainder(env.episode_length_buf - 1, reference.shape[0]).long()
        active = phase >= int(start_frame)
        if end_frame >= 0:
            active = active & (phase <= int(end_frame))

        side_ids = torch.as_tensor(
            [0 if side == "L" else 1 for side in sides],
            device=reference.device,
            dtype=torch.long,
        )
        target = reference[phase][:, side_ids, :]
        current = self._current_foot_centers(env, sides)
        current = current.clone()
        current[:, :, :2] -= _root_xy_origin(env, current[:, 0, :])[:, None, :]

        xy_error = torch.sum(torch.square(current[:, :, :2] - target[:, :, :2]), dim=-1)
        z_error = torch.square(current[:, :, 2] - target[:, :, 2])
        error = torch.mean(float(xy_weight) * xy_error + float(z_weight) * z_error, dim=1)
        return active.to(dtype=error.dtype) * torch.exp(-error / (std * std))

    def _reference_feet(self, motion_file: str, episode: str, robot_xml: str) -> torch.Tensor:
        key = (motion_file, episode, robot_xml, self._target_dt)
        reference = self._reference_cache.get(key)
        if reference is None:
            reference = _load_resampled_mocap_foot_centers(
                Path(motion_file),
                episode,
                self._target_dt,
                Path(robot_xml),
            )
            self._reference_cache[key] = reference.to(device=self._device, dtype=torch.float32)
        return self._reference_cache[key]

    def _current_foot_centers(self, env, sides: tuple[str, ...]) -> torch.Tensor:
        asset: Entity = env.scene["robot"]
        body_ids = self._body_ids(asset)
        body_pos = asset.data.body_link_pos_w
        centers = []
        for side in sides:
            ankle_id, toe_id = body_ids[side]
            centers.append(0.5 * (body_pos[:, ankle_id, :] + body_pos[:, toe_id, :]))
        return torch.stack(centers, dim=1)

    def _body_ids(self, asset: Entity) -> dict[str, tuple[int, int]]:
        cache_key = id(asset)
        cached = self._body_id_cache.get(cache_key)
        if cached is not None:
            return cached
        body_names = list(asset.body_names)
        ids = {
            side: (body_names.index(names[0]), body_names.index(names[1]))
            for side, names in FOOT_BODY_PAIRS.items()
        }
        self._body_id_cache[cache_key] = ids
        return ids


class mocap_foot_event_tracking_reward(mocap_foot_pos_tracking_reward):
    """Track elevated mocap touchdown/support events for the matching foot only."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._event_cache: dict[tuple[str, str, str, float, float, int, float], torch.Tensor] = {}

    def __call__(
        self,
        env,
        motion_file: str = str(DEFAULT_MOCAP_MOTION),
        episode: str = "ep_0",
        robot_xml: str = "",
        std: float = 0.18,
        xy_weight: float = 1.0,
        z_weight: float = 1.0,
        stance_speed_threshold: float = 0.18,
        min_stance_frames: int = 4,
        min_event_height_delta: float = 0.05,
        window_margin_frames: int = 0,
    ) -> torch.Tensor:
        if std <= 0.0:
            raise ValueError("mocap foot event tracking std must be positive")
        events = self._reference_events(
            motion_file,
            episode,
            robot_xml,
            stance_speed_threshold,
            min_stance_frames,
            min_event_height_delta,
        )
        if events.numel() == 0:
            return torch.zeros(env.episode_length_buf.shape[0], device=self._device)

        phase = torch.remainder(env.episode_length_buf - 1, self._reference_feet(motion_file, episode, robot_xml).shape[0])
        start = events[:, 1][None, :] - int(window_margin_frames)
        end = events[:, 2][None, :] + int(window_margin_frames)
        active = (phase[:, None] >= start) & (phase[:, None] <= end)
        if not bool(torch.any(active)):
            return torch.zeros(phase.shape[0], device=self._device)

        current = self._current_foot_centers(env, ("L", "R")).clone()
        current[:, :, :2] -= _root_xy_origin(env, current[:, 0, :])[:, None, :]
        side_ids = events[:, 0].long()
        current_for_event = current[:, side_ids, :]
        target = events[:, 3:6][None, :, :]
        xy_error = torch.sum(torch.square(current_for_event[:, :, :2] - target[:, :, :2]), dim=-1)
        z_error = torch.square(current_for_event[:, :, 2] - target[:, :, 2])
        event_reward = torch.exp(-(float(xy_weight) * xy_error + float(z_weight) * z_error) / (std * std))
        event_reward = event_reward * active.to(dtype=event_reward.dtype)
        active_count = torch.clamp(active.sum(dim=1), min=1).to(dtype=event_reward.dtype)
        return event_reward.sum(dim=1) / active_count

    def _reference_events(
        self,
        motion_file: str,
        episode: str,
        robot_xml: str,
        stance_speed_threshold: float,
        min_stance_frames: int,
        min_event_height_delta: float,
    ) -> torch.Tensor:
        key = (
            motion_file,
            episode,
            robot_xml,
            self._target_dt,
            float(stance_speed_threshold),
            int(min_stance_frames),
            float(min_event_height_delta),
        )
        cached = self._event_cache.get(key)
        if cached is not None:
            return cached
        reference = self._reference_feet(motion_file, episode, robot_xml).detach().cpu()
        events = _mocap_foot_events_from_centers(
            reference,
            target_dt=self._target_dt,
            stance_speed_threshold=stance_speed_threshold,
            min_stance_frames=min_stance_frames,
            min_event_height_delta=min_event_height_delta,
        )
        self._event_cache[key] = events.to(device=self._device, dtype=torch.float32)
        return self._event_cache[key]


def _root_xy_origin(env, sample: torch.Tensor) -> torch.Tensor:
    scene = getattr(env, "scene", None)
    try:
        asset = scene["robot"] if scene is not None else None
    except (KeyError, TypeError):
        asset = None
    default_root_state = getattr(getattr(asset, "data", None), "default_root_state", None)
    if default_root_state is not None and default_root_state.shape[0] == sample.shape[0]:
        origin = default_root_state[:, :2].to(device=sample.device, dtype=sample.dtype)
        env_origins = getattr(scene, "env_origins", None)
        if env_origins is not None and env_origins.shape[0] == sample.shape[0]:
            origin = origin + env_origins[:, :2].to(device=sample.device, dtype=sample.dtype)
        return origin
    qpos = getattr(getattr(getattr(env, "sim", None), "data", None), "qpos", None)
    if qpos is not None and qpos.shape[0] == sample.shape[0] and qpos.shape[1] >= 2:
        return qpos[:, :2].to(device=sample.device, dtype=sample.dtype).detach()
    return torch.zeros(sample.shape[0], 2, device=sample.device, dtype=sample.dtype)


def _load_resampled_mocap_qpos(path: Path, episode: str, target_dt: float) -> torch.Tensor:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float32)
        source_dt = float(ep.attrs.get("dt", hf.attrs.get("dt", 1.0 / 30.0)))
    source_t = np.arange(qpos.shape[0], dtype=np.float32) * source_dt
    target_steps = max(2, int(np.ceil(source_t[-1] / target_dt)) + 1)
    target_t = np.arange(target_steps, dtype=np.float32) * target_dt
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    out = np.empty((target_steps, qpos.shape[1]), dtype=np.float32)
    for col in range(qpos.shape[1]):
        out[:, col] = np.interp(target_t, source_t, qpos[:, col]).astype(np.float32)
    out[:, :2] -= out[0:1, :2]
    return torch.from_numpy(out)


def _mocap_foot_events_from_centers(
    centers: torch.Tensor,
    *,
    target_dt: float,
    stance_speed_threshold: float,
    min_stance_frames: int,
    min_event_height_delta: float,
) -> torch.Tensor:
    if centers.ndim != 3 or centers.shape[1:] != (2, 3):
        raise ValueError("centers must have shape (frames, 2, 3)")
    if stance_speed_threshold <= 0.0:
        raise ValueError("stance_speed_threshold must be positive")
    if min_stance_frames <= 0:
        raise ValueError("min_stance_frames must be positive")
    speed_scale = 1.0 / max(float(target_dt), 1.0e-6)
    ground_z = torch.amin(centers[:, :, 2]).item()
    rows: list[list[float]] = []
    for side_idx in range(2):
        foot = centers[:, side_idx, :]
        if foot.shape[0] < 2:
            continue
        velocity_xy = torch.gradient(foot[:, :2], dim=0)[0]
        speed_xy = torch.linalg.norm(velocity_xy, dim=1) * speed_scale
        stance = speed_xy < float(stance_speed_threshold)
        for start, end in _contiguous_true_ranges(stance):
            if end - start < int(min_stance_frames) or start <= 1:
                continue
            center = torch.median(foot[start:end], dim=0).values
            if float(center[2] - ground_z) < float(min_event_height_delta):
                continue
            rows.append([float(side_idx), float(start), float(end - 1), *[float(v) for v in center]])
    rows.sort(key=lambda item: (item[1], item[0]))
    if not rows:
        return torch.empty(0, 6, dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _contiguous_true_ranges(mask: torch.Tensor) -> list[tuple[int, int]]:
    values = mask.detach().cpu().numpy().astype(bool)
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append((start, idx))
            start = None
    if start is not None:
        ranges.append((start, len(values)))
    return ranges


def _load_resampled_mocap_foot_centers(
    path: Path,
    episode: str,
    target_dt: float,
    robot_xml: Path,
) -> torch.Tensor:
    import h5py
    import mujoco
    import numpy as np

    if not robot_xml:
        raise ValueError("robot_xml is required for mocap foot tracking")

    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float64)
        source_dt = float(ep.attrs.get("dt", hf.attrs.get("dt", 1.0 / 30.0)))

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    body_ids = {
        side: tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        )
        for side, body_names in FOOT_BODY_PAIRS.items()
    }
    if any(body_id < 0 for ids in body_ids.values() for body_id in ids):
        raise ValueError(f"Missing foot bodies in {robot_xml}")

    centers = np.empty((qpos.shape[0], 2, 3), dtype=np.float32)
    for frame, pos in enumerate(qpos):
        data.qpos[:] = pos
        mujoco.mj_forward(model, data)
        for side_idx, side in enumerate(("L", "R")):
            ankle_id, toe_id = body_ids[side]
            centers[frame, side_idx] = 0.5 * (data.xpos[ankle_id] + data.xpos[toe_id])
    centers[:, :, :2] -= qpos[0:1, None, :2]

    source_t = np.arange(centers.shape[0], dtype=np.float32) * source_dt
    target_steps = max(2, int(np.ceil(source_t[-1] / target_dt)) + 1)
    target_t = np.arange(target_steps, dtype=np.float32) * target_dt
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    out = np.empty((target_steps, 2, 3), dtype=np.float32)
    for side in range(2):
        for axis in range(3):
            out[:, side, axis] = np.interp(target_t, source_t, centers[:, side, axis]).astype(np.float32)
    return torch.from_numpy(out)


def root_height_below(
    env,
    min_height: float = 0.5,
    asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height
