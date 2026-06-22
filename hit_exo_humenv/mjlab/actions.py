from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from hit_exo_humenv.latent_z_config import cfg_tuple
from hit_exo_humenv.s1_policy import DEFAULT_S1_BUFFER, DEFAULT_S1_MODEL_ID, MetamotivoS1Policy


@dataclass(kw_only=True)
class S1HumanActionCfg(ActionTermCfg):
    """Drive the human with a frozen MetaMotivo S-1 policy without consuming RL actions."""

    task: str = "move-ego-0-2"
    speed_command_name: str | None = None
    target_angle_deg: float = 0.0
    speed_bins: tuple[float, ...] = cfg_tuple("walking_command", "speed_choices")
    direction_bins_deg: tuple[float, ...] = cfg_tuple("walking_command", "direction_choices_deg")
    model_id: str = DEFAULT_S1_MODEL_ID
    buffer_path: str = str(DEFAULT_S1_BUFFER)
    device: str | None = None
    num_samples_per_inference: int = 100_000
    max_workers: int = 12
    mean_action: bool = True
    latent_speed_scale: float = 1.0
    action_repeat: int = 1
    action_smoothing: float = 0.0
    latent_cache_dir: str = ".cache/hit_exo_humenv/s1_latents"
    tracking_motion_file: str | None = None
    tracking_episode: str = "ep_0"
    residual_joint_names: tuple[str, ...] = ()
    residual_scale: float = 0.0
    residual_policy_checkpoint: str | None = None
    mocap_assist_start: float = 0.0
    mocap_assist_end: float = 0.0
    mocap_assist_decay_steps: int = 1
    mocap_assist_position_gain: float = 1.0
    mocap_assist_velocity_gain: float = 0.05
    mocap_assist_max_action: float = 0.5

    def build(self, env):
        return S1HumanAction(self, env)


class S1HumanAction(ActionTerm):
    cfg: S1HumanActionCfg

    def __init__(self, cfg: S1HumanActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        self._residual_joint_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._residual_joint_names: tuple[str, ...] = ()
        self._residual_policy: _ResidualMlpPolicy | None = None
        self._knee_joint_ids = torch.empty(0, dtype=torch.long, device=self.device)
        if cfg.residual_joint_names and cfg.residual_scale > 0.0:
            joint_ids, joint_names = self._entity.find_joints(cfg.residual_joint_names, preserve_order=True)
            self._residual_joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
            self._residual_joint_names = tuple(joint_names)
            knee_ids, _knee_names = self._entity.find_joints(("L_Knee_x", "R_Knee_x"), preserve_order=True)
            self._knee_joint_ids = torch.as_tensor(knee_ids, dtype=torch.long, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, len(self._residual_joint_ids), device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._env.sim.mj_model.nu, device=self.device)
        self._held_actions = torch.zeros_like(self._processed_actions)
        self._residual_action_full = torch.zeros_like(self._processed_actions)
        self._mocap_assist_action = torch.zeros_like(self._processed_actions)
        self._mocap_assist_replacement_target = torch.zeros_like(self._processed_actions)
        self._steps_until_update = 0
        self._mocap_assist_qpos: torch.Tensor | None = None
        self._mocap_assist_qvel: torch.Tensor | None = None
        if cfg.tracking_motion_file and (cfg.mocap_assist_start > 0.0 or cfg.mocap_assist_end > 0.0):
            qpos, qvel = _load_resampled_mocap_state(
                cfg.tracking_motion_file,
                cfg.tracking_episode,
                self._env.step_dt,
            )
            self._mocap_assist_qpos = qpos.to(device=self.device, dtype=torch.float32)
            self._mocap_assist_qvel = qvel.to(device=self.device, dtype=torch.float32)
        if cfg.residual_policy_checkpoint:
            if len(self._residual_joint_ids) == 0:
                raise ValueError("residual_policy_checkpoint requires residual_joint_names and residual_scale")
            self._residual_policy = _ResidualMlpPolicy.load(
                cfg.residual_policy_checkpoint,
                expected_output_dim=len(self._residual_joint_ids),
                device=self.device,
            )
        self._policy = MetamotivoS1Policy(
            task=cfg.task,
            model_id=cfg.model_id,
            buffer_path=cfg.buffer_path,
            device=cfg.device or self.device,
            num_samples_per_inference=cfg.num_samples_per_inference,
            max_workers=cfg.max_workers,
            mean_action=cfg.mean_action,
            latent_cache_dir=cfg.latent_cache_dir,
        )
        self._speed_bins = torch.as_tensor(cfg.speed_bins, device=self.device, dtype=torch.float32)
        self._direction_bins = torch.as_tensor(cfg.direction_bins_deg, device=self.device, dtype=torch.float32)
        self._latent_table: torch.Tensor | None = None
        self._tracking_latents = self._load_tracking_latents() if cfg.tracking_motion_file else None

    @property
    def action_dim(self) -> int:
        if self._residual_policy is not None:
            return 0
        return self._raw_actions.shape[1]

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def residual_action_full(self) -> torch.Tensor:
        return self._residual_action_full

    @property
    def mocap_assist_replacement_target(self) -> torch.Tensor:
        return self._mocap_assist_replacement_target

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._steps_until_update > 0:
            self._steps_until_update -= 1
            self._processed_actions[:] = self._held_actions
            return

        if self._residual_policy is not None:
            self._raw_actions[:] = torch.clamp(
                self._residual_policy(self._compensation_policy_obs()),
                -1.0,
                1.0,
            )
        elif self.action_dim > 0:
            self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)

        obs = _compute_humenv_proprio_obs(self._entity)
        if self._tracking_latents is not None:
            z = self._tracking_latents_for_current_phase()
        elif self.cfg.speed_command_name:
            z = self._commanded_latents()
        else:
            z = None
        action_tensor = self._policy.act_tensor(obs, z=z).to(device=self.device, dtype=torch.float32)
        action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
        smoothing = min(max(self.cfg.action_smoothing, 0.0), 0.99)
        if smoothing > 0.0:
            action_tensor = smoothing * self._held_actions + (1.0 - smoothing) * action_tensor
        self._residual_action_full.zero_()
        if len(self._residual_joint_ids) > 0:
            residual = self._raw_actions * float(self.cfg.residual_scale)
            self._residual_action_full[:, self._residual_joint_ids] = residual
            action_tensor[:, self._residual_joint_ids] += residual
        self._mocap_assist_action.zero_()
        self._mocap_assist_replacement_target.zero_()
        if self._mocap_assist_qpos is not None and self._mocap_assist_qvel is not None:
            assist_action, replacement_target = self._mocap_pd_assist_actions()
            self._mocap_assist_action[:] = assist_action
            self._mocap_assist_replacement_target[:] = replacement_target
            action_tensor += assist_action
        if len(self._residual_joint_ids) > 0 or self._mocap_assist_qpos is not None:
            action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
        self._processed_actions[:] = action_tensor
        self._held_actions[:] = action_tensor
        self._steps_until_update = max(0, self.cfg.action_repeat - 1)

    def apply_actions(self) -> None:
        self._entity.set_joint_effort_target(self._processed_actions)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = 0.0
        self._held_actions[env_ids] = 0.0
        self._residual_action_full[env_ids] = 0.0
        self._mocap_assist_action[env_ids] = 0.0
        self._mocap_assist_replacement_target[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._steps_until_update = 0

    def _load_tracking_latents(self) -> torch.Tensor:
        assert self.cfg.tracking_motion_file is not None
        latents = self._policy.z_for_tracking_motion(
            self.cfg.tracking_motion_file,
            episode=self.cfg.tracking_episode,
        )
        return latents.to(device=self.device, dtype=torch.float32)

    def _tracking_latents_for_current_phase(self) -> torch.Tensor:
        assert self._tracking_latents is not None
        phase = torch.remainder(
            self._env.episode_length_buf,
            self._tracking_latents.shape[0],
        ).long()
        return self._tracking_latents[phase]

    def _commanded_latents(self) -> torch.Tensor:
        assert self.cfg.speed_command_name is not None
        if self._latent_table is None:
            self._latent_table = self._build_command_latent_table()
        command = self._env.command_manager.get_command(self.cfg.speed_command_name)
        speed = command[:, 0].to(device=self.device, dtype=torch.float32)
        direction = (
            command[:, 1].to(device=self.device, dtype=torch.float32)
            if command.shape[1] > 1
            else torch.full_like(speed, self.cfg.target_angle_deg)
        )
        nearest_speed = torch.argmin(torch.abs(speed[:, None] - self._speed_bins[None, :]), dim=1)
        nearest_direction = torch.argmin(
            torch.abs(direction[:, None] - self._direction_bins[None, :]), dim=1
        )
        flat_index = nearest_speed * len(self.cfg.direction_bins_deg) + nearest_direction
        flat_table = self._latent_table.reshape(-1, self._latent_table.shape[-1])
        return flat_table[flat_index]

    def _build_command_latent_table(self) -> torch.Tensor:
        rows = []
        for speed in self.cfg.speed_bins:
            cols = []
            for direction in self.cfg.direction_bins_deg:
                task = f"move-ego-{direction:g}-{speed * self.cfg.latent_speed_scale:g}"
                latent = self._policy.z_for_task(task).to(device=self.device, dtype=torch.float32).reshape(-1)
                cols.append(latent)
            rows.append(torch.stack(cols, dim=0))
        return torch.stack(rows, dim=0)

    def _compensation_policy_obs(self) -> torch.Tensor:
        joint_pos = self._entity.data.joint_pos - self._entity.data.default_joint_pos
        joint_vel = self._entity.data.joint_vel - self._entity.data.default_joint_vel
        return torch.cat(
            [
                self._entity.data.root_link_lin_vel_b,
                self._entity.data.root_link_ang_vel_b,
                self._entity.data.projected_gravity_b,
                joint_pos,
                joint_vel,
                joint_pos[:, self._knee_joint_ids],
                joint_vel[:, self._knee_joint_ids],
                self._raw_actions,
            ],
            dim=-1,
        ).to(dtype=torch.float32)

    def _mocap_pd_assist_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._mocap_assist_qpos is not None
        assert self._mocap_assist_qvel is not None
        phase = torch.remainder(self._env.episode_length_buf, self._mocap_assist_qpos.shape[0]).long()
        target_joint_pos = self._mocap_assist_qpos[phase, 7 : 7 + self._entity.data.joint_pos.shape[1]]
        target_joint_vel = self._mocap_assist_qvel[phase, 6 : 6 + self._entity.data.joint_vel.shape[1]]
        pd_action = (
            float(self.cfg.mocap_assist_position_gain) * (target_joint_pos - self._entity.data.joint_pos)
            + float(self.cfg.mocap_assist_velocity_gain) * (target_joint_vel - self._entity.data.joint_vel)
        )
        max_action = max(float(self.cfg.mocap_assist_max_action), 0.0)
        if max_action > 0.0:
            pd_action = torch.clamp(pd_action, -max_action, max_action)
        strength = self._mocap_assist_strength()
        start = max(float(self.cfg.mocap_assist_start), float(self.cfg.mocap_assist_end))
        replacement_strength = max(start - strength, 0.0)
        return pd_action * strength, pd_action * replacement_strength

    def _mocap_assist_strength(self) -> float:
        start = float(self.cfg.mocap_assist_start)
        end = float(self.cfg.mocap_assist_end)
        decay_steps = max(int(self.cfg.mocap_assist_decay_steps), 1)
        step = float(getattr(self._env, "common_step_counter", 0))
        blend = min(max(step / float(decay_steps), 0.0), 1.0)
        return start + (end - start) * blend


class _ResidualMlpPolicy(torch.nn.Module):
    def __init__(
        self,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        weights: list[tuple[torch.Tensor, torch.Tensor]],
        device: str,
    ) -> None:
        super().__init__()
        self.register_buffer("_mean", mean.to(device=device, dtype=torch.float32))
        self.register_buffer("_std", std.to(device=device, dtype=torch.float32))
        self._layers = torch.nn.ModuleList()
        for weight, bias in weights:
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight.to(device=device, dtype=torch.float32))
            layer.bias.data.copy_(bias.to(device=device, dtype=torch.float32))
            self._layers.append(layer.to(device))
        self.eval()

    @classmethod
    def load(cls, checkpoint: str, *, expected_output_dim: int, device: str) -> "_ResidualMlpPolicy":
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Residual policy checkpoint not found: {path}")
        loaded = torch.load(path, map_location=device, weights_only=False)
        actor_state = loaded.get("actor_state_dict")
        if not actor_state:
            raise KeyError(f"{path} does not contain actor_state_dict")

        mean = actor_state["obs_normalizer._mean"]
        std = actor_state["obs_normalizer._std"]
        weights = []
        layer_indices = sorted(
            int(key.split(".")[1])
            for key in actor_state
            if key.startswith("mlp.") and key.endswith(".weight")
        )
        for idx in layer_indices:
            weights.append((actor_state[f"mlp.{idx}.weight"], actor_state[f"mlp.{idx}.bias"]))
        output_dim = int(weights[-1][0].shape[0])
        if output_dim != expected_output_dim:
            raise ValueError(
                "Residual policy output dimension does not match configured joints: "
                f"checkpoint={output_dim} configured={expected_output_dim}"
            )
        return cls(mean=mean, std=std, weights=weights, device=device)

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self._mean) / (self._std + 1.0e-2)
        for layer in self._layers[:-1]:
            x = torch.nn.functional.elu(layer(x))
        return self._layers[-1](x)


def _load_resampled_mocap_state(path: str, episode: str, target_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as hf:
        ep = hf[episode]
        qpos = np.asarray(ep["qpos"][:], dtype=np.float32)
        qvel = np.asarray(ep["qvel"][:], dtype=np.float32)
        source_dt = float(ep.attrs.get("dt", hf.attrs.get("dt", 1.0 / 30.0)))
    source_t = np.arange(qpos.shape[0], dtype=np.float32) * source_dt
    target_steps = max(2, int(np.ceil(source_t[-1] / target_dt)) + 1)
    target_t = np.arange(target_steps, dtype=np.float32) * target_dt
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    qpos_out = np.empty((target_steps, qpos.shape[1]), dtype=np.float32)
    qvel_out = np.empty((target_steps, qvel.shape[1]), dtype=np.float32)
    for col in range(qpos.shape[1]):
        qpos_out[:, col] = np.interp(target_t, source_t, qpos[:, col]).astype(np.float32)
    for col in range(qvel.shape[1]):
        qvel_out[:, col] = np.interp(target_t, source_t, qvel[:, col]).astype(np.float32)
    return torch.from_numpy(qpos_out), torch.from_numpy(qvel_out)


@dataclass(kw_only=True)
class KneeExoTorqueActionCfg(ActionTermCfg):
    """Apply normalized exoskeleton torques to configured DoFs via qfrc_applied."""

    joint_names: tuple[str, ...] = ("L_Knee_x", "R_Knee_x")
    max_torque: float | tuple[float, ...] = 80.0

    def build(self, env):
        return KneeExoTorqueAction(self, env)


class KneeExoTorqueAction(ActionTerm):
    cfg: KneeExoTorqueActionCfg

    def __init__(self, cfg: KneeExoTorqueActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        joint_ids, joint_names = self._entity.find_joints(cfg.joint_names, preserve_order=True)
        self._joint_names = tuple(joint_names)
        self._dof_ids = self._entity.data.indexing.joint_v_adr[joint_ids].long()
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_names), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        max_torque = cfg.max_torque
        if isinstance(max_torque, tuple):
            if len(max_torque) != len(self._joint_names):
                raise ValueError(
                    "max_torque tuple length must match joint_names: "
                    f"{len(max_torque)} != {len(self._joint_names)}"
                )
            self._max_torque = torch.as_tensor(max_torque, dtype=torch.float32, device=self.device)
        else:
            self._max_torque = torch.full(
                (len(self._joint_names),),
                float(max_torque),
                dtype=torch.float32,
                device=self.device,
            )

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def dof_ids(self) -> torch.Tensor:
        return self._dof_ids

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._processed_actions[:] = self._raw_actions * self._max_torque

    def apply_actions(self) -> None:
        self._env.sim.data.qfrc_applied[:, self._dof_ids] = self._processed_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        if isinstance(env_ids, slice):
            self._env.sim.data.qfrc_applied[env_ids, self._dof_ids] = 0.0
        else:
            self._env.sim.data.qfrc_applied[env_ids[:, None], self._dof_ids[None, :]] = 0.0


def _compute_humenv_proprio_obs(entity) -> torch.Tensor:
    body_pose = entity.data.body_link_pose_w
    body_vel = entity.data.body_link_vel_w
    body_pos = body_pose[:, :24, :3]
    body_rot = body_pose[:, :24, 3:7]
    body_lin_vel = body_vel[:, :24, :3]
    body_ang_vel = body_vel[:, :24, 3:6]

    root_pos = body_pos[:, 0, :]
    root_rot = _remove_smpl_base_rot(body_rot[:, 0, :])
    heading_rot_inv = _calc_heading_quat_inv(root_rot)

    num_envs, num_bodies, _ = body_pos.shape
    heading_flat = heading_rot_inv[:, None, :].expand(num_envs, num_bodies, 4).reshape(-1, 4)

    root_h_obs = root_pos[:, 2:3]
    local_body_pos = body_pos - root_pos[:, None, :]
    local_body_pos = quat_apply(heading_flat, local_body_pos.reshape(-1, 3)).reshape(num_envs, num_bodies * 3)
    local_body_pos = local_body_pos[:, 3:]

    local_body_rot = quat_mul(heading_flat, body_rot.reshape(-1, 4))
    local_body_rot_obs = _quat_to_tan_norm(local_body_rot).reshape(num_envs, num_bodies * 6)

    local_body_vel = quat_apply(heading_flat, body_lin_vel.reshape(-1, 3)).reshape(num_envs, num_bodies * 3)
    local_body_ang_vel = quat_apply(heading_flat, body_ang_vel.reshape(-1, 3)).reshape(num_envs, num_bodies * 3)

    return torch.cat(
        [root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel],
        dim=-1,
    ).to(dtype=torch.float32)


def _remove_smpl_base_rot(quat: torch.Tensor) -> torch.Tensor:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return torch.stack(
        [
            0.5 * (w + x + y + z),
            0.5 * (-w + x - y + z),
            0.5 * (-w + x + y - z),
            0.5 * (-w - x + y + z),
        ],
        dim=-1,
    )


def _calc_heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_heading = -0.5 * heading
    return torch.stack(
        [
            torch.cos(half_heading),
            torch.zeros_like(heading),
            torch.zeros_like(heading),
            torch.sin(half_heading),
        ],
        dim=-1,
    )


def _quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    two = 2.0
    tan = torch.stack(
        [
            1.0 - two * (y * y + z * z),
            two * (x * y + w * z),
            two * (x * z - w * y),
        ],
        dim=-1,
    )
    norm_vec = torch.stack(
        [
            two * (x * z + w * y),
            two * (y * z - w * x),
            1.0 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return torch.cat([tan, norm_vec], dim=-1)
