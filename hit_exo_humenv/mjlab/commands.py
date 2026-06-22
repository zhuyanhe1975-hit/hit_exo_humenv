from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class WalkingSpeedCommandCfg(CommandTermCfg):
    """Walking speed command with an optional egocentric direction in degrees."""

    speed_range: tuple[float, float] = (0.8, 1.8)
    speed_choices: tuple[float, ...] | None = None
    include_direction: bool = False
    direction_range_deg: tuple[float, float] = (-90.0, 90.0)
    direction_choices_deg: tuple[float, ...] | None = None
    reset_on_resample: bool = False

    def build(self, env: ManagerBasedRlEnv) -> "WalkingSpeedCommand":
        return WalkingSpeedCommand(self, env)


class WalkingSpeedCommand(CommandTerm):
    cfg: WalkingSpeedCommandCfg

    def __init__(self, cfg: WalkingSpeedCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2 if cfg.include_direction else 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def compute(self, dt: float) -> None:
        self._update_metrics()
        self.time_left -= dt
        if self.cfg.resampling_time_range[0] == self.cfg.resampling_time_range[1] >= 1.0e8:
            self._update_command()
            return
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        self._update_command()

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        previous_command = self._command[env_ids].clone()
        if self.cfg.speed_choices is not None:
            choices = torch.as_tensor(self.cfg.speed_choices, device=self.device)
            choice_ids = torch.randint(len(choices), (len(env_ids),), device=self.device)
            self._command[env_ids, 0] = choices[choice_ids]
        else:
            self._command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
                *self.cfg.speed_range
            )

        if self.cfg.include_direction:
            if self.cfg.direction_choices_deg is not None:
                choices = torch.as_tensor(self.cfg.direction_choices_deg, device=self.device)
                choice_ids = torch.randint(len(choices), (len(env_ids),), device=self.device)
                self._command[env_ids, 1] = choices[choice_ids]
            else:
                self._command[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
                    *self.cfg.direction_range_deg
                )
        self._reset_state_after_running_resample(env_ids, previous_command)

    def _update_command(self) -> None:
        pass

    def _reset_state_after_running_resample(
        self,
        env_ids: torch.Tensor,
        previous_command: torch.Tensor,
    ) -> None:
        if not self.cfg.reset_on_resample:
            return
        command_changed = torch.any(self._command[env_ids] != previous_command, dim=1)
        running_env_ids = env_ids[(self.command_counter[env_ids] > 0) & command_changed]
        if len(running_env_ids) == 0:
            return
        # Resampling is on the hot collection path. Re-write the humanoid default
        # state directly instead of doing a full sim/scene reset for these envs.
        reset_scene_to_default(self._env, running_env_ids)
        self._env.action_manager.reset(running_env_ids)
        self._env.scene.write_data_to_sim()
        self._env.sim.forward()
