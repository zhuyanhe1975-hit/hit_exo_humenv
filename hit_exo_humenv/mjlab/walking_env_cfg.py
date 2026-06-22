from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
    reset_scene_to_default,
    time_out,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

import hit_exo_humenv.mjlab.mdp as mdp
from hit_exo_humenv.latent_z_config import cfg_path, cfg_tuple
from hit_exo_humenv.mjlab.actions import KneeExoTorqueActionCfg, S1HumanActionCfg
from hit_exo_humenv.mjlab.commands import WalkingSpeedCommandCfg


TASK_ID = cfg_path("task_id")
HUMENV_XML = Path(cfg_path("environment", "humenv_xml"))
NORMAL_WALKING_SPEED = cfg_path("walking_command", "normal_speed")
TRAIN_WALKING_SPEED_RANGE = cfg_tuple("walking_command", "speed_range")
TRAIN_WALKING_SPEED_CHOICES = cfg_tuple("walking_command", "speed_choices")
TRAIN_WALKING_DIRECTION_CHOICES_DEG = cfg_tuple("walking_command", "direction_choices_deg")
EXO_JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "knee": ("L_Knee_x", "R_Knee_x"),
    "hip": ("L_Hip_x", "R_Hip_x"),
    "ankle": ("L_Ankle_x", "R_Ankle_x"),
    "hip_knee": ("L_Hip_x", "R_Hip_x", "L_Knee_x", "R_Knee_x"),
    "knee_ankle": ("L_Knee_x", "R_Knee_x", "L_Ankle_x", "R_Ankle_x"),
    "lower_limb": ("L_Hip_x", "R_Hip_x", "L_Knee_x", "R_Knee_x", "L_Ankle_x", "R_Ankle_x"),
}


def exo_joint_group() -> str:
    group = os.environ.get("EXO_JOINT_GROUP", cfg_path("exo", "joint_group"))
    if group not in EXO_JOINT_GROUPS:
        supported = ", ".join(sorted(EXO_JOINT_GROUPS))
        raise ValueError(f"Unsupported EXO_JOINT_GROUP={group!r}. Supported groups: {supported}")
    return group


def exo_joint_names(group: str | None = None) -> tuple[str, ...]:
    return EXO_JOINT_GROUPS[group or exo_joint_group()]


def exo_torque_limits(group: str | None = None) -> tuple[float, ...]:
    limits = {
        "Hip": float(cfg_path("exo", "max_hip_torque")),
        "Knee": float(cfg_path("exo", "max_knee_torque")),
        "Ankle": float(cfg_path("exo", "max_ankle_torque")),
    }
    return tuple(limits[_joint_family(name)] for name in exo_joint_names(group))


def _joint_family(joint_name: str) -> str:
    for family in ("Hip", "Knee", "Ankle"):
        if family in joint_name:
            return family
    raise ValueError(f"Unsupported exoskeleton joint name: {joint_name}")


def _humenv_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HUMENV_XML))


def _humenv_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_humenv_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=(".*",), command_field="effort"),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, cfg_path("human_s1", "root_height")),
            rot=(0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        sort_actuators=False,
    )


def humenv_knee_exo_walking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = SceneEntityCfg("robot")
    knee_cfg = SceneEntityCfg("robot", joint_names=("L_Knee_x", "R_Knee_x"))
    hip_power_cfg = SceneEntityCfg("robot", joint_names=("L_Hip_.*", "R_Hip_.*"))
    knee_power_cfg = SceneEntityCfg("robot", joint_names=("L_Knee_.*", "R_Knee_.*"))
    ankle_power_cfg = SceneEntityCfg("robot", joint_names=("L_Ankle_.*", "R_Ankle_.*"))
    fall_min_height = cfg_path("reward", "fall_min_height")
    power_weights = cfg_path("reward", "power_joint_weights")
    smoothness_weights = cfg_path("reward", "smoothness_joint_weights")

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel, params={"asset_cfg": robot_cfg}),
        "base_ang_vel": ObservationTermCfg(func=base_ang_vel, params={"asset_cfg": robot_cfg}),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity,
            params={"asset_cfg": robot_cfg},
        ),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": robot_cfg}),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": robot_cfg}),
        "knee_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": knee_cfg}),
        "knee_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": knee_cfg}),
        "walk_speed": ObservationTermCfg(
            func=mdp.commanded_forward_speed,
            params={"command_name": "walk_speed"},
        ),
        "last_action": ObservationTermCfg(func=last_action),
    }

    speed_range = TRAIN_WALKING_SPEED_RANGE
    speed_choices = TRAIN_WALKING_SPEED_CHOICES
    actions = {
        "human_s1": S1HumanActionCfg(
            entity_name="robot",
            task=f"move-ego-0-{NORMAL_WALKING_SPEED:g}",
            speed_command_name="walk_speed",
            target_angle_deg=0.0,
            speed_bins=speed_choices,
            direction_bins_deg=TRAIN_WALKING_DIRECTION_CHOICES_DEG,
            num_samples_per_inference=cfg_path("human_s1", "num_samples_per_inference"),
            max_workers=cfg_path("human_s1", "max_workers"),
            latent_speed_scale=cfg_path("human_s1", "latent_speed_scale"),
            action_repeat=cfg_path("human_s1", "action_repeat"),
            action_smoothing=cfg_path("human_s1", "action_smoothing"),
        ),
        "knee_exo": KneeExoTorqueActionCfg(
            entity_name="robot",
            joint_names=exo_joint_names(),
            max_torque=exo_torque_limits(),
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=128 if not play else 1,
            env_spacing=cfg_path("simulation", "env_spacing"),
            entities={"robot": _humenv_robot_cfg()},
            terrain=None,
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
        },
        actions=actions,
        commands={
            "walk_speed": WalkingSpeedCommandCfg(
                resampling_time_range=cfg_tuple("walking_command", "resampling_time_range"),
                speed_range=speed_range,
                speed_choices=speed_choices,
                include_direction=True,
                direction_choices_deg=TRAIN_WALKING_DIRECTION_CHOICES_DEG,
                reset_on_resample=cfg_path("walking_command", "reset_on_resample"),
            ),
        },
        events={
            "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        },
        rewards={
            "not_fallen_progress": RewardTermCfg(
                func=mdp.not_fallen_progress_reward,
                weight=cfg_path("reward", "not_fallen_progress_weight"),
                params={"min_height": fall_min_height, "asset_cfg": robot_cfg},
            ),
            "fallen_penalty": RewardTermCfg(
                func=mdp.root_height_below,
                weight=cfg_path("reward", "fallen_penalty_weight"),
                params={"min_height": fall_min_height, "asset_cfg": robot_cfg},
            ),
            "lower_limb_joint_power_cost": RewardTermCfg(
                func=mdp.cached_lower_limb_joint_power_cost,
                weight=cfg_path("reward", "lower_limb_joint_power_weight"),
                params={
                    "hip_cfg": hip_power_cfg,
                    "knee_cfg": knee_power_cfg,
                    "ankle_cfg": ankle_power_cfg,
                    "hip_weight": power_weights["hip"],
                    "knee_weight": power_weights["knee"],
                    "ankle_weight": power_weights["ankle"],
                    "positive_efficiency": cfg_path("metabolic", "positive_work_efficiency"),
                    "negative_efficiency": cfg_path("metabolic", "negative_work_efficiency"),
                },
            ),
            "lower_limb_joint_velocity_delta_l2": RewardTermCfg(
                func=mdp.lower_limb_joint_velocity_delta_l2,
                weight=cfg_path("reward", "lower_limb_joint_velocity_delta_weight"),
                params={
                    "hip_cfg": hip_power_cfg,
                    "knee_cfg": knee_power_cfg,
                    "ankle_cfg": ankle_power_cfg,
                    "hip_weight": smoothness_weights["hip"],
                    "knee_weight": smoothness_weights["knee"],
                    "ankle_weight": smoothness_weights["ankle"],
                },
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "fallen": TerminationTermCfg(
                func=mdp.root_height_below,
                params={"min_height": fall_min_height, "asset_cfg": robot_cfg},
            ),
        },
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=cfg_path("simulation", "mujoco_timestep"),
                disableflags=("nativeccd", "multiccd"),
            )
        ),
        decimation=cfg_path("simulation", "decimation"),
        episode_length_s=10.0 if not play else 1e9,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="Pelvis",
            distance=4.0,
            elevation=-12.0,
            azimuth=135.0,
        ),
    )


def humenv_knee_exo_walking_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="humenv_knee_exo_walking",
        save_interval=cfg_path("train", "save_interval"),
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
        num_steps_per_env=cfg_path("train", "num_steps_per_env"),
        max_iterations=cfg_path("train", "max_iterations"),
    )
