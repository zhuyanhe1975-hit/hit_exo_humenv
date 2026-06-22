from __future__ import annotations

import json
import shlex
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "latent_z.json"


def emit(name: str, value) -> None:
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"{name}={shlex.quote(str(value))}")


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    emit("LATENT_Z_ENV_NAME", cfg["environment"]["conda_env"])
    emit("LATENT_Z_CHECKPOINT_ROOT", cfg["paths"]["checkpoint_root"])
    emit("LATENT_Z_POWER_OUTPUT_ROOT", cfg["paths"]["latent_z_power_output_root"])
    emit("LATENT_Z_POWER_RUNS_ROOT", cfg["paths"]["latent_z_power_runs_root"])
    emit("LATENT_Z_VIEWER_POWER_LOG_ROOT", cfg["paths"]["viewer_power_log_root"])
    emit("LATENT_Z_LEGACY_EVAL_ROOT", cfg["paths"]["legacy_eval_root"])
    emit("LATENT_Z_S1_LATENT_SELECTION_ROOT", cfg["paths"]["s1_latent_selection_root"])
    emit("LATENT_Z_TRAIN_NUM_ENVS", cfg["train"]["num_envs"])
    emit("LATENT_Z_TRAIN_NUM_STEPS_PER_ENV", cfg["train"]["num_steps_per_env"])
    emit("LATENT_Z_TRAIN_MAX_ITERATIONS", cfg["train"]["max_iterations"])
    emit("LATENT_Z_TRAIN_SAVE_INTERVAL", cfg["train"]["save_interval"])
    emit("LATENT_Z_TRAIN_LOGGER", cfg["train"]["logger"])
    emit("LATENT_Z_EVAL_NUM_ENVS", cfg["eval"]["num_envs"])
    emit("LATENT_Z_EVAL_STEPS", cfg["eval"]["steps"])
    emit("LATENT_Z_EVAL_SEED", cfg["eval"]["seed"])
    emit("LATENT_Z_EVAL_DROP_FALLEN", cfg["eval"]["drop_fallen"])
    emit("LATENT_Z_VIEWER_NUM_ENVS", cfg["viewer"]["num_envs"])
    emit("LATENT_Z_ENV_SPACING", cfg["simulation"]["env_spacing"])
    emit("LATENT_Z_MUJOCO_TIMESTEP", cfg["simulation"]["mujoco_timestep"])
    emit("LATENT_Z_DECIMATION", cfg["simulation"]["decimation"])
    emit("LATENT_Z_NORMAL_WALK_SPEED", cfg["walking_command"]["normal_speed"])
    emit("LATENT_Z_RANDOM_WALK_SPEED", cfg["eval"]["random_walk_speed"])
    emit("LATENT_Z_RANDOM_WALK_DIRECTION", cfg["eval"]["random_walk_direction"])
    emit("LATENT_Z_WALK_DIRECTION", cfg["walking_command"]["direction_choices_deg"][0])
    emit("LATENT_Z_S1_LATENT_SPEED_SCALE", cfg["human_s1"]["latent_speed_scale"])
    emit("LATENT_Z_HUMAN_ACTION_REPEAT", cfg["human_s1"]["action_repeat"])
    emit("LATENT_Z_HUMAN_ACTION_SMOOTHING", cfg["human_s1"]["action_smoothing"])
    emit("LATENT_Z_HUMAN_ROOT_HEIGHT", cfg["human_s1"]["root_height"])
    emit("LATENT_Z_EXO_JOINT_GROUP", cfg["exo"]["joint_group"])


if __name__ == "__main__":
    main()
