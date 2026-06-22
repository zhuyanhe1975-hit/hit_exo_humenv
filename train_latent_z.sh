#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
NUM_ENVS="${NUM_ENVS:-$LATENT_Z_TRAIN_NUM_ENVS}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-$LATENT_Z_TRAIN_NUM_STEPS_PER_ENV}"
MAX_ITERATIONS="${MAX_ITERATIONS:-$LATENT_Z_TRAIN_MAX_ITERATIONS}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$LATENT_Z_TRAIN_SAVE_INTERVAL}"
LOGGER="${LOGGER:-$LATENT_Z_TRAIN_LOGGER}"
EXO_JOINT_GROUP="${EXO_JOINT_GROUP:-$LATENT_Z_EXO_JOINT_GROUP}"
export EXO_JOINT_GROUP

conda run --no-capture-output -n "$ENV_NAME" python -m mjlab.scripts.train Mjlab-HumEnv-KneeExo-Walking \
    --env.scene.num-envs "$NUM_ENVS" \
    --env.sim.mujoco.timestep "$LATENT_Z_MUJOCO_TIMESTEP" \
    --env.decimation "$LATENT_Z_DECIMATION" \
    --agent.num-steps-per-env "$NUM_STEPS_PER_ENV" \
    --agent.max-iterations "$MAX_ITERATIONS" \
    --agent.logger "$LOGGER" \
    --agent.save-interval "$SAVE_INTERVAL" \
    "$@"
