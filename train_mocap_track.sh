#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
DEFAULT_MOCAP_DIR="/home/yhzhu/AI/humenv/data_preparation/humenv_from_phc_amass_transitions_kit_cmu"
DEFAULT_MOCAP_MOTION="$DEFAULT_MOCAP_DIR/08_KIT_167_walking_medium_resampled_poses.hdf5"
MOCAP_MOTION="${MOCAP_MOTION:-$DEFAULT_MOCAP_MOTION}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
NUM_ENVS="${NUM_ENVS:-$LATENT_Z_TRAIN_NUM_ENVS}"
MAX_ITERATIONS="${MAX_ITERATIONS:-$LATENT_Z_TRAIN_MAX_ITERATIONS}"
RUN_NAME="${RUN_NAME:-mocap_track}"

conda run --no-capture-output -n "$ENV_NAME" python scripts/train_mjlab_knee_exo_mocap_track.py \
    --motion "$MOCAP_MOTION" \
    --episode "$MOCAP_EPISODE" \
    --num-envs "$NUM_ENVS" \
    --max-iterations "$MAX_ITERATIONS" \
    --run-name "$RUN_NAME" \
    "$@"
