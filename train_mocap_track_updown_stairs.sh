#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

usage() {
    cat <<'EOF'
Usage: ./train_mocap_track_updown_stairs.sh [extra train_mjlab_knee_exo_mocap_track.py args]

Train an upstairs mocap tracking stage with frozen S-1, stair terrain, and
explicit mocap qpos tracking rewards. Use the stage-specific wrappers for the
recommended two-stage stair pipeline.

Environment overrides:
  ENV_NAME=$LATENT_Z_ENV_NAME
  HUMENV_DATA_ROOT=/home/yhzhu/AI/humenv/data_preparation
  MOCAP_MOTION=$HUMENV_DATA_ROOT/humenv_from_protomotions/KIT_167_upstairs03_poses.hdf5
  MOCAP_EPISODE=ep_0
  TERRAIN=stairs
  TERRAIN_OUT_DIR=.omx/mjlab_updown_stairs_train_terrain
  STAIR_WIDTH=1.4
  STAIR_HEIGHT=0.135
  STAIR_STEPS=0
  NUM_ENVS=$LATENT_Z_TRAIN_NUM_ENVS
  MAX_ITERATIONS=$LATENT_Z_TRAIN_MAX_ITERATIONS
  NUM_STEPS_PER_ENV=$LATENT_Z_TRAIN_NUM_STEPS_PER_ENV
  TRAINING_STAGE=knee-exo
  BASE_COMPENSATION_CHECKPOINT=
  RUN_NAME=stairs_mocap_track
  ACTOR_INIT_STD=
  ENTROPY_COEF=
  HUMAN_RESIDUAL_MODE=none
  HUMAN_RESIDUAL_SCALE=0.0
  HUMAN_RESIDUAL_ACTION_WEIGHT=0.0
  HUMAN_RESIDUAL_ACTION_RATE_WEIGHT=0.0
  MOCAP_ASSIST_REPLACEMENT_WEIGHT=0.0
  MOCAP_ASSIST_START=0.0
  MOCAP_ASSIST_END=0.0
  MOCAP_ASSIST_DECAY_FRACTION=0.0
  MOCAP_ASSIST_DECAY_STEPS=0
  MOCAP_ASSIST_POSITION_GAIN=1.0
  MOCAP_ASSIST_VELOCITY_GAIN=0.05
  MOCAP_ASSIST_MAX_ACTION=0.5
  MOCAP_ROOT_XYZ_WEIGHT=0.4
  MOCAP_ROOT_XYZ_STD=0.25
  MOCAP_ROOT_ORIENTATION_WEIGHT=0.3
  MOCAP_ROOT_ORIENTATION_STD=0.35
  MOCAP_LOWER_BODY_WEIGHT=0.8
  MOCAP_LOWER_BODY_STD=0.45
  MOCAP_KNEE_WEIGHT=1.2
  MOCAP_KNEE_STD=0.25
  MOCAP_FOOT_WEIGHT=0.0
  MOCAP_FOOT_SIDES=L,R
  MOCAP_FOOT_Z_WEIGHT=1.0
  MOCAP_FIRST_FOOT_WEIGHT=0.0
  MOCAP_FIRST_FOOT_SIDES=L,R
  MOCAP_FIRST_FOOT_END_FRAME=90
  MOCAP_FOOT_EVENT_WEIGHT=0.0
  MOCAP_FOOT_EVENT_XY_WEIGHT=1.0
  MOCAP_FOOT_EVENT_Z_WEIGHT=1.0
  CPU=0
  DRY_RUN=1
EOF
}

bool_enabled() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
HUMENV_DATA_ROOT="${HUMENV_DATA_ROOT:-/home/yhzhu/AI/humenv/data_preparation}"
MOCAP_MOTION="${MOCAP_MOTION:-$HUMENV_DATA_ROOT/humenv_from_protomotions/KIT_167_upstairs03_poses.hdf5}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
TERRAIN="${TERRAIN:-stairs}"
TERRAIN_OUT_DIR="${TERRAIN_OUT_DIR:-.omx/mjlab_updown_stairs_train_terrain}"
STAIR_WIDTH="${STAIR_WIDTH:-1.4}"
STAIR_HEIGHT="${STAIR_HEIGHT:-0.135}"
STAIR_STEPS="${STAIR_STEPS:-0}"
NUM_ENVS="${NUM_ENVS:-$LATENT_Z_TRAIN_NUM_ENVS}"
MAX_ITERATIONS="${MAX_ITERATIONS:-$LATENT_Z_TRAIN_MAX_ITERATIONS}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-$LATENT_Z_TRAIN_NUM_STEPS_PER_ENV}"
TRAINING_STAGE="${TRAINING_STAGE:-knee-exo}"
BASE_COMPENSATION_CHECKPOINT="${BASE_COMPENSATION_CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-stairs_mocap_track}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"
ACTOR_INIT_STD="${ACTOR_INIT_STD:-}"
ENTROPY_COEF="${ENTROPY_COEF:-}"
HUMAN_RESIDUAL_MODE="${HUMAN_RESIDUAL_MODE:-none}"
HUMAN_RESIDUAL_SCALE="${HUMAN_RESIDUAL_SCALE:-0.0}"
HUMAN_RESIDUAL_ACTION_WEIGHT="${HUMAN_RESIDUAL_ACTION_WEIGHT:-0.0}"
HUMAN_RESIDUAL_ACTION_RATE_WEIGHT="${HUMAN_RESIDUAL_ACTION_RATE_WEIGHT:-0.0}"
MOCAP_ASSIST_REPLACEMENT_WEIGHT="${MOCAP_ASSIST_REPLACEMENT_WEIGHT:-0.0}"
MOCAP_ASSIST_START="${MOCAP_ASSIST_START:-0.0}"
MOCAP_ASSIST_END="${MOCAP_ASSIST_END:-0.0}"
MOCAP_ASSIST_DECAY_FRACTION="${MOCAP_ASSIST_DECAY_FRACTION:-0.0}"
MOCAP_ASSIST_DECAY_STEPS="${MOCAP_ASSIST_DECAY_STEPS:-0}"
MOCAP_ASSIST_POSITION_GAIN="${MOCAP_ASSIST_POSITION_GAIN:-1.0}"
MOCAP_ASSIST_VELOCITY_GAIN="${MOCAP_ASSIST_VELOCITY_GAIN:-0.05}"
MOCAP_ASSIST_MAX_ACTION="${MOCAP_ASSIST_MAX_ACTION:-0.5}"
MOCAP_ROOT_XYZ_WEIGHT="${MOCAP_ROOT_XYZ_WEIGHT:-0.4}"
MOCAP_ROOT_XYZ_STD="${MOCAP_ROOT_XYZ_STD:-0.25}"
MOCAP_ROOT_ORIENTATION_WEIGHT="${MOCAP_ROOT_ORIENTATION_WEIGHT:-0.3}"
MOCAP_ROOT_ORIENTATION_STD="${MOCAP_ROOT_ORIENTATION_STD:-0.35}"
MOCAP_LOWER_BODY_WEIGHT="${MOCAP_LOWER_BODY_WEIGHT:-0.8}"
MOCAP_LOWER_BODY_STD="${MOCAP_LOWER_BODY_STD:-0.45}"
MOCAP_KNEE_WEIGHT="${MOCAP_KNEE_WEIGHT:-1.2}"
MOCAP_KNEE_STD="${MOCAP_KNEE_STD:-0.25}"
MOCAP_FOOT_WEIGHT="${MOCAP_FOOT_WEIGHT:-0.0}"
MOCAP_FOOT_STD="${MOCAP_FOOT_STD:-0.25}"
MOCAP_FOOT_SIDES="${MOCAP_FOOT_SIDES:-L,R}"
MOCAP_FOOT_Z_WEIGHT="${MOCAP_FOOT_Z_WEIGHT:-1.0}"
MOCAP_FIRST_FOOT_WEIGHT="${MOCAP_FIRST_FOOT_WEIGHT:-0.0}"
MOCAP_FIRST_FOOT_STD="${MOCAP_FIRST_FOOT_STD:-0.18}"
MOCAP_FIRST_FOOT_SIDES="${MOCAP_FIRST_FOOT_SIDES:-L,R}"
MOCAP_FIRST_FOOT_END_FRAME="${MOCAP_FIRST_FOOT_END_FRAME:-90}"
MOCAP_FOOT_EVENT_WEIGHT="${MOCAP_FOOT_EVENT_WEIGHT:-0.0}"
MOCAP_FOOT_EVENT_STD="${MOCAP_FOOT_EVENT_STD:-0.18}"
MOCAP_FOOT_EVENT_XY_WEIGHT="${MOCAP_FOOT_EVENT_XY_WEIGHT:-1.0}"
MOCAP_FOOT_EVENT_Z_WEIGHT="${MOCAP_FOOT_EVENT_Z_WEIGHT:-1.0}"
MOCAP_FOOT_EVENT_SPEED_THRESHOLD="${MOCAP_FOOT_EVENT_SPEED_THRESHOLD:-0.18}"
MOCAP_FOOT_EVENT_MIN_STANCE_FRAMES="${MOCAP_FOOT_EVENT_MIN_STANCE_FRAMES:-4}"
MOCAP_FOOT_EVENT_MIN_HEIGHT_DELTA="${MOCAP_FOOT_EVENT_MIN_HEIGHT_DELTA:-0.05}"
MOCAP_FOOT_EVENT_WINDOW_MARGIN="${MOCAP_FOOT_EVENT_WINDOW_MARGIN:-0}"
CPU="${CPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$MOCAP_MOTION" ]] && ! bool_enabled "$DRY_RUN"; then
    echo "[ERROR] Motion file not found: $MOCAP_MOTION" >&2
    exit 1
fi

CMD=(
    conda run --no-capture-output -n "$ENV_NAME"
    python scripts/train_mjlab_knee_exo_mocap_track.py
    --motion "$MOCAP_MOTION"
    --episode "$MOCAP_EPISODE"
    --training-stage "$TRAINING_STAGE"
    --terrain "$TERRAIN"
    --terrain-out-dir "$TERRAIN_OUT_DIR"
    --stair-width "$STAIR_WIDTH"
    --stair-height "$STAIR_HEIGHT"
    --stair-steps "$STAIR_STEPS"
    --num-envs "$NUM_ENVS"
    --max-iterations "$MAX_ITERATIONS"
    --num-steps-per-env "$NUM_STEPS_PER_ENV"
    --run-name "$RUN_NAME"
    --log-root "$LOG_ROOT"
    --human-residual-mode "$HUMAN_RESIDUAL_MODE"
    --human-residual-scale "$HUMAN_RESIDUAL_SCALE"
    --human-residual-action-weight "$HUMAN_RESIDUAL_ACTION_WEIGHT"
    --human-residual-action-rate-weight "$HUMAN_RESIDUAL_ACTION_RATE_WEIGHT"
    --mocap-assist-replacement-weight "$MOCAP_ASSIST_REPLACEMENT_WEIGHT"
    --mocap-assist-start "$MOCAP_ASSIST_START"
    --mocap-assist-end "$MOCAP_ASSIST_END"
    --mocap-assist-decay-fraction "$MOCAP_ASSIST_DECAY_FRACTION"
    --mocap-assist-decay-steps "$MOCAP_ASSIST_DECAY_STEPS"
    --mocap-assist-position-gain "$MOCAP_ASSIST_POSITION_GAIN"
    --mocap-assist-velocity-gain "$MOCAP_ASSIST_VELOCITY_GAIN"
    --mocap-assist-max-action "$MOCAP_ASSIST_MAX_ACTION"
    --mocap-root-xyz-weight "$MOCAP_ROOT_XYZ_WEIGHT"
    --mocap-root-xyz-std "$MOCAP_ROOT_XYZ_STD"
    --mocap-root-orientation-weight "$MOCAP_ROOT_ORIENTATION_WEIGHT"
    --mocap-root-orientation-std "$MOCAP_ROOT_ORIENTATION_STD"
    --mocap-lower-body-weight "$MOCAP_LOWER_BODY_WEIGHT"
    --mocap-lower-body-std "$MOCAP_LOWER_BODY_STD"
    --mocap-knee-weight "$MOCAP_KNEE_WEIGHT"
    --mocap-knee-std "$MOCAP_KNEE_STD"
    --mocap-foot-weight "$MOCAP_FOOT_WEIGHT"
    --mocap-foot-std "$MOCAP_FOOT_STD"
    --mocap-foot-sides "$MOCAP_FOOT_SIDES"
    --mocap-foot-z-weight "$MOCAP_FOOT_Z_WEIGHT"
    --mocap-first-foot-weight "$MOCAP_FIRST_FOOT_WEIGHT"
    --mocap-first-foot-std "$MOCAP_FIRST_FOOT_STD"
    --mocap-first-foot-sides "$MOCAP_FIRST_FOOT_SIDES"
    --mocap-first-foot-end-frame "$MOCAP_FIRST_FOOT_END_FRAME"
    --mocap-foot-event-weight "$MOCAP_FOOT_EVENT_WEIGHT"
    --mocap-foot-event-std "$MOCAP_FOOT_EVENT_STD"
    --mocap-foot-event-xy-weight "$MOCAP_FOOT_EVENT_XY_WEIGHT"
    --mocap-foot-event-z-weight "$MOCAP_FOOT_EVENT_Z_WEIGHT"
    --mocap-foot-event-speed-threshold "$MOCAP_FOOT_EVENT_SPEED_THRESHOLD"
    --mocap-foot-event-min-stance-frames "$MOCAP_FOOT_EVENT_MIN_STANCE_FRAMES"
    --mocap-foot-event-min-height-delta "$MOCAP_FOOT_EVENT_MIN_HEIGHT_DELTA"
    --mocap-foot-event-window-margin "$MOCAP_FOOT_EVENT_WINDOW_MARGIN"
)
if [[ -n "$ACTOR_INIT_STD" ]]; then
    CMD+=(--actor-init-std "$ACTOR_INIT_STD")
fi
if [[ -n "$ENTROPY_COEF" ]]; then
    CMD+=(--entropy-coef "$ENTROPY_COEF")
fi
if [[ -n "$BASE_COMPENSATION_CHECKPOINT" ]]; then
    CMD+=(--base-compensation-checkpoint "$BASE_COMPENSATION_CHECKPOINT")
fi
if bool_enabled "$CPU"; then
    CMD+=(--cpu)
fi
CMD+=("$@")

echo "[INFO] Stair mocap training wrapper"
echo "[INFO] motion=$MOCAP_MOTION:$MOCAP_EPISODE"
echo "[INFO] training_stage=$TRAINING_STAGE base_compensation_checkpoint=${BASE_COMPENSATION_CHECKPOINT:-<none>}"
echo "[INFO] terrain=$TERRAIN stair_width=$STAIR_WIDTH stair_height=$STAIR_HEIGHT stair_steps=$STAIR_STEPS"
echo "[INFO] residual mode=$HUMAN_RESIDUAL_MODE scale=$HUMAN_RESIDUAL_SCALE action_weight=$HUMAN_RESIDUAL_ACTION_WEIGHT action_rate_weight=$HUMAN_RESIDUAL_ACTION_RATE_WEIGHT"
echo "[INFO] actor_init_std=${ACTOR_INIT_STD:-<default>} entropy_coef=${ENTROPY_COEF:-<default>}"
echo "[INFO] assist_replacement_weight=$MOCAP_ASSIST_REPLACEMENT_WEIGHT"
echo "[INFO] mocap_assist start=$MOCAP_ASSIST_START end=$MOCAP_ASSIST_END decay_fraction=$MOCAP_ASSIST_DECAY_FRACTION decay_steps=$MOCAP_ASSIST_DECAY_STEPS pos_gain=$MOCAP_ASSIST_POSITION_GAIN vel_gain=$MOCAP_ASSIST_VELOCITY_GAIN max_action=$MOCAP_ASSIST_MAX_ACTION"
echo "[INFO] rewards root_xyz=$MOCAP_ROOT_XYZ_WEIGHT root_orientation=$MOCAP_ROOT_ORIENTATION_WEIGHT lower_body=$MOCAP_LOWER_BODY_WEIGHT knee=$MOCAP_KNEE_WEIGHT foot=$MOCAP_FOOT_WEIGHT($MOCAP_FOOT_SIDES) first_foot=$MOCAP_FIRST_FOOT_WEIGHT($MOCAP_FIRST_FOOT_SIDES)"
echo "[INFO] foot_event weight=$MOCAP_FOOT_EVENT_WEIGHT std=$MOCAP_FOOT_EVENT_STD xy_weight=$MOCAP_FOOT_EVENT_XY_WEIGHT z_weight=$MOCAP_FOOT_EVENT_Z_WEIGHT"
echo "[INFO] num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS num_steps_per_env=$NUM_STEPS_PER_ENV run_name=$RUN_NAME"

if bool_enabled "$DRY_RUN"; then
    printf '[DRY-RUN] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
