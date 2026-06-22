#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${BASE_COMPENSATION_CHECKPOINT:-}" ]]; then
    cat >&2 <<'EOF'
[ERROR] BASE_COMPENSATION_CHECKPOINT is required.
Example:
  BASE_COMPENSATION_CHECKPOINT=logs/rsl_rl/humenv_s1_stair_compensation/<run>/model_999.pt \
    ./train_knee_exo_updown_stairs_on_compensation.sh
EOF
    exit 1
fi

export TRAINING_STAGE="${TRAINING_STAGE:-knee-exo-on-compensation}"
export RUN_NAME="${RUN_NAME:-knee_exo_on_s1_stair_compensation}"
export HUMAN_RESIDUAL_MODE="${HUMAN_RESIDUAL_MODE:-all}"
export HUMAN_RESIDUAL_SCALE="${HUMAN_RESIDUAL_SCALE:-0.35}"
export HUMAN_RESIDUAL_ACTION_WEIGHT=0.0
export HUMAN_RESIDUAL_ACTION_RATE_WEIGHT=0.0
export MOCAP_ASSIST_START=0.0
export MOCAP_ASSIST_END=0.0
export MOCAP_ASSIST_DECAY_FRACTION=0.0
export MOCAP_ROOT_XYZ_WEIGHT="${MOCAP_ROOT_XYZ_WEIGHT:-0.6}"
export MOCAP_ROOT_ORIENTATION_WEIGHT="${MOCAP_ROOT_ORIENTATION_WEIGHT:-0.3}"
export MOCAP_LOWER_BODY_WEIGHT="${MOCAP_LOWER_BODY_WEIGHT:-0.8}"
export MOCAP_KNEE_WEIGHT="${MOCAP_KNEE_WEIGHT:-1.0}"
export MOCAP_FOOT_WEIGHT="${MOCAP_FOOT_WEIGHT:-0.5}"
export MOCAP_FOOT_Z_WEIGHT="${MOCAP_FOOT_Z_WEIGHT:-1.4}"
export MOCAP_FIRST_FOOT_WEIGHT="${MOCAP_FIRST_FOOT_WEIGHT:-0.5}"
export MOCAP_FIRST_FOOT_END_FRAME="${MOCAP_FIRST_FOOT_END_FRAME:-90}"

exec ./train_mocap_track_updown_stairs.sh "$@"
