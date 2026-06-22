#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$LATENT_Z_CHECKPOINT_ROOT}"
NUM_ENVS="${NUM_ENVS:-$LATENT_Z_VIEWER_NUM_ENVS}"
ENV_SPACING="${ENV_SPACING:-$LATENT_Z_ENV_SPACING}"
S1_LATENT_SPEED_SCALE="${S1_LATENT_SPEED_SCALE:-$LATENT_Z_S1_LATENT_SPEED_SCALE}"
HUMAN_ACTION_REPEAT="${HUMAN_ACTION_REPEAT:-$LATENT_Z_HUMAN_ACTION_REPEAT}"
HUMAN_ACTION_SMOOTHING="${HUMAN_ACTION_SMOOTHING:-$LATENT_Z_HUMAN_ACTION_SMOOTHING}"
HUMAN_ROOT_HEIGHT="${HUMAN_ROOT_HEIGHT:-$LATENT_Z_HUMAN_ROOT_HEIGHT}"
RANDOM_WALK_SPEED="${RANDOM_WALK_SPEED:-$LATENT_Z_RANDOM_WALK_SPEED}"
WALK_SPEED="${WALK_SPEED:-$LATENT_Z_NORMAL_WALK_SPEED}"
RANDOM_WALK_DIRECTION="${RANDOM_WALK_DIRECTION:-$LATENT_Z_RANDOM_WALK_DIRECTION}"
WALK_DIRECTION="${WALK_DIRECTION:-$LATENT_Z_WALK_DIRECTION}"

LATEST_CHECKPOINT="$(find "$CHECKPOINT_ROOT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "No checkpoint found under $CHECKPOINT_ROOT" >&2
    exit 1
fi

CMD=(
    python scripts/run_mjlab_knee_exo_viewer.py
    --checkpoint-file "$LATEST_CHECKPOINT"
    --num-envs "$NUM_ENVS"
    --env-spacing "$ENV_SPACING"
    --s1-latent-speed-scale "$S1_LATENT_SPEED_SCALE"
    --human-action-repeat "$HUMAN_ACTION_REPEAT"
    --human-action-smoothing "$HUMAN_ACTION_SMOOTHING"
    --human-root-height "$HUMAN_ROOT_HEIGHT"
)
if [[ "$RANDOM_WALK_SPEED" == "1" || "$RANDOM_WALK_SPEED" == "true" || "$RANDOM_WALK_SPEED" == "yes" ]]; then
    CMD+=(--random-walk-speed)
else
    CMD+=(--no-random-walk-speed --walk-speed "$WALK_SPEED")
fi
if [[ "$RANDOM_WALK_DIRECTION" == "1" || "$RANDOM_WALK_DIRECTION" == "true" || "$RANDOM_WALK_DIRECTION" == "yes" ]]; then
    CMD+=(--random-walk-direction)
else
    CMD+=(--no-random-walk-direction --walk-direction "$WALK_DIRECTION")
fi
CMD+=("$@")

child_pid=""
cleanup() {
    local status=$?
    trap - INT TERM EXIT
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
        sleep 0.5
        kill -KILL -- "-$child_pid" 2>/dev/null || kill -KILL "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    exit "$status"
}

trap cleanup INT TERM EXIT
if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    setsid "${CMD[@]}" &
else
    setsid conda run --no-capture-output -n "$ENV_NAME" "${CMD[@]}" &
fi
child_pid=$!
wait "$child_pid"
status=$?
trap - INT TERM EXIT
exit "$status"
