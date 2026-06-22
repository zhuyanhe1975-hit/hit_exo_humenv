#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/rsl_rl/humenv_knee_exo_mocap_track}"
NUM_ENVS="${NUM_ENVS:-$LATENT_Z_VIEWER_NUM_ENVS}"
ENV_SPACING="${ENV_SPACING:-$LATENT_Z_ENV_SPACING}"
DEFAULT_MOCAP_DIR="/home/yhzhu/AI/humenv/data_preparation/humenv_from_phc_amass_transitions_kit_cmu"
DEFAULT_MOCAP_MOTION="$DEFAULT_MOCAP_DIR/08_KIT_167_walking_medium_resampled_poses.hdf5"
MOCAP_MOTION="${MOCAP_MOTION:-$DEFAULT_MOCAP_MOTION}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
HUMAN_ACTION_SMOOTHING="${HUMAN_ACTION_SMOOTHING:-0.15}"

LATEST_CHECKPOINT="$(
    find "$CHECKPOINT_ROOT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
)"

if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "No checkpoint found under $CHECKPOINT_ROOT" >&2
    exit 1
fi

CMD=(
    python scripts/run_mjlab_knee_exo_mocap_track_viewer.py
    --checkpoint-file "$LATEST_CHECKPOINT"
    --motion "$MOCAP_MOTION"
    --episode "$MOCAP_EPISODE"
    --num-envs "$NUM_ENVS"
    --env-spacing "$ENV_SPACING"
    --s1-action-smoothing "$HUMAN_ACTION_SMOOTHING"
)
CMD+=("$@")

echo "[INFO] Using checkpoint: $LATEST_CHECKPOINT"
echo "[INFO] Using fixed mocap reference: $MOCAP_MOTION:$MOCAP_EPISODE"

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
