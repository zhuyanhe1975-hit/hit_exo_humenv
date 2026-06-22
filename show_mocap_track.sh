#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mjwarp_env}"
DEFAULT_MOCAP_DIR="/home/yhzhu/AI/humenv/data_preparation/humenv_from_phc_amass_transitions_kit_cmu"
DEFAULT_MOCAP_MOTION="$DEFAULT_MOCAP_DIR/08_KIT_167_walking_medium_resampled_poses.hdf5"
MOCAP_MOTION="${MOCAP_MOTION:-$DEFAULT_MOCAP_MOTION}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
SPEED="${SPEED:-1.0}"
DURATION="${DURATION:-0}"
START_FRAME="${START_FRAME:-0}"
REPEAT="${REPEAT:-1}"
OUT_DIR="${OUT_DIR:-.omx/mocap_kinematic_visual}"
TERRAIN="${TERRAIN:-flat}"
STAIR_WIDTH="${STAIR_WIDTH:-1.4}"
STAIR_HEIGHT="${STAIR_HEIGHT:-0.135}"
STAIR_STEPS="${STAIR_STEPS:-0}"
MUJOCO_GL="${MUJOCO_GL:-glfw}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MUJOCO_GL OMP_NUM_THREADS

cd "$(dirname "$0")"

has_arg() {
    local needle="$1"
    shift
    for arg in "$@"; do
        if [[ "$arg" == "$needle" || "$arg" == "$needle="* ]]; then
            return 0
        fi
    done
    return 1
}

arg_value() {
    local needle="$1"
    shift
    local previous=""
    for arg in "$@"; do
        if [[ "$previous" == "$needle" ]]; then
            printf '%s\n' "$arg"
            return 0
        fi
        if [[ "$arg" == "$needle="* ]]; then
            printf '%s\n' "${arg#*=}"
            return 0
        fi
        previous="$arg"
    done
    return 1
}

CMD=(
    python scripts/show_mocap_kinematic.py
    --episode "$MOCAP_EPISODE"
    --speed "$SPEED"
    --duration "$DURATION"
    --start-frame "$START_FRAME"
    --repeat "$REPEAT"
    --out-dir "$OUT_DIR"
    --terrain "$TERRAIN"
    --stair-width "$STAIR_WIDTH"
    --stair-height "$STAIR_HEIGHT"
    --stair-steps "$STAIR_STEPS"
)
if ! has_arg "--motion" "$@"; then
    CMD+=(--motion "$MOCAP_MOTION")
fi
CMD+=("$@")

DISPLAY_MOTION="$MOCAP_MOTION"
if cli_motion="$(arg_value "--motion" "$@" 2>/dev/null)"; then
    DISPLAY_MOTION="$cli_motion"
fi

echo "[INFO] Showing pure kinematic mocap replay; no S-1, no RL, no dynamics."
echo "[INFO] Mocap reference: $DISPLAY_MOTION:$MOCAP_EPISODE"
echo "[INFO] speed=$SPEED duration=$DURATION start_frame=$START_FRAME repeat=$REPEAT"
echo "[INFO] Terrain: $TERRAIN stair_width=$STAIR_WIDTH stair_height=$STAIR_HEIGHT stair_steps=$STAIR_STEPS"

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
