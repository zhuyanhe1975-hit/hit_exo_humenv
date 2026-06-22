#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mjwarp_env}"
DEFAULT_MOCAP_DIR="/home/yhzhu/AI/humenv/data_preparation/humenv_from_phc_amass_transitions_kit_cmu"
DEFAULT_MOCAP_MOTION="$DEFAULT_MOCAP_DIR/08_KIT_167_walking_medium_resampled_poses.hdf5"
MOCAP_MOTION="${MOCAP_MOTION:-$DEFAULT_MOCAP_MOTION}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
MODEL_ID="${MODEL_ID:-facebook/metamotivo-S-1}"
DEVICE="${DEVICE:-cpu}"
DURATION="${DURATION:-0}"
START_FRAME="${START_FRAME:-0}"
REPEAT="${REPEAT:-10}"
OUT_DIR="${OUT_DIR:-.omx/s1_mocap_track_visual}"
TERRAIN="${TERRAIN:-flat}"
STAIR_WIDTH="${STAIR_WIDTH:-1.4}"
STAIR_HEIGHT="${STAIR_HEIGHT:-0.135}"
STAIR_STEPS="${STAIR_STEPS:-0}"
ROOT_XY_TRACK="${ROOT_XY_TRACK:-0}"
ROOT_XY_TRACK_GAIN="${ROOT_XY_TRACK_GAIN:-1.0}"
ROOT_XY_VELOCITY_GAIN="${ROOT_XY_VELOCITY_GAIN:-1.0}"
ROOT_Z_TRACK="${ROOT_Z_TRACK:-0}"
ROOT_Z_TRACK_GAIN="${ROOT_Z_TRACK_GAIN:-1.0}"
ROOT_Z_VELOCITY_GAIN="${ROOT_Z_VELOCITY_GAIN:-1.0}"
ROOT_ORIENTATION_TRACK="${ROOT_ORIENTATION_TRACK:-0}"
ROOT_ORIENTATION_TRACK_GAIN="${ROOT_ORIENTATION_TRACK_GAIN:-0.5}"
ROOT_ANGULAR_VELOCITY_GAIN="${ROOT_ANGULAR_VELOCITY_GAIN:-0.5}"
JOINT_POSE_TRACK_GAIN="${JOINT_POSE_TRACK_GAIN:-0.0}"
JOINT_VELOCITY_TRACK_GAIN="${JOINT_VELOCITY_TRACK_GAIN:-0.0}"
MUJOCO_GL="${MUJOCO_GL:-glfw}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MUJOCO_GL OMP_NUM_THREADS

cd "$(dirname "$0")"

CMD=(
    python scripts/run_s1_mocap_track_visual.py
    --motion "$MOCAP_MOTION"
    --episode "$MOCAP_EPISODE"
    --model-id "$MODEL_ID"
    --device "$DEVICE"
    --duration "$DURATION"
    --start-frame "$START_FRAME"
    --repeat "$REPEAT"
    --out-dir "$OUT_DIR"
    --terrain "$TERRAIN"
    --stair-width "$STAIR_WIDTH"
    --stair-height "$STAIR_HEIGHT"
    --stair-steps "$STAIR_STEPS"
    --root-xy-track-gain "$ROOT_XY_TRACK_GAIN"
    --root-xy-velocity-gain "$ROOT_XY_VELOCITY_GAIN"
    --root-z-track-gain "$ROOT_Z_TRACK_GAIN"
    --root-z-velocity-gain "$ROOT_Z_VELOCITY_GAIN"
    --root-orientation-track-gain "$ROOT_ORIENTATION_TRACK_GAIN"
    --root-angular-velocity-gain "$ROOT_ANGULAR_VELOCITY_GAIN"
    --joint-pose-track-gain "$JOINT_POSE_TRACK_GAIN"
    --joint-velocity-track-gain "$JOINT_VELOCITY_TRACK_GAIN"
)
if [[ "$ROOT_XY_TRACK" == "1" || "$ROOT_XY_TRACK" == "true" || "$ROOT_XY_TRACK" == "yes" ]]; then
    CMD+=(--track-root-xy)
fi
if [[ "$ROOT_Z_TRACK" == "1" || "$ROOT_Z_TRACK" == "true" || "$ROOT_Z_TRACK" == "yes" ]]; then
    CMD+=(--track-root-z)
fi
if [[ "$ROOT_ORIENTATION_TRACK" == "1" || "$ROOT_ORIENTATION_TRACK" == "true" || "$ROOT_ORIENTATION_TRACK" == "yes" ]]; then
    CMD+=(--track-root-orientation)
fi
CMD+=("$@")

echo "[INFO] Testing native HumEnv frozen S-1 mocap tracking only; no training and no knee-exo."
echo "[INFO] Mocap reference: $MOCAP_MOTION:$MOCAP_EPISODE"
echo "[INFO] Model: $MODEL_ID device=$DEVICE duration=$DURATION start_frame=$START_FRAME repeat=$REPEAT"
echo "[INFO] Terrain: $TERRAIN stair_width=$STAIR_WIDTH stair_height=$STAIR_HEIGHT stair_steps=$STAIR_STEPS"
echo "[INFO] Root XY tracking: $ROOT_XY_TRACK position_gain=$ROOT_XY_TRACK_GAIN velocity_gain=$ROOT_XY_VELOCITY_GAIN"
echo "[INFO] Root Z tracking: $ROOT_Z_TRACK position_gain=$ROOT_Z_TRACK_GAIN velocity_gain=$ROOT_Z_VELOCITY_GAIN"
echo "[INFO] Root orientation tracking: $ROOT_ORIENTATION_TRACK quat_gain=$ROOT_ORIENTATION_TRACK_GAIN angular_velocity_gain=$ROOT_ANGULAR_VELOCITY_GAIN"
echo "[INFO] Joint tracking assist: pose_gain=$JOINT_POSE_TRACK_GAIN velocity_gain=$JOINT_VELOCITY_TRACK_GAIN"

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
