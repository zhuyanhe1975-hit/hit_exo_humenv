#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: ./test_mocap_track_updown_stairs.sh [extra run_s1_mocap_track_visual.py args]

Tracking-first stair skill smoke test for frozen MetaMotivo S-1.

Environment overrides:
  DATASET=kit167|upx|terrain|mimic|raw|custom
  DIRECTION=up|down|both
  MOCAP_MOTION=/path/to/motion.hdf5       Required for DATASET=custom
  MOCAP_EPISODE=ep_0
  TERRAIN=stairs|supports|mimic-stairs|flat
  STAIR_WIDTH=1.4
  STAIR_HEIGHT=0.135
  STAIR_STEPS=0                         0 lets the Python runner infer from motion rise
  KINEMATIC_PREFLIGHT=1                 Run short kinematic terrain/mocap alignment check
  PREFLIGHT_ONLY=0                      Exit after the kinematic preflight
  PREFLIGHT_DURATION=0.35
  PREFLIGHT_OUT_DIR=.omx/.../kinematic_preflight
  ASSIST_PRESET=none|rootxy|climb       climb uses the best current assisted upstairs settings
  ROOT_XY_TRACK=0                       Set to 1 only for visual/root-aligned diagnostics
  ROOT_XY_TRACK_GAIN=1.0
  ROOT_XY_VELOCITY_GAIN=1.0
  ROOT_Z_TRACK=0                        Set to 1 for assisted stair-climb diagnostics
  ROOT_Z_TRACK_GAIN=1.0
  ROOT_Z_VELOCITY_GAIN=1.0
  ROOT_ORIENTATION_TRACK=0
  ROOT_ORIENTATION_TRACK_GAIN=0.5
  ROOT_ANGULAR_VELOCITY_GAIN=0.5
  JOINT_POSE_TRACK_GAIN=0.0             Weak mocap joint assist; 0 keeps frozen S-1 unassisted
  JOINT_VELOCITY_TRACK_GAIN=0.0
  LOG_DIR=$OUT_DIR/logs
  RUN_ID=YYYYmmdd_HHMMSS
  DRY_RUN=1                             Print resolved commands without executing

Examples:
  ./test_mocap_track_updown_stairs.sh --headless --duration 2
  DATASET=mimic DIRECTION=up ./test_mocap_track_updown_stairs.sh --headless
  DATASET=custom MOCAP_MOTION=/tmp/stairs.hdf5 TERRAIN=stairs ./test_mocap_track_updown_stairs.sh
EOF
}

bool_enabled() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

run_in_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
        "$@"
    else
        conda run --no-capture-output -n "$ENV_NAME" "$@"
    fi
}

write_run_status() {
    if [[ -n "${RUN_STATUS:-}" ]]; then
        printf '%s=%s\n' "$1" "$2" >> "$RUN_STATUS"
    fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

HUMENV_DATA_ROOT="${HUMENV_DATA_ROOT:-/home/yhzhu/AI/humenv/data_preparation}"
DATASET="${DATASET:-kit167}"
DIRECTION="${DIRECTION:-up}"
DATASET="${DATASET,,}"
DIRECTION="${DIRECTION,,}"
ENV_NAME="${ENV_NAME:-mjwarp_env}"
MIMIC_UP_SOURCE="$HUMENV_DATA_ROOT/humenv_amass_terrain_fixed/stairs_up.hdf5"
MIMIC_REFERENCE_YAW_DEG="${MIMIC_REFERENCE_YAW_DEG:-36.254852}"
MIMIC_UP_HEADING_FIXED="${MIMIC_UP_HEADING_FIXED:-.omx/mimic_stairs_up_reference_yaw_36p254852.hdf5}"
UNITREE_TERRAIN_GENERATOR_PATH="${UNITREE_TERRAIN_GENERATOR_PATH:-/home/yhzhu/myWorks_vips/unitree_mujoco/terrain_tool/terrain_generator.py}"
DRY_RUN="${DRY_RUN:-0}"

case "$DATASET:$DIRECTION" in
    kit167:up)
        DEFAULT_MOCAP_MOTION="$HUMENV_DATA_ROOT/humenv_from_protomotions/KIT_167_upstairs03_poses.hdf5"
        ;;
    upx:up)
        DEFAULT_MOCAP_MOTION="$HUMENV_DATA_ROOT/humenv_from_protomotions/KIT_167_upstairs_downstairs01_poses_upx_upstairs.hdf5"
        ;;
    upx:both)
        DEFAULT_MOCAP_MOTION="$HUMENV_DATA_ROOT/humenv_from_protomotions/KIT_167_upstairs_downstairs01_poses.hdf5"
        ;;
    upx:down)
        echo "[ERROR] DATASET=upx only has DIRECTION=up or both." >&2
        echo "        Try DATASET=terrain DIRECTION=down." >&2
        exit 2
        ;;
    terrain:up)
        DEFAULT_MOCAP_MOTION="$MIMIC_UP_SOURCE"
        ;;
    terrain:down)
        DEFAULT_MOCAP_MOTION="$HUMENV_DATA_ROOT/humenv_amass_terrain_fixed/stairs_down.hdf5"
        ;;
    terrain:both)
        echo "[ERROR] DATASET=terrain requires DIRECTION=up or down." >&2
        exit 2
        ;;
    mimic:up)
        DEFAULT_MOCAP_MOTION="$MIMIC_UP_HEADING_FIXED"
        ;;
    mimic:down|mimic:both)
        echo "[ERROR] DATASET=mimic currently provides the hit_exo_mimic AMASS-matched upstairs motion only." >&2
        exit 2
        ;;
    raw:both)
        DEFAULT_MOCAP_MOTION="$HUMENV_DATA_ROOT/humenv_from_protomotions/0016_upstairs_downstairs01_poses.hdf5"
        ;;
    raw:up)
        DEFAULT_MOCAP_MOTION="$HUMENV_DATA_ROOT/humenv_from_protomotions/0017_upstairs01_poses.hdf5"
        ;;
    raw:down)
        echo "[ERROR] DATASET=raw has no default down-only file in this wrapper." >&2
        echo "        Try DATASET=terrain DIRECTION=down or set MOCAP_MOTION=/path/to/file.hdf5." >&2
        exit 2
        ;;
    custom:up|custom:down|custom:both)
        if [[ -z "${MOCAP_MOTION:-}" ]]; then
            echo "[ERROR] DATASET=custom requires MOCAP_MOTION=/path/to/file.hdf5." >&2
            exit 2
        fi
        DEFAULT_MOCAP_MOTION="$MOCAP_MOTION"
        ;;
    *)
        echo "[ERROR] Unsupported DATASET=$DATASET DIRECTION=$DIRECTION." >&2
        echo "        DATASET: kit167 | upx | terrain | mimic | raw | custom" >&2
        echo "        DIRECTION: up | down | both" >&2
        exit 2
        ;;
esac

MOCAP_MOTION="${MOCAP_MOTION:-$DEFAULT_MOCAP_MOTION}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
MODEL_ID="${MODEL_ID:-facebook/metamotivo-S-1}"
DEVICE="${DEVICE:-cpu}"
DURATION="${DURATION:-0}"
START_FRAME="${START_FRAME:-0}"
REPEAT="${REPEAT:-10}"
OUT_DIR="${OUT_DIR:-.omx/s1_mocap_track_updown_stairs_visual}"
if [[ -z "${TERRAIN:-}" ]]; then
    if [[ "$DATASET" == "upx" ]]; then
        TERRAIN="supports"
    elif [[ "$DATASET" == "mimic" ]]; then
        TERRAIN="mimic-stairs"
    else
        TERRAIN="stairs"
    fi
fi
STAIR_WIDTH="${STAIR_WIDTH:-1.4}"
STAIR_HEIGHT="${STAIR_HEIGHT:-0.135}"
STAIR_STEPS="${STAIR_STEPS:-0}"
KINEMATIC_PREFLIGHT="${KINEMATIC_PREFLIGHT:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PREFLIGHT_DURATION="${PREFLIGHT_DURATION:-0.35}"
PREFLIGHT_OUT_DIR="${PREFLIGHT_OUT_DIR:-$OUT_DIR/kinematic_preflight}"
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
ASSIST_PRESET="${ASSIST_PRESET:-none}"
case "$ASSIST_PRESET" in
    none)
        ;;
    rootxy)
        ROOT_XY_TRACK=1
        ;;
    climb)
        ROOT_XY_TRACK=1
        ROOT_Z_TRACK=1
        ROOT_ORIENTATION_TRACK=1
        JOINT_POSE_TRACK_GAIN=0.60
        JOINT_VELOCITY_TRACK_GAIN=0.60
        ;;
    *)
        echo "[ERROR] Unsupported ASSIST_PRESET=$ASSIST_PRESET. Use none, rootxy, or climb." >&2
        exit 2
        ;;
esac
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
RUN_LOG="${RUN_LOG:-$LOG_DIR/${RUN_ID}.log}"
RUN_STATUS="${RUN_STATUS:-$LOG_DIR/${RUN_ID}.status}"

mkdir -p "$LOG_DIR"
: > "$RUN_STATUS"
exec > >(tee -a "$RUN_LOG") 2>&1

write_run_status "run_id" "$RUN_ID"
write_run_status "started_at" "$(date +%Y-%m-%dT%H:%M:%S%z)"
write_run_status "state" "starting"
write_run_status "log" "$RUN_LOG"
write_run_status "dataset" "$DATASET"
write_run_status "direction" "$DIRECTION"
write_run_status "motion" "$MOCAP_MOTION"
write_run_status "episode" "$MOCAP_EPISODE"
write_run_status "terrain" "$TERRAIN"
write_run_status "assist_preset" "$ASSIST_PRESET"
write_run_status "root_xy_track" "$ROOT_XY_TRACK"
write_run_status "root_z_track" "$ROOT_Z_TRACK"
write_run_status "root_orientation_track" "$ROOT_ORIENTATION_TRACK"
write_run_status "joint_pose_track_gain" "$JOINT_POSE_TRACK_GAIN"
write_run_status "joint_velocity_track_gain" "$JOINT_VELOCITY_TRACK_GAIN"

run_phase() {
    local phase="$1"
    shift
    echo "[INFO] Starting phase: $phase"
    write_run_status "${phase}_started_at" "$(date +%Y-%m-%dT%H:%M:%S%z)"
    write_run_status "${phase}_command" "$*"
    if "$@"; then
        write_run_status "${phase}_status" "ok"
        write_run_status "${phase}_completed_at" "$(date +%Y-%m-%dT%H:%M:%S%z)"
    else
        local status=$?
        write_run_status "${phase}_status" "failed"
        write_run_status "${phase}_exit_code" "$status"
        write_run_status "state" "failed"
        write_run_status "failed_phase" "$phase"
        write_run_status "failed_at" "$(date +%Y-%m-%dT%H:%M:%S%z)"
        return "$status"
    fi
}

finalize_status() {
    local status=$?
    trap - EXIT
    if [[ "$status" -eq 0 ]]; then
        write_run_status "state" "completed"
        write_run_status "completed_at" "$(date +%Y-%m-%dT%H:%M:%S%z)"
    else
        write_run_status "state" "failed"
        write_run_status "exit_code" "$status"
        write_run_status "failed_at" "$(date +%Y-%m-%dT%H:%M:%S%z)"
    fi
    echo "[INFO] Wrapper log: $RUN_LOG"
    echo "[INFO] Wrapper status: $RUN_STATUS"
}
trap finalize_status EXIT

echo "[INFO] Wrapper log: $RUN_LOG"
echo "[INFO] Wrapper status: $RUN_STATUS"

if [[ "$DATASET:$DIRECTION" == "mimic:up" && "$MOCAP_MOTION" == "$DEFAULT_MOCAP_MOTION" && ! -f "$MOCAP_MOTION" ]]; then
    echo "[INFO] Generating heading-corrected mimic stairs mocap: $MOCAP_MOTION"
    MIMIC_CMD=(
        python scripts/rotate_humenv_root_yaw.py
        --input "$MIMIC_UP_SOURCE"
        --output "$MOCAP_MOTION"
        --yaw-deg "$MIMIC_REFERENCE_YAW_DEG"
    )
    if bool_enabled "$DRY_RUN"; then
        printf '[DRY-RUN] '
        printf '%q ' "${MIMIC_CMD[@]}"
        printf '\n'
    else
        mkdir -p "$(dirname "$MOCAP_MOTION")"
        run_phase "prepare_mimic_motion" run_in_env "${MIMIC_CMD[@]}"
    fi
fi

if [[ ! -f "$MOCAP_MOTION" ]] && ! bool_enabled "$DRY_RUN"; then
    echo "[ERROR] Motion file not found: $MOCAP_MOTION" >&2
    exit 1
elif [[ ! -f "$MOCAP_MOTION" ]]; then
    echo "[WARN] Motion file not found yet, but continuing because DRY_RUN=1: $MOCAP_MOTION" >&2
fi

if [[ "$TERRAIN" != "flat" && ! -f "$UNITREE_TERRAIN_GENERATOR_PATH" ]] && ! bool_enabled "$DRY_RUN"; then
    echo "[ERROR] Unitree terrain generator not found: $UNITREE_TERRAIN_GENERATOR_PATH" >&2
    echo "        Non-flat terrain in scripts/run_s1_mocap_track_visual.py currently depends on this path." >&2
    exit 1
elif [[ "$TERRAIN" != "flat" && ! -f "$UNITREE_TERRAIN_GENERATOR_PATH" ]]; then
    echo "[WARN] Unitree terrain generator not found, but continuing because DRY_RUN=1: $UNITREE_TERRAIN_GENERATOR_PATH" >&2
fi

if [[ "$TERRAIN" == "supports" ]]; then
    if [[ "$MOCAP_MOTION" == *_upx_upstairs.hdf5 ]]; then
        SUPPORT_SIDECAR="${MOCAP_MOTION%.hdf5}_pillars.supports.json"
    else
        SUPPORT_SIDECAR="${MOCAP_MOTION%.hdf5}.supports.json"
    fi
    if [[ ! -f "$SUPPORT_SIDECAR" ]] && ! bool_enabled "$DRY_RUN"; then
        echo "[ERROR] TERRAIN=supports requires support sidecar: $SUPPORT_SIDECAR" >&2
        echo "        Use TERRAIN=stairs for generated stair boxes or provide the sidecar." >&2
        exit 1
    elif [[ ! -f "$SUPPORT_SIDECAR" ]]; then
        echo "[WARN] Support sidecar not found, but continuing because DRY_RUN=1: $SUPPORT_SIDECAR" >&2
    fi
fi

echo "[INFO] Stair mocap dynamics tracking wrapper"
echo "[INFO] dataset=$DATASET direction=$DIRECTION"
echo "[INFO] Tracking-first: infer z[t] from mocap observation, then run S-1 dynamics on stair terrain."
echo "[INFO] Mocap reference: $MOCAP_MOTION:$MOCAP_EPISODE"
echo "[INFO] Terrain: $TERRAIN stair_width=$STAIR_WIDTH stair_height=$STAIR_HEIGHT stair_steps=$STAIR_STEPS"
echo "[INFO] Assist preset: $ASSIST_PRESET"
echo "[INFO] Root XY tracking: $ROOT_XY_TRACK position_gain=$ROOT_XY_TRACK_GAIN velocity_gain=$ROOT_XY_VELOCITY_GAIN"
echo "[INFO] Root Z tracking: $ROOT_Z_TRACK position_gain=$ROOT_Z_TRACK_GAIN velocity_gain=$ROOT_Z_VELOCITY_GAIN"
echo "[INFO] Root orientation tracking: $ROOT_ORIENTATION_TRACK quat_gain=$ROOT_ORIENTATION_TRACK_GAIN angular_velocity_gain=$ROOT_ANGULAR_VELOCITY_GAIN"
echo "[INFO] Joint tracking assist: pose_gain=$JOINT_POSE_TRACK_GAIN velocity_gain=$JOINT_VELOCITY_TRACK_GAIN"
echo "[INFO] Outputs: $OUT_DIR/tracking_latents.pt and $OUT_DIR/rollout.csv"
echo "[INFO] Viewer overlay: mocap reference humanoid is green; dynamic S-1 keeps the XML color."

if bool_enabled "$KINEMATIC_PREFLIGHT"; then
    echo "[INFO] Running kinematic preflight to verify mocap/terrain alignment before dynamics."
    PREFLIGHT_CMD=(
        env
        HUMENV_DATA_ROOT="$HUMENV_DATA_ROOT"
        DATASET="$DATASET"
        DIRECTION="$DIRECTION"
        ENV_NAME="$ENV_NAME"
        MOCAP_MOTION="$MOCAP_MOTION"
        MOCAP_EPISODE="$MOCAP_EPISODE"
        DURATION="$PREFLIGHT_DURATION"
        START_FRAME="$START_FRAME"
        REPEAT=1
        OUT_DIR="$PREFLIGHT_OUT_DIR"
        TERRAIN="$TERRAIN"
        STAIR_WIDTH="$STAIR_WIDTH"
        STAIR_HEIGHT="$STAIR_HEIGHT"
        STAIR_STEPS="$STAIR_STEPS"
        ./show_mocap_track_updown_stairs.sh
        --headless
    )
    if bool_enabled "$DRY_RUN"; then
        printf '[DRY-RUN] '
        printf '%q ' "${PREFLIGHT_CMD[@]}"
        printf '\n'
    else
        run_phase "kinematic_preflight" "${PREFLIGHT_CMD[@]}"
    fi
else
    echo "[INFO] Kinematic preflight disabled."
    write_run_status "kinematic_preflight_status" "skipped"
fi

if bool_enabled "$PREFLIGHT_ONLY"; then
    if ! bool_enabled "$KINEMATIC_PREFLIGHT"; then
        echo "[ERROR] PREFLIGHT_ONLY=1 requires KINEMATIC_PREFLIGHT=1." >&2
        exit 2
    fi
    write_run_status "s1_dynamics_rollout_status" "skipped"
    write_run_status "stop_reason" "preflight_only"
    echo "[INFO] PREFLIGHT_ONLY=1, exiting before S-1 dynamics rollout."
    exit 0
fi

FINAL_ENV=(
    env
    MOCAP_MOTION="$MOCAP_MOTION"
    MOCAP_EPISODE="$MOCAP_EPISODE"
    MODEL_ID="$MODEL_ID"
    DEVICE="$DEVICE"
    DURATION="$DURATION"
    START_FRAME="$START_FRAME"
    REPEAT="$REPEAT"
    OUT_DIR="$OUT_DIR"
    TERRAIN="$TERRAIN"
    STAIR_WIDTH="$STAIR_WIDTH"
    STAIR_HEIGHT="$STAIR_HEIGHT"
    STAIR_STEPS="$STAIR_STEPS"
    ROOT_XY_TRACK="$ROOT_XY_TRACK"
    ROOT_XY_TRACK_GAIN="$ROOT_XY_TRACK_GAIN"
    ROOT_XY_VELOCITY_GAIN="$ROOT_XY_VELOCITY_GAIN"
    ROOT_Z_TRACK="$ROOT_Z_TRACK"
    ROOT_Z_TRACK_GAIN="$ROOT_Z_TRACK_GAIN"
    ROOT_Z_VELOCITY_GAIN="$ROOT_Z_VELOCITY_GAIN"
    ROOT_ORIENTATION_TRACK="$ROOT_ORIENTATION_TRACK"
    ROOT_ORIENTATION_TRACK_GAIN="$ROOT_ORIENTATION_TRACK_GAIN"
    ROOT_ANGULAR_VELOCITY_GAIN="$ROOT_ANGULAR_VELOCITY_GAIN"
    JOINT_POSE_TRACK_GAIN="$JOINT_POSE_TRACK_GAIN"
    JOINT_VELOCITY_TRACK_GAIN="$JOINT_VELOCITY_TRACK_GAIN"
    ./test_mocap_track.sh
)

if bool_enabled "$DRY_RUN"; then
    write_run_status "dry_run" "true"
    if bool_enabled "$KINEMATIC_PREFLIGHT"; then
        write_run_status "kinematic_preflight_status" "dry_run"
    fi
    write_run_status "s1_dynamics_rollout_status" "dry_run"
    printf '[DRY-RUN] '
    printf '%q ' "${FINAL_ENV[@]}" "$@"
    printf '\n'
    exit 0
fi

run_phase "s1_dynamics_rollout" "${FINAL_ENV[@]}" "$@"
