#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HUMENV_DATA_ROOT="${HUMENV_DATA_ROOT:-/home/yhzhu/AI/humenv/data_preparation}"
DATASET="${DATASET:-kit167}"
DIRECTION="${DIRECTION:-up}"
ENV_NAME="${ENV_NAME:-mjwarp_env}"
MIMIC_UP_SOURCE="$HUMENV_DATA_ROOT/humenv_amass_terrain_fixed/stairs_up.hdf5"
MIMIC_REFERENCE_YAW_DEG="${MIMIC_REFERENCE_YAW_DEG:-36.254852}"
MIMIC_UP_HEADING_FIXED="${MIMIC_UP_HEADING_FIXED:-.omx/mimic_stairs_up_reference_yaw_36p254852.hdf5}"

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
    *)
        echo "[ERROR] Unsupported DATASET=$DATASET DIRECTION=$DIRECTION." >&2
        echo "        DATASET: kit167 | upx | terrain | mimic | raw" >&2
        echo "        DIRECTION: up | down | both" >&2
        exit 2
        ;;
esac

if [[ "$DATASET:$DIRECTION" == "mimic:up" && ! -f "$DEFAULT_MOCAP_MOTION" ]]; then
    echo "[INFO] Generating heading-corrected mimic stairs mocap: $DEFAULT_MOCAP_MOTION"
    conda run --no-capture-output -n "$ENV_NAME" \
        python scripts/rotate_humenv_root_yaw.py \
        --input "$MIMIC_UP_SOURCE" \
        --output "$DEFAULT_MOCAP_MOTION" \
        --yaw-deg "$MIMIC_REFERENCE_YAW_DEG"
fi

MOCAP_MOTION="${MOCAP_MOTION:-$DEFAULT_MOCAP_MOTION}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
SPEED="${SPEED:-1.0}"
DURATION="${DURATION:-0}"
START_FRAME="${START_FRAME:-0}"
REPEAT="${REPEAT:-1}"
OUT_DIR="${OUT_DIR:-.omx/mocap_track_updown_stairs_kinematic_visual}"
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

if [[ ! -f "$MOCAP_MOTION" ]]; then
    echo "[ERROR] Motion file not found: $MOCAP_MOTION" >&2
    exit 1
fi

echo "[INFO] Stair mocap replay wrapper"
echo "[INFO] dataset=$DATASET direction=$DIRECTION"

exec env \
    MOCAP_MOTION="$MOCAP_MOTION" \
    MOCAP_EPISODE="$MOCAP_EPISODE" \
    SPEED="$SPEED" \
    DURATION="$DURATION" \
    START_FRAME="$START_FRAME" \
    REPEAT="$REPEAT" \
    OUT_DIR="$OUT_DIR" \
    TERRAIN="$TERRAIN" \
    STAIR_WIDTH="$STAIR_WIDTH" \
    STAIR_HEIGHT="$STAIR_HEIGHT" \
    STAIR_STEPS="$STAIR_STEPS" \
    ./show_mocap_track.sh "$@"
