#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-mjwarp_env}"
AMASS_RAW_ROOT="${AMASS_RAW_ROOT:-/home/yhzhu/AI/PHC/data/amass/raw}"
HUMENV_ROOT="${HUMENV_ROOT:-/home/yhzhu/AI/humenv}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-$HUMENV_ROOT/scripts/convert_amass_smplsim_motion.py}"
CONVERTED_DIR="${CONVERTED_DIR:-$HUMENV_ROOT/data_preparation/humenv_from_protomotions}"
OUT_ROOT="${OUT_ROOT:-.omx/amass_kit_stairs_playlist}"

AUTO_CONVERT="${AUTO_CONVERT:-0}"
STRICT="${STRICT:-0}"
PLAY_MODE="${PLAY_MODE:-kinematic}"
MOCAP_EPISODE="${MOCAP_EPISODE:-ep_0}"
SPEED="${SPEED:-1.0}"
DURATION="${DURATION:-0}"
START_FRAME="${START_FRAME:-0}"
REPEAT="${REPEAT:-1}"
PAUSE_BETWEEN="${PAUSE_BETWEEN:-0.75}"
TERRAIN="${TERRAIN:-stairs}"
STAIR_WIDTH="${STAIR_WIDTH:-1.4}"
STAIR_HEIGHT="${STAIR_HEIGHT:-0.135}"
STAIR_STEPS="${STAIR_STEPS:-0}"

SEQUENCES=(
    "KIT/167/downstairs01"
    "KIT/167/downstairs02"
    "KIT/167/downstairs03"
    "KIT/167/downstairs04"
    "KIT/167/downstairs05"
    "KIT/167/downstairs06"
    "KIT/167/downstairs07"
    "KIT/167/downstairs08"
    "KIT/167/downstairs09"
    "KIT/167/downstairs10"
    "KIT/167/upstairs01"
    "KIT/167/upstairs02"
    "KIT/167/upstairs03"
    "KIT/167/upstairs04"
    "KIT/167/upstairs05"
    "KIT/167/upstairs06"
    "KIT/167/upstairs07"
    "KIT/167/upstairs08"
    "KIT/167/upstairs09"
    "KIT/167/upstairs10"
    "KIT/167/upstairs_downstairs01"
    "KIT/167/upstairs_downstairs02"
    "KIT/167/upstairs_downstairs03"
    "KIT/167/upstairs_downstairs04"
    "KIT/183/downstairs01"
    "KIT/183/downstairs02"
    "KIT/183/downstairs03"
    "KIT/183/upstairs01"
    "KIT/183/upstairs02"
    "KIT/183/upstairs03"
    "KIT/183/upstairs04"
    "KIT/183/upstairs05"
    "KIT/183/upstairs06"
    "KIT/183/upstairs07"
    "KIT/183/upstairs08"
    "KIT/183/upstairs09"
    "KIT/183/upstairs10"
)

usage() {
    cat <<'EOF'
Play all AMASS/KIT upstairs/downstairs mocap clips in order.

Usage:
  ./show_amass_kit_stairs_playlist.sh [extra playback args...]

Useful env vars:
  AUTO_CONVERT=1|0      Convert missing HumEnv HDF5 files before playback. Default: 0
  PLAY_MODE=kinematic   Use one persistent MuJoCo viewer. Default
  PLAY_MODE=s1          Use test_mocap_track.sh with frozen MetaMotivo S-1 tracking
  DURATION=5            Limit each clip to N seconds. Default: 0, full clip
  PAUSE_BETWEEN=0.75    Seconds to hold between clips in persistent viewer
  STRICT=1              Stop on missing/failed clips. Default: 0, continue
  TERRAIN=stairs        Playback terrain. Default: stairs
  ENV_NAME=mjwarp_env   Conda env used by existing wrappers/conversion

By default this script never creates HDF5 data; it only plays existing files.
Set AUTO_CONVERT=1 explicitly to convert missing clips.
Kinematic mode keeps one MuJoCo viewer open and switches clips inside it.
Use PLAY_MODE=s1 for frozen MetaMotivo S-1 tracking; that mode may reopen
the viewer because each dynamics rollout owns a MuJoCo model.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

sequence_hdf5() {
    local sequence="$1"
    local subject name
    subject="$(cut -d/ -f2 <<<"$sequence")"
    name="$(cut -d/ -f3 <<<"$sequence")"
    printf '%s/KIT_%s_%s_poses.hdf5' "$CONVERTED_DIR" "$subject" "$name"
}

sequence_npz() {
    local sequence="$1"
    printf '%s/%s_poses.npz' "$AMASS_RAW_ROOT" "$sequence"
}

legacy_hdf5_fallback() {
    local sequence="$1"
    case "$sequence" in
        KIT/167/upstairs_downstairs01)
            printf '%s/KIT_167_upstairs_downstairs01_poses.hdf5' "$CONVERTED_DIR"
            ;;
        *)
            return 1
            ;;
    esac
}

run_in_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
        "$@"
    else
        conda run --no-capture-output -n "$ENV_NAME" "$@"
    fi
}

convert_if_needed() {
    local sequence="$1"
    local source="$2"
    local hdf5="$3"
    local temp_npz="${hdf5%.hdf5}_smplsim_qpos.npz"

    if [[ -f "$hdf5" ]]; then
        return 0
    fi
    if [[ "$AUTO_CONVERT" != "1" ]]; then
        return 1
    fi
    if [[ ! -f "$source" ]]; then
        echo "[WARN] Missing AMASS source for $sequence: $source" >&2
        return 1
    fi

    echo "[INFO] Converting $sequence"
    run_in_env python "$CONVERT_SCRIPT" \
        --amass-source "$source" \
        --output "$hdf5" \
        --temp-npz "$temp_npz"
}

play_clip() {
    local sequence="$1"
    local hdf5="$2"
    local out_dir="$OUT_ROOT/${sequence//\//_}"
    shift 2

    echo
    echo "[INFO] Playing $sequence"
    echo "[INFO] HDF5: $hdf5"

    case "$PLAY_MODE" in
        kinematic)
            MOCAP_MOTION="$hdf5" \
            MOCAP_EPISODE="$MOCAP_EPISODE" \
            SPEED="$SPEED" \
            DURATION="$DURATION" \
            START_FRAME="$START_FRAME" \
            REPEAT="$REPEAT" \
            OUT_DIR="$out_dir" \
            TERRAIN="$TERRAIN" \
            STAIR_WIDTH="$STAIR_WIDTH" \
            STAIR_HEIGHT="$STAIR_HEIGHT" \
            STAIR_STEPS="$STAIR_STEPS" \
            ./show_mocap_track.sh "$@"
            ;;
        s1)
            MOCAP_MOTION="$hdf5" \
            MOCAP_EPISODE="$MOCAP_EPISODE" \
            DURATION="$DURATION" \
            START_FRAME="$START_FRAME" \
            REPEAT="$REPEAT" \
            OUT_DIR="$out_dir" \
            TERRAIN="$TERRAIN" \
            STAIR_WIDTH="$STAIR_WIDTH" \
            STAIR_HEIGHT="$STAIR_HEIGHT" \
            STAIR_STEPS="$STAIR_STEPS" \
            ./test_mocap_track.sh "$@"
            ;;
        *)
            echo "[ERROR] Unsupported PLAY_MODE=$PLAY_MODE. Use kinematic or s1." >&2
            exit 2
            ;;
    esac
}

mkdir -p "$OUT_ROOT" "$CONVERTED_DIR"

echo "[INFO] AMASS/KIT stair playlist: ${#SEQUENCES[@]} clips"
echo "[INFO] PLAY_MODE=$PLAY_MODE AUTO_CONVERT=$AUTO_CONVERT TERRAIN=$TERRAIN"

played=0
skipped=0
playable_sequences=()
playable_hdf5=()
for sequence in "${SEQUENCES[@]}"; do
    source="$(sequence_npz "$sequence")"
    hdf5="$(sequence_hdf5 "$sequence")"
    if [[ ! -f "$hdf5" ]] && fallback="$(legacy_hdf5_fallback "$sequence" 2>/dev/null)" && [[ -f "$fallback" ]]; then
        hdf5="$fallback"
    fi

    if ! convert_if_needed "$sequence" "$source" "$hdf5"; then
        echo "[WARN] Skipping $sequence; missing converted HDF5: $hdf5" >&2
        skipped=$((skipped + 1))
        if [[ "$STRICT" == "1" ]]; then
            exit 1
        fi
        continue
    fi

    if [[ "$PLAY_MODE" == "kinematic" ]]; then
        playable_sequences+=("$sequence")
        playable_hdf5+=("$hdf5")
        played=$((played + 1))
        continue
    fi

    if play_clip "$sequence" "$hdf5" "$@"; then
        played=$((played + 1))
    else
        echo "[WARN] Playback failed for $sequence" >&2
        skipped=$((skipped + 1))
        if [[ "$STRICT" == "1" ]]; then
            exit 1
        fi
    fi
done

if [[ "$PLAY_MODE" == "kinematic" ]]; then
    if [[ "${#playable_hdf5[@]}" -eq 0 ]]; then
        echo "[ERROR] No converted stair clips available to play." >&2
        exit 1
    fi

    playlist_cmd=(
        python scripts/show_mocap_playlist_kinematic.py
        --episode "$MOCAP_EPISODE"
        --speed "$SPEED"
        --duration "$DURATION"
        --start-frame "$START_FRAME"
        --repeat "$REPEAT"
        --pause-between "$PAUSE_BETWEEN"
        --out-dir "$OUT_ROOT"
        --terrain "$TERRAIN"
        --stair-width "$STAIR_WIDTH"
        --stair-height "$STAIR_HEIGHT"
        --stair-steps "$STAIR_STEPS"
    )
    for idx in "${!playable_hdf5[@]}"; do
        playlist_cmd+=(--motion "${playable_hdf5[$idx]}" --label "${playable_sequences[$idx]}")
    done
    playlist_cmd+=("$@")

    echo
    echo "[INFO] Launching one persistent viewer for ${#playable_hdf5[@]} clips"
    run_in_env "${playlist_cmd[@]}"
fi

echo
echo "[INFO] Stair playlist complete: played=$played skipped=$skipped total=${#SEQUENCES[@]}"
