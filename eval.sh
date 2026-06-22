#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$LATENT_Z_CHECKPOINT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LATENT_Z_POWER_OUTPUT_ROOT}"
NUM_ENVS="${NUM_ENVS:-$LATENT_Z_EVAL_NUM_ENVS}"
STEPS="${STEPS:-$LATENT_Z_EVAL_STEPS}"
SEED="${SEED:-$LATENT_Z_EVAL_SEED}"
RANDOM_WALK_SPEED="${RANDOM_WALK_SPEED:-$LATENT_Z_RANDOM_WALK_SPEED}"
WALK_SPEED="${WALK_SPEED:-$LATENT_Z_NORMAL_WALK_SPEED}"
RANDOM_WALK_DIRECTION="${RANDOM_WALK_DIRECTION:-$LATENT_Z_RANDOM_WALK_DIRECTION}"
WALK_DIRECTION="${WALK_DIRECTION:-$LATENT_Z_WALK_DIRECTION}"
S1_LATENT_SPEED_SCALE="${S1_LATENT_SPEED_SCALE:-$LATENT_Z_S1_LATENT_SPEED_SCALE}"
HUMAN_ACTION_REPEAT="${HUMAN_ACTION_REPEAT:-$LATENT_Z_HUMAN_ACTION_REPEAT}"
HUMAN_ACTION_SMOOTHING="${HUMAN_ACTION_SMOOTHING:-$LATENT_Z_HUMAN_ACTION_SMOOTHING}"
HUMAN_ROOT_HEIGHT="${HUMAN_ROOT_HEIGHT:-$LATENT_Z_HUMAN_ROOT_HEIGHT}"
EXO_JOINT_GROUP="${EXO_JOINT_GROUP:-$LATENT_Z_EXO_JOINT_GROUP}"
DROP_FALLEN="${DROP_FALLEN:-$LATENT_Z_EVAL_DROP_FALLEN}"
export EXO_JOINT_GROUP

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$OUTPUT_ROOT/${TIMESTAMP}_headless_compare"
BASELINE_CSV="$RUN_DIR/baseline_zero.csv"
ASSISTED_CSV="$RUN_DIR/assisted_trained.csv"
ANALYSIS_JSON="$RUN_DIR/assist_power.json"
REPORT_MD="$RUN_DIR/report.md"
mkdir -p "$RUN_DIR"

COMMON_ARGS=(
    --num-envs "$NUM_ENVS"
    --steps "$STEPS"
    --seed "$SEED"
    --s1-latent-speed-scale "$S1_LATENT_SPEED_SCALE"
    --human-action-repeat "$HUMAN_ACTION_REPEAT"
    --human-action-smoothing "$HUMAN_ACTION_SMOOTHING"
    --human-root-height "$HUMAN_ROOT_HEIGHT"
)

if [[ "$RANDOM_WALK_SPEED" == "1" || "$RANDOM_WALK_SPEED" == "true" || "$RANDOM_WALK_SPEED" == "yes" ]]; then
    COMMON_ARGS+=(--random-walk-speed)
else
    COMMON_ARGS+=(--no-random-walk-speed --walk-speed "$WALK_SPEED")
fi

if [[ "$RANDOM_WALK_DIRECTION" == "1" || "$RANDOM_WALK_DIRECTION" == "true" || "$RANDOM_WALK_DIRECTION" == "yes" ]]; then
    COMMON_ARGS+=(--random-walk-direction)
else
    COMMON_ARGS+=(--no-random-walk-direction --walk-direction "$WALK_DIRECTION")
fi

run_python() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
        python "$@"
    else
        conda run --no-capture-output -n "$ENV_NAME" python "$@"
    fi
}

echo "[INFO] running baseline zero-exo rollout -> $BASELINE_CSV"
run_python scripts/eval_latent_z_power.py \
    --agent zero \
    --output-csv "$BASELINE_CSV" \
    "${COMMON_ARGS[@]}" \
    "$@"

echo "[INFO] running assisted trained rollout -> $ASSISTED_CSV"
run_python scripts/eval_latent_z_power.py \
    --agent trained \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-csv "$ASSISTED_CSV" \
    "${COMMON_ARGS[@]}" \
    "$@"

ANALYZE_ARGS=("$ASSISTED_CSV" --before "$BASELINE_CSV" --output "$ANALYSIS_JSON")
if [[ "$DROP_FALLEN" == "1" || "$DROP_FALLEN" == "true" || "$DROP_FALLEN" == "yes" ]]; then
    ANALYZE_ARGS+=(--drop-fallen)
else
    ANALYZE_ARGS+=(--no-drop-fallen)
fi

run_python scripts/analyze_assist_power.py "${ANALYZE_ARGS[@]}"
run_python scripts/write_latent_z_power_report.py "$ANALYSIS_JSON" --output "$REPORT_MD"

echo "[INFO] comparison dir: $RUN_DIR"
echo "[INFO] concise report: $REPORT_MD"
