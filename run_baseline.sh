#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LATENT_Z_POWER_RUNS_ROOT}"
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

CMD=(
    python scripts/eval_latent_z_power.py
    --agent zero
    --output-root "$OUTPUT_ROOT"
    --num-envs "$NUM_ENVS"
    --steps "$STEPS"
    --seed "$SEED"
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

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    exec "${CMD[@]}"
else
    exec conda run --no-capture-output -n "$ENV_NAME" "${CMD[@]}"
fi
