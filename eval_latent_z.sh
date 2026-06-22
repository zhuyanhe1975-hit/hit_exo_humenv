#!/usr/bin/env bash
set -euo pipefail

AFTER_LOG="${AFTER_LOG:-}"
BEFORE_LOG="${BEFORE_LOG:-}"

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-$LATENT_Z_VIEWER_POWER_LOG_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LATENT_Z_POWER_OUTPUT_ROOT}"
DROP_FALLEN="${DROP_FALLEN:-$LATENT_Z_EVAL_DROP_FALLEN}"

latest_csv() {
    local root="$1"
    find "$root" -type f -name '*.csv' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

if [[ -z "$AFTER_LOG" ]]; then
    AFTER_LOG="$(latest_csv "$RUN_LOG_ROOT")"
fi

if [[ -z "$AFTER_LOG" ]]; then
    echo "No run CSV found under $RUN_LOG_ROOT. Set AFTER_LOG=/path/to/assisted.csv." >&2
    exit 1
fi

if [[ ! -f "$AFTER_LOG" ]]; then
    echo "AFTER_LOG does not exist: $AFTER_LOG" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_ROOT/${TIMESTAMP}_assist_power.json}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

CMD=(
    python scripts/analyze_assist_power.py
    "$AFTER_LOG"
    --output "$OUTPUT_FILE"
)

if [[ -n "$BEFORE_LOG" ]]; then
    if [[ ! -f "$BEFORE_LOG" ]]; then
        echo "BEFORE_LOG does not exist: $BEFORE_LOG" >&2
        exit 1
    fi
    CMD+=(--before "$BEFORE_LOG")
fi

if [[ "$DROP_FALLEN" == "1" || "$DROP_FALLEN" == "true" || "$DROP_FALLEN" == "yes" ]]; then
    CMD+=(--drop-fallen)
else
    CMD+=(--no-drop-fallen)
fi

echo "[INFO] assisted run log: $AFTER_LOG"
if [[ -n "$BEFORE_LOG" ]]; then
    echo "[INFO] baseline run log: $BEFORE_LOG"
else
    echo "[INFO] baseline run log: <none>; only assisted-run power summary will be written"
fi
echo "[INFO] output: $OUTPUT_FILE"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    "${CMD[@]}"
else
    conda run --no-capture-output -n "$ENV_NAME" "${CMD[@]}"
fi
