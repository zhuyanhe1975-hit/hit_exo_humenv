#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mjwarp_env}"

cd "$(dirname "$0")"

CMD=(python scripts/test_mjlab_viewer.py "$@")

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
