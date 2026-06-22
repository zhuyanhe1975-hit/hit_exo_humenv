#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
eval "$(python3 scripts/print_latent_z_shell_config.py)"

ENV_NAME="${ENV_NAME:-$LATENT_Z_ENV_NAME}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    python scripts/train_eval_sweep.py "$@"
else
    conda run --no-capture-output -n "$ENV_NAME" python scripts/train_eval_sweep.py "$@"
fi
