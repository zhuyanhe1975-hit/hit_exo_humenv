#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
echo "[INFO] run.sh has been renamed to run_latent_z.sh for the move-ego latent-z setup."
exec ./run_latent_z.sh "$@"
