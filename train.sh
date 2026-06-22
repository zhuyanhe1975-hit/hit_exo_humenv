#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
echo "[INFO] train.sh has been renamed to train_latent_z.sh for the move-ego latent-z setup."
exec ./train_latent_z.sh "$@"
