#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
echo "[INFO] test.sh has been renamed to test_latent_z.sh for the move-ego latent-z setup."
exec ./test_latent_z.sh "$@"
