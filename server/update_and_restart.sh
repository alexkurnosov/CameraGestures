#!/bin/bash
# Pull the latest code from main and rebuild/restart the server container.
# The db container is left untouched (its data lives in a Docker named volume).
#
# Intended to be called by the cameragestures-update.service systemd unit,
# but can also be run manually: bash update_and_restart.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "[update] $(date -Iseconds) Pulling latest changes from origin/main..."
git -C "$REPO_DIR" pull origin main

echo "[update] $(date -Iseconds) Rebuilding and restarting server container..."
cd "$SCRIPT_DIR"
docker compose up --build -d server

echo "[update] $(date -Iseconds) Done."
