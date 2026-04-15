#!/bin/bash
# Pull the latest code from main and rebuild/restart Docker Compose services.
# Run this script from any location — it always operates on the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "[update] Pulling latest changes from origin/main..."
git -C "$REPO_DIR" pull origin main

echo "[update] Rebuilding and restarting Docker Compose services..."
cd "$SCRIPT_DIR"
docker compose up --build -d

echo "[update] Done."
