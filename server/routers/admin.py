"""POST /admin/update — request a git pull + server rebuild from the host."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends

from auth import get_current_device
from config import settings

router = APIRouter(prefix="/admin", tags=["admin"])

_TRIGGER_FILE = settings.data_dir / ".update_trigger"


@router.post("/update")
async def update_server(_: str = Depends(get_current_device)) -> dict:
    """Write a trigger file that the host-side watcher picks up.

    The host systemd path unit detects the file, runs update_and_restart.sh
    (git pull + docker compose up --build -d), then deletes the trigger.
    """
    _TRIGGER_FILE.write_text(f"triggered_at={time.time()}\n")
    return {"status": "updating"}
