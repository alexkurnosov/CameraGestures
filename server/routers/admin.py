"""POST /admin/update — pull latest code from git and rebuild/restart via Docker Compose."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_device

router = APIRouter(prefix="/admin", tags=["admin"])

_UPDATE_SCRIPT = Path(__file__).parent.parent / "update_and_restart.sh"


@router.post("/update")
async def update_server(_: str = Depends(get_current_device)) -> dict:
    """Pull latest code from origin/main and rebuild + restart the server.

    Returns immediately with {"status": "updating"} — the server will restart
    in the background so the caller should not expect a follow-up response.
    """
    if not _UPDATE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="update script not found")

    asyncio.create_task(_run_update())
    return {"status": "updating"}


async def _run_update() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_update_sync)


def _run_update_sync() -> None:
    subprocess.run(["bash", str(_UPDATE_SCRIPT)], check=True)
