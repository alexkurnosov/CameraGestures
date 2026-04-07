"""POST /auth/register — exchange the pre-shared registration_token for a per-device JWT."""

import hmac
import time

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from auth import create_access_token
from config import settings
from database import devices_table, get_engine
from models import DeviceRegisterRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register_device(body: DeviceRegisterRequest) -> TokenResponse:
    """
    Validate the pre-shared registration_token, record the device, and return a JWT.

    Idempotent: re-registering the same device_id refreshes registered_at and
    issues a fresh token (used by the "Re-register Device" button in iOS Settings).
    """
    if not hmac.compare_digest(body.registration_token, settings.registration_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid registration token",
        )

    engine = get_engine()
    async with engine.begin() as conn:
        now = time.time()
        await conn.execute(
            sqlite_insert(devices_table)
            .values(device_id=body.device_id, registered_at=now)
            .on_conflict_do_update(
                index_elements=["device_id"],
                set_={"registered_at": now},
            )
        )

    return TokenResponse(access_token=create_access_token(body.device_id))
