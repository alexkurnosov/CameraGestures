"""JWT creation and validation utilities.

create_access_token — signs a new JWT for a given device_id.
get_current_device  — FastAPI dependency; decodes the Bearer token from the
                      Authorization header and returns the device_id claim.
                      Raises HTTP 401 if the token is missing, malformed, or expired.
"""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/register")


def create_access_token(device_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_expire_days)
    payload = {"sub": device_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


async def get_current_device(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        device_id: str | None = payload.get("sub")
        if device_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return device_id
