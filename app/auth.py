from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    settings = get_settings()
    if not settings.api_key:
        return x_api_key
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key
