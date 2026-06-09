from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth import verify_api_key


@pytest.mark.asyncio
async def test_verify_api_key_accepts_valid_key(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "secret")
    from app.config import get_settings

    get_settings.cache_clear()
    result = await verify_api_key(x_api_key="secret")
    assert result == "secret"


@pytest.mark.asyncio
async def test_verify_api_key_rejects_invalid_key(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "secret")
    from app.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(x_api_key="bad")
    assert exc.value.status_code == 401
