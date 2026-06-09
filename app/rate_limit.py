from __future__ import annotations

import time

from fastapi import HTTPException, status

from app.config import get_settings
from app.queue.client import get_redis


async def check_rate_limit(api_key: str) -> None:
    settings = get_settings()
    if settings.rate_limit_runs_per_minute <= 0:
        return

    redis = get_redis()
    window = int(time.time() // 60)
    key = f"maestro:ratelimit:{api_key}:{window}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 120)

    if count > settings.rate_limit_runs_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for run submissions",
        )
