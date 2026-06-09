from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.config import get_settings
from app.queue.client import get_redis

logger = logging.getLogger(__name__)

STREAM_END = "_stream_end"


class RunEventPublisher:
    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._redis = get_redis()
        settings = get_settings()
        self._stream_key = f"{settings.run_events_stream_prefix}:{run_id}"

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, **payload}
        await self._redis.xadd(
            self._stream_key,
            {"data": json.dumps(event)},
            maxlen=10_000,
            approximate=True,
        )

    async def close(self) -> None:
        await self._redis.xadd(
            self._stream_key,
            {"data": json.dumps({"type": STREAM_END})},
        )
        await self._redis.expire(self._stream_key, 3600)


async def stream_run_events(run_id: str, last_id: str = "0-0") -> AsyncIterator[str]:
    """Read SSE-formatted events from a Redis stream."""
    redis = get_redis()
    settings = get_settings()
    stream_key = f"{settings.run_events_stream_prefix}:{run_id}"
    cursor = last_id

    while True:
        entries = await redis.xread({stream_key: cursor}, block=5000, count=50)
        if not entries:
            continue

        for _stream, messages in entries:
            for message_id, fields in messages:
                cursor = message_id
                raw = fields.get("data", "{}")
                event = json.loads(raw)
                if event.get("type") == STREAM_END:
                    return
                yield f"data: {json.dumps(event)}\n\n"
