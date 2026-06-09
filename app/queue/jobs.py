from __future__ import annotations

import json
import logging
import time
import uuid

from app.config import get_settings
from app.observability.metrics import JOB_DURATION, QUEUE_DEPTH, WORKER_ACTIVE_JOBS
from app.queue.client import get_redis

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self) -> None:
        self._redis = get_redis()
        self._settings = get_settings()

    async def enqueue(self, run_id: uuid.UUID) -> None:
        payload = json.dumps({"run_id": str(run_id)})
        await self._redis.lpush(self._settings.job_queue_key, payload)
        depth = await self._redis.llen(self._settings.job_queue_key)
        QUEUE_DEPTH.set(depth)
        logger.info("Enqueued run %s (queue_depth=%d)", run_id, depth)

    async def dequeue(self, timeout: int = 5) -> uuid.UUID | None:
        result = await self._redis.brpop(self._settings.job_queue_key, timeout=timeout)
        if result is None:
            depth = await self._redis.llen(self._settings.job_queue_key)
            QUEUE_DEPTH.set(depth)
            return None
        _key, payload = result
        data = json.loads(payload)
        depth = await self._redis.llen(self._settings.job_queue_key)
        QUEUE_DEPTH.set(depth)
        return uuid.UUID(data["run_id"])

    async def process_one(self, handler) -> bool:
        """Dequeue and process a single job; returns False if queue was empty."""
        run_id = await self.dequeue(timeout=5)
        if run_id is None:
            return False

        WORKER_ACTIVE_JOBS.inc()
        start = time.time()
        try:
            await handler(run_id)
            JOB_DURATION.observe(time.time() - start)
            return True
        finally:
            WORKER_ACTIVE_JOBS.dec()
