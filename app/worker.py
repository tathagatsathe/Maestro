from __future__ import annotations

import asyncio
import logging
import uuid

from prometheus_client import start_http_server

from app.db.session import init_db
from app.observability.langsmith_tracing import configure_langsmith
from app.queue.jobs import JobQueue
from app.runner import execute_run

logger = logging.getLogger(__name__)


async def _handle_run(run_id: uuid.UUID) -> None:
    await execute_run(run_id)


async def run_worker_loop() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    configure_langsmith()
    await init_db()
    start_http_server(9100)
    logger.info("Worker started (metrics on :9100)")

    queue = JobQueue()
    while True:
        processed = await queue.process_one(_handle_run)
        if not processed:
            await asyncio.sleep(0.1)


def main() -> None:
    asyncio.run(run_worker_loop())


if __name__ == "__main__":
    main()
