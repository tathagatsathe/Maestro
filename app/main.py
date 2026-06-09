from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from app.auth import verify_api_key
from app.db.models import RunStatus
from app.db.repository import RunRepository
from app.db.session import get_session_factory, init_db
from app.observability.langsmith_tracing import configure_langsmith
from app.queue.events import stream_run_events
from app.queue.jobs import JobQueue
from app.rag.ingest import ingest_document
from app.rate_limit import check_rate_limit

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    configure_langsmith()
    await init_db()
    logger.info("Multi-agent workflow API started")
    yield


app = FastAPI(
    title="Multi-Agent Workflow Engine",
    version="0.2.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000, description="User task / prompt")


class RunResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunDetailResponse(BaseModel):
    run_id: str
    task: str
    status: RunStatus
    quality_score: Optional[float] = None
    retry_count: Optional[int] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, max_length=100_000)
    source: Optional[str] = Field(default=None, max_length=512)


class DocumentResponse(BaseModel):
    document_id: str


async def _get_repo() -> RunRepository:
    factory = get_session_factory()
    async with factory() as session:
        yield RunRepository(session)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def start_run(
    body: RunRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
) -> RunResponse:
    await check_rate_limit(api_key)

    factory = get_session_factory()
    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.create(body.task)

    queue = JobQueue()
    await queue.enqueue(run.id)

    return RunResponse(run_id=str(run.id), status=RunStatus(run.status))


@app.get("/run/{run_id}", response_model=RunDetailResponse)
async def get_run(
    run_id: str,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> RunDetailResponse:
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id") from exc

    factory = get_session_factory()
    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.get(run_uuid)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return RunDetailResponse(
            run_id=str(run.id),
            task=run.task,
            status=RunStatus(run.status),
            quality_score=run.quality_score,
            retry_count=run.retry_count,
            output_path=run.output_path,
            error=run.error,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            estimated_cost_usd=run.estimated_cost_usd,
        )


@app.get("/run/{run_id}/report")
async def get_run_report(
    run_id: str,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> PlainTextResponse:
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id") from exc

    factory = get_session_factory()
    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.get(run_uuid)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status != RunStatus.COMPLETED.value or not run.final_output:
            raise HTTPException(status_code=404, detail="Report not available")

        return PlainTextResponse(run.final_output, media_type="text/markdown")


@app.get("/run/{run_id}/stream")
async def stream_run(
    run_id: str,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> StreamingResponse:
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id") from exc

    factory = get_session_factory()
    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.get(uuid.UUID(run_id))
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

    return StreamingResponse(
        stream_run_events(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/documents", response_model=DocumentResponse)
async def create_document(
    body: DocumentRequest,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> DocumentResponse:
    try:
        document_id = await ingest_document(
            title=body.title,
            content=body.content,
            source=body.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentResponse(document_id=document_id)
