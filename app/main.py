from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.graph import build_graph, initial_state
from app.observability.langsmith_tracing import configure_langsmith
from app.output import save_report

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    configure_langsmith()
    logger.info("Multi-agent workflow engine started")
    yield


app = FastAPI(
    title="Multi-Agent Workflow Engine",
    version="0.1.0",
    lifespan=lifespan,
)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunRecord:
    run_id: str
    task: str
    status: RunStatus = RunStatus.PENDING
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    final_output: str = ""
    error: str | None = None


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, task: str) -> RunRecord:
        run_id = str(uuid.uuid4())
        record = RunRecord(run_id=run_id, task=task)
        async with self._lock:
            self._runs[run_id] = record
        return record

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._lock:
            return self._runs.get(run_id)


run_store = RunStore()
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, description="User task / prompt")


class RunResponse(BaseModel):
    run_id: str
    status: RunStatus


async def _publish(record: RunRecord, event_type: str, payload: dict[str, Any]) -> None:
    event = {"type": event_type, **payload}
    await record.events.put(event)


async def _execute_run(record: RunRecord) -> None:
    record.status = RunStatus.RUNNING
    graph = get_graph()
    state = initial_state(record.task)

    try:
        await _publish(
            record,
            "run_started",
            {"task": record.task, "run_id": record.run_id},
        )

        final_state: dict[str, Any] | None = None

        async for event in graph.astream_events(state, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})
            metadata = event.get("metadata", {})

            if kind == "on_chain_start" and name in {
                "supervisor",
                "researcher",
                "writer",
                "critic",
            }:
                await _publish(
                    record,
                    "agent_start",
                    {"agent": name},
                )

            if kind == "on_chain_end" and name in {
                "supervisor",
                "researcher",
                "writer",
                "critic",
            }:
                output = data.get("output") or {}
                await _publish(
                    record,
                    "agent_end",
                    {
                        "agent": name,
                        "updates": {
                            k: v
                            for k, v in output.items()
                            if k
                            in {
                                "plan",
                                "research",
                                "draft",
                                "critique",
                                "quality_score",
                                "retry_count",
                                "current_agent",
                                "next_route",
                            }
                            and (not isinstance(v, str) or len(v) <= 500)
                        },
                    },
                )

            if kind == "on_chain_end" and name == "LangGraph":
                final_state = data.get("output")

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is not None:
                    content = getattr(chunk, "content", None)
                    if content:
                        if isinstance(content, list):
                            text = "".join(
                                block.get("text", "")
                                if isinstance(block, dict)
                                else str(block)
                                for block in content
                            )
                        else:
                            text = str(content)
                        if text:
                            await _publish(
                                record,
                                "token",
                                {
                                    "agent": metadata.get("langgraph_node", ""),
                                    "text": text,
                                },
                            )

        if final_state is None:
            final_state = await graph.ainvoke(state)

        record.final_output = final_state.get("final_output") or final_state.get(
            "draft", ""
        )
        quality_score = float(final_state.get("quality_score", 0.0))
        output_path: str | None = None
        if record.final_output.strip():
            saved = save_report(
                task=record.task,
                content=record.final_output,
                quality_score=quality_score,
            )
            output_path = str(saved)
            logger.info("Report saved to %s (score=%.2f)", output_path, quality_score)

        record.status = RunStatus.COMPLETED
        await _publish(
            record,
            "done",
            {
                "final_output": record.final_output,
                "quality_score": quality_score,
                "retry_count": final_state.get("retry_count", 0),
                "output_path": output_path,
            },
        )
    except Exception as exc:
        record.status = RunStatus.FAILED
        record.error = str(exc)
        await _publish(record, "error", {"message": str(exc)})
    finally:
        await record.events.put({"type": "_stream_end"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def start_run(body: RunRequest) -> RunResponse:
    record = await run_store.create(body.task)
    asyncio.create_task(_execute_run(record))
    return RunResponse(run_id=record.run_id, status=record.status)


async def _sse_generator(record: RunRecord) -> AsyncIterator[str]:
    while True:
        event = await record.events.get()
        if event.get("type") == "_stream_end":
            break
        yield f"data: {json.dumps(event)}\n\n"


@app.get("/run/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    record = await run_store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return StreamingResponse(
        _sse_generator(record),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
