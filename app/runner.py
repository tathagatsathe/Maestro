from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.db.models import RunStatus
from app.db.repository import RunRepository
from app.db.session import get_session_factory
from app.graph.graph import build_graph, initial_state
from app.observability.metrics import (
    QUALITY_SCORE,
    RETRY_COUNT,
    RUN_COST_USD,
    WORKFLOW_DURATION,
)
from app.output import save_report
from app.queue.events import RunEventPublisher
from app.run_context import RunContext, clear_run_context, set_run_context

logger = logging.getLogger(__name__)

_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def execute_run(run_id: uuid.UUID) -> None:
    """Execute a workflow run and publish events to Redis."""
    factory = get_session_factory()
    publisher = RunEventPublisher(str(run_id))
    ctx = RunContext(run_id=run_id)
    set_run_context(ctx)
    start = time.time()

    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.get(run_id)
        if run is None:
            logger.error("Run %s not found", run_id)
            clear_run_context()
            return

        await repo.update_status(run_id, RunStatus.RUNNING)

        try:
            await publisher.publish(
                "run_started",
                {"task": run.task, "run_id": str(run_id)},
            )

            graph = get_graph()
            state = initial_state(run.task)
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
                    await publisher.publish("agent_start", {"agent": name})

                if kind == "on_chain_end" and name in {
                    "supervisor",
                    "researcher",
                    "writer",
                    "critic",
                }:
                    output = data.get("output") or {}
                    await publisher.publish(
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
                                await publisher.publish(
                                    "token",
                                    {
                                        "agent": metadata.get("langgraph_node", ""),
                                        "text": text,
                                    },
                                )

            if final_state is None:
                final_state = await graph.ainvoke(state)

            final_output = final_state.get("final_output") or final_state.get(
                "draft", ""
            )
            quality_score = float(final_state.get("quality_score", 0.0))
            retry_count = int(final_state.get("retry_count", 0))
            output_path: str | None = None

            if final_output.strip():
                saved = save_report(
                    task=run.task,
                    content=final_output,
                    quality_score=quality_score,
                )
                output_path = str(saved)
                logger.info("Report saved to %s (score=%.2f)", output_path, quality_score)

            cost = ctx.estimated_cost_usd()
            await repo.complete(
                run_id,
                final_output=final_output,
                quality_score=quality_score,
                retry_count=retry_count,
                output_path=output_path,
                input_tokens=ctx.input_tokens,
                output_tokens=ctx.output_tokens,
                estimated_cost_usd=cost,
            )

            QUALITY_SCORE.set(quality_score)
            RETRY_COUNT.observe(retry_count)
            RUN_COST_USD.observe(cost)
            await publisher.publish(
                "done",
                {
                    "final_output": final_output,
                    "quality_score": quality_score,
                    "retry_count": retry_count,
                    "output_path": output_path,
                    "input_tokens": ctx.input_tokens,
                    "output_tokens": ctx.output_tokens,
                    "estimated_cost_usd": cost,
                },
            )
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            await repo.fail(run_id, str(exc))
            await publisher.publish("error", {"message": str(exc)})
        finally:
            WORKFLOW_DURATION.observe(time.time() - start)
            await publisher.close()
            clear_run_context()
