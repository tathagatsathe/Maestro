from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.db.models import Run, RunStatus
from app.db.repository import RunRepository
from app.graph.explain_graph import build_explain_graph, initial_explain_state
from app.observability.metrics import (
    QUALITY_SCORE,
    RETRY_COUNT,
    RUN_COST_USD,
    WORKFLOW_DURATION,
)
from app.output import save_report
from app.pdf.storage import load_extracted_text
from app.queue.events import RunEventPublisher
from app.run_context import RunContext, clear_run_context, set_run_context

logger = logging.getLogger(__name__)

_compiled_explain_graph = None

EXPLAIN_AGENTS = {"analyzer", "explainer", "readability_critic"}

EXPLAIN_AGENT_END_KEYS = {
    "paper_brief",
    "draft",
    "critique",
    "quality_score",
    "retry_count",
    "current_agent",
}


def get_explain_graph():
    global _compiled_explain_graph
    if _compiled_explain_graph is None:
        _compiled_explain_graph = build_explain_graph()
    return _compiled_explain_graph


def _paper_title_from_run(run: Run) -> str:
    if run.source_filename:
        return Path(run.source_filename).stem.replace("_", " ").replace("-", " ")
    return "Research paper"


async def execute_explain_run(run_id: uuid.UUID) -> None:
    """Execute a paper explanation workflow and publish events to Redis."""
    from app.db.session import get_session_factory

    factory = get_session_factory()
    publisher = RunEventPublisher(str(run_id))
    ctx = RunContext(run_id=run_id)
    set_run_context(ctx)
    start = time.time()

    async with factory() as session:
        repo = RunRepository(session)
        run = await repo.get(run_id)
        if run is None:
            logger.error("Explain run %s not found", run_id)
            clear_run_context()
            return

        await repo.update_status(run_id, RunStatus.RUNNING)

        try:
            paper_text = load_extracted_text(run_id)
            paper_title = _paper_title_from_run(run)

            await publisher.publish(
                "run_started",
                {
                    "task": run.task,
                    "run_id": str(run_id),
                    "run_type": run.run_type,
                    "source_filename": run.source_filename,
                },
            )

            graph = get_explain_graph()
            state = initial_explain_state(
                paper_text=paper_text,
                paper_title=paper_title,
            )
            final_state: dict[str, Any] | None = None

            async for event in graph.astream_events(state, version="v2"):
                kind = event.get("event")
                name = event.get("name", "")
                data = event.get("data", {})
                metadata = event.get("metadata", {})

                if kind == "on_chain_start" and name in EXPLAIN_AGENTS:
                    await publisher.publish("agent_start", {"agent": name})

                if kind == "on_chain_end" and name in EXPLAIN_AGENTS:
                    output = data.get("output") or {}
                    await publisher.publish(
                        "agent_end",
                        {
                            "agent": name,
                            "updates": {
                                k: v
                                for k, v in output.items()
                                if k in EXPLAIN_AGENT_END_KEYS
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
                logger.info(
                    "Explanation saved to %s (score=%.2f)", output_path, quality_score
                )

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
            logger.exception("Explain run %s failed", run_id)
            await repo.fail(run_id, str(exc))
            await publisher.publish("error", {"message": str(exc)})
        finally:
            WORKFLOW_DURATION.observe(time.time() - start)
            await publisher.close()
            clear_run_context()
