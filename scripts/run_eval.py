#!/usr/bin/env python3
"""Run golden-task evals against mocked or live workflow outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evals.runner import (  # noqa: E402
    load_golden_tasks,
    record_eval_metrics,
    run_eval_with_outputs,
    summarize_results,
)


def _mock_outputs() -> dict[str, str]:
    """Deterministic outputs for CI without LLM calls."""
    return {
        "langgraph-overview": (
            "LangGraph is a framework for building multi-agent workflows as graphs. "
            "Each agent node updates shared state while a supervisor routes tasks between "
            "specialized researcher, writer, and critic agents until quality thresholds are met."
        ),
        "rag-basics": (
            "- Retrieval-augmented generation combines document retrieval with LLM generation.\n"
            "- Retrieval grounds answers in source material.\n"
            "- Generation synthesizes retrieved context into a final response."
        ),
        "supervisor-pattern": (
            "The supervisor pattern uses a coordinating agent to delegate work to specialized "
            "sub-agents, inspect intermediate outputs, and decide whether to continue, revise, "
            "or finish the workflow."
        ),
    }


async def _live_outputs() -> dict[str, str]:

    from app.db.repository import RunRepository
    from app.db.session import get_session_factory, init_db
    from app.runner import execute_run

    await init_db()
    tasks = load_golden_tasks()
    outputs: dict[str, str] = {}
    factory = get_session_factory()

    for task in tasks:
        async with factory() as session:
            repo = RunRepository(session)
            run = await repo.create(task["task"])
            run_id = run.id
        await execute_run(run_id)
        async with factory() as session:
            repo = RunRepository(session)
            completed = await repo.get(run_id)
            outputs[str(task["id"])] = (completed.final_output if completed else "") or ""
    return outputs


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run Maestro golden-task evals")
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="mock uses deterministic outputs; live runs the full workflow",
    )
    args = parser.parse_args()

    if args.mode == "mock":
        outputs = _mock_outputs()
    else:
        outputs = await _live_outputs()

    results = await run_eval_with_outputs(outputs)
    rate = record_eval_metrics(results)
    summary = summarize_results(results)
    print(json.dumps(summary, indent=2))
    return 0 if rate >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
