from __future__ import annotations

import pytest

from app.evals.runner import evaluate_output, run_eval_with_outputs, summarize_results


@pytest.mark.asyncio
async def test_eval_passes_with_valid_mock_output() -> None:
    outputs = {
        "langgraph-overview": (
            "LangGraph is a framework for building multi-agent workflows as graphs. "
            "Each agent node updates shared state while a supervisor routes tasks between "
            "specialized researcher, writer, and critic agents until quality thresholds are met. "
            "This pattern makes long-running research pipelines observable and testable."
        ),
        "rag-basics": (
            "- Retrieval-augmented generation combines document retrieval with LLM generation.\n"
            "- Retrieval grounds answers in source material from a knowledge base or the web.\n"
            "- Generation synthesizes retrieved context into a final response for the user."
        ),
        "supervisor-pattern": (
            "The supervisor pattern uses a coordinating agent to delegate work to specialized "
            "sub-agents, inspect intermediate outputs, and decide whether to continue, revise, "
            "or finish the workflow when quality and retry limits are satisfied."
        ),
    }
    results = await run_eval_with_outputs(outputs)
    summary = summarize_results(results)
    assert summary["passed"] == summary["total"]


def test_eval_fails_when_term_missing() -> None:
    task = {
        "id": "rag-basics",
        "min_output_length": 50,
        "required_terms": ["retrieval", "generation"],
    }
    result = evaluate_output(task, "too short")
    assert result.passed is False
