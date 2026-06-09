from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.graph.graph import build_graph, initial_state


@pytest.mark.asyncio
async def test_graph_completes_with_mocked_llm() -> None:
    async def fake_invoke(system: str, user: str, *, agent: str) -> str:
        if agent == "supervisor":
            return '{"plan": ["research", "write"], "next_route": "researcher"}'
        if agent == "researcher":
            return "## Research\nLangGraph supports multi-agent workflows."
        if agent == "writer":
            return "# Report\nLangGraph enables multi-agent systems with shared state."
        if agent == "critic":
            return '{"quality_score": 0.95, "critique": "Strong report."}'
        return "ok"

    graph = build_graph()
    state = initial_state("Summarize LangGraph")

    with patch(
        "app.graph.nodes.supervisor.invoke_llm",
        new=AsyncMock(side_effect=fake_invoke),
    ), patch(
        "app.graph.nodes.researcher.invoke_llm",
        new=AsyncMock(side_effect=fake_invoke),
    ), patch(
        "app.graph.nodes.writer.invoke_llm",
        new=AsyncMock(side_effect=fake_invoke),
    ), patch(
        "app.graph.nodes.critic.invoke_llm",
        new=AsyncMock(side_effect=fake_invoke),
    ), patch(
        "app.graph.nodes.researcher.tavily_search",
        new=AsyncMock(return_value="web: LangGraph docs"),
    ), patch(
        "app.graph.nodes.researcher.vector_store_lookup",
        new=AsyncMock(return_value="kb: LangGraph overview"),
    ):
        result = await graph.ainvoke(state)

    assert result.get("draft")
    assert result.get("quality_score", 0) >= 0.75
