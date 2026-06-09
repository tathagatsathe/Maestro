from __future__ import annotations

from app.graph.graph import initial_state, supervisor_router
from app.graph.state import AgentState


def test_initial_state_defaults() -> None:
    state = initial_state("Analyze LangGraph patterns")
    assert state["task"] == "Analyze LangGraph patterns"
    assert state["retry_count"] == 0
    assert state["quality_score"] == 0.0
    assert state["next_route"] == "researcher"
    assert state["messages"] == []


def test_supervisor_routes_to_researcher_without_research() -> None:
    state: AgentState = initial_state("test task")
    assert supervisor_router(state) == "researcher"


def test_supervisor_routes_to_writer_with_research_only() -> None:
    state: AgentState = {
        **initial_state("test"),
        "research": "some research",
        "next_route": "",
    }
    assert supervisor_router(state) == "writer"


def test_supervisor_routes_to_critic_with_draft() -> None:
    state: AgentState = {
        **initial_state("test"),
        "research": "research",
        "draft": "draft content",
        "next_route": "",
    }
    assert supervisor_router(state) == "critic"


def test_supervisor_routes_end_on_high_quality() -> None:
    state: AgentState = {
        **initial_state("test"),
        "quality_score": 0.9,
        "final_output": "approved report",
        "draft": "approved report",
    }
    from langgraph.graph import END

    assert supervisor_router(state) == END


def test_supervisor_routes_end_at_max_retries() -> None:
    state: AgentState = {
        **initial_state("test"),
        "retry_count": 3,
        "draft": "best effort draft",
        "quality_score": 0.4,
    }
    from langgraph.graph import END

    assert supervisor_router(state) == END


def test_supervisor_respects_explicit_next_route() -> None:
    state: AgentState = {
        **initial_state("test"),
        "research": "r",
        "draft": "d",
        "next_route": "writer",
    }
    assert supervisor_router(state) == "writer"


def test_merge_messages_reducer() -> None:
    from app.graph.state import merge_messages

    assert merge_messages([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert merge_messages(None, [{"x": 1}]) == [{"x": 1}]
