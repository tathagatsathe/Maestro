from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    critic_node,
    researcher_node,
    supervisor_node,
    writer_node,
)
from app.graph.state import AgentState


def supervisor_router(state: AgentState) -> str:
    """Route from supervisor to researcher, writer, critic, or END."""
    next_route = state.get("next_route", "")
    quality = state.get("quality_score", 0.0)
    retry_count = state.get("retry_count", 0)

    from app.config import get_settings

    settings = get_settings()
    if quality >= settings.quality_threshold and state.get("final_output"):
        return END
    if retry_count >= settings.max_retries and state.get("draft"):
        return END

    route = str(next_route).lower()
    if route == "end":
        return END
    if route in {"researcher", "writer", "critic"}:
        return route
    if not state.get("research"):
        return "researcher"
    if not state.get("draft"):
        return "writer"
    return "critic"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            END: END,
        },
    )

    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", "supervisor")

    return graph.compile()


def initial_state(task: str) -> AgentState:
    return AgentState(
        task=task,
        plan=[],
        research="",
        draft="",
        critique="",
        quality_score=0.0,
        retry_count=0,
        final_output="",
        current_agent="supervisor",
        messages=[],
        next_route="researcher",
    )
