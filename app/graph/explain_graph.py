from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.graph.explain_state import ExplainState
from app.graph.nodes.analyzer import analyzer_node
from app.graph.nodes.explainer import explainer_node
from app.graph.nodes.readability_critic import readability_critic_node
from app.observability.langsmith_tracing import traced_agent


def readability_router(state: ExplainState) -> str:
    """Route from readability critic to explainer or END."""
    settings = get_settings()
    quality = state.get("quality_score", 0.0)
    retry_count = state.get("retry_count", 0)

    if quality >= settings.quality_threshold and state.get("final_output"):
        return END
    if retry_count >= settings.max_retries and state.get("draft"):
        return END

    return "explainer"


def build_explain_graph():
    graph = StateGraph(ExplainState)

    graph.add_node("analyzer", traced_agent("analyzer", analyzer_node))
    graph.add_node("explainer", traced_agent("explainer", explainer_node))
    graph.add_node(
        "readability_critic",
        traced_agent("readability_critic", readability_critic_node),
    )

    graph.set_entry_point("analyzer")
    graph.add_edge("analyzer", "explainer")
    graph.add_edge("explainer", "readability_critic")
    graph.add_conditional_edges(
        "readability_critic",
        readability_router,
        {
            "explainer": "explainer",
            END: END,
        },
    )

    return graph.compile()


def initial_explain_state(
    *,
    paper_text: str,
    paper_title: str,
) -> ExplainState:
    return ExplainState(
        paper_text=paper_text,
        paper_title=paper_title,
        paper_brief="",
        draft="",
        critique="",
        quality_score=0.0,
        retry_count=0,
        final_output="",
        current_agent="analyzer",
        messages=[],
    )
