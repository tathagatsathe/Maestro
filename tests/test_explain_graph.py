from __future__ import annotations

from app.graph.explain_graph import initial_explain_state, readability_router


def test_initial_explain_state_defaults() -> None:
    state = initial_explain_state(
        paper_text="Paper body text",
        paper_title="Attention Is All You Need",
    )
    assert state["paper_text"] == "Paper body text"
    assert state["paper_title"] == "Attention Is All You Need"
    assert state["retry_count"] == 0
    assert state["quality_score"] == 0.0
    assert state["messages"] == []


def test_readability_router_returns_end_on_approval() -> None:
    from langgraph.graph import END

    state = {
        **initial_explain_state(paper_text="t", paper_title="title"),
        "quality_score": 0.9,
        "final_output": "approved explanation",
        "draft": "approved explanation",
    }
    assert readability_router(state) == END


def test_readability_router_returns_explainer_when_not_approved() -> None:
    state = {
        **initial_explain_state(paper_text="t", paper_title="title"),
        "quality_score": 0.5,
        "draft": "needs work",
        "retry_count": 1,
    }
    assert readability_router(state) == "explainer"


def test_readability_router_returns_end_at_max_retries() -> None:
    from langgraph.graph import END

    state = {
        **initial_explain_state(paper_text="t", paper_title="title"),
        "quality_score": 0.4,
        "draft": "best effort",
        "retry_count": 3,
    }
    assert readability_router(state) == END
