from __future__ import annotations

from app.graph.nodes._llm import append_message, invoke_llm
from app.graph.state import AgentState

WRITER_SYSTEM = """You are the Writer agent.
Produce a polished, structured markdown report based on the research brief.

Requirements:
- Title (# heading)
- Executive summary
- Main sections with ## headings
- Bullet points and short paragraphs
- References section listing sources mentioned in research

If critic feedback is provided, revise the draft to address every point."""


async def writer_node(state: AgentState) -> dict:
    task = state.get("task", "")
    research = state.get("research", "")
    critique = state.get("critique", "")
    previous_draft = state.get("draft", "")
    retry_count = state.get("retry_count", 0)
    messages = list(state.get("messages", []))

    revision_note = ""
    if critique and retry_count > 0:
        revision_note = f"""
Previous draft (excerpt):
{previous_draft[:2000]}

Critic feedback to address:
{critique}
"""

    user_prompt = f"""Task: {task}

Research brief:
{research}
{revision_note}

Write the full markdown report now."""

    draft = await invoke_llm(WRITER_SYSTEM, user_prompt, agent="writer")
    messages = append_message(messages, "writer", draft[:500] + ("..." if len(draft) > 500 else ""))

    return {
        "current_agent": "writer",
        "draft": draft,
        "messages": messages,
    }
