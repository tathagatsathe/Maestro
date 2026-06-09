from __future__ import annotations

from app.graph.explain_state import ExplainState
from app.graph.nodes._llm import append_message, invoke_llm

EXPLAINER_SYSTEM = """You are the Explainer agent. Write a clear explanation of a research paper
for someone who is NOT an expert in the field.

Requirements:
- Use plain language; define jargon when you must use technical terms
- Use short paragraphs and bullet points where helpful
- Do not use equations unless absolutely necessary (and explain them simply)
- Stay accurate to the paper brief; do not invent findings

Structure your markdown with these sections:
# [Paper title]
## TL;DR
## Why this matters
## What they found
## How they did it
## Limitations
## Glossary

If critic feedback is provided, revise the draft to address every point."""


async def explainer_node(state: ExplainState) -> dict:
    paper_title = state.get("paper_title", "Research paper")
    paper_brief = state.get("paper_brief", "")
    critique = state.get("critique", "")
    previous_draft = state.get("draft", "")
    retry_count = state.get("retry_count", 0)
    messages = list(state.get("messages", []))

    revision_note = ""
    if critique and retry_count > 0:
        revision_note = f"""
Previous draft (excerpt):
{previous_draft[:2000]}

Readability feedback to address:
{critique}
"""

    user_prompt = f"""Paper title: {paper_title}

Structured paper brief:
{paper_brief}
{revision_note}

Write the full plain-language explanation now."""

    draft = await invoke_llm(EXPLAINER_SYSTEM, user_prompt, agent="explainer")
    messages = append_message(
        messages,
        "explainer",
        draft[:500] + ("..." if len(draft) > 500 else ""),
    )

    return {
        "current_agent": "explainer",
        "draft": draft,
        "messages": messages,
    }
