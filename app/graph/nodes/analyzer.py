from __future__ import annotations

from app.config import get_settings
from app.graph.explain_state import ExplainState
from app.graph.nodes._llm import append_message, invoke_llm

ANALYZER_SYSTEM = """You are the Analyzer agent for research papers.
Read the paper text and produce a structured brief for a layperson explainer.

Include these sections:
- Problem: What question or problem does the paper address?
- Background: Minimum context a non-expert needs
- Methods: How the researchers approached the problem (plain language)
- Results: Key findings with numbers or comparisons when available
- Claims: What the authors conclude (distinguish fact from speculation)
- Limitations: What the study does not prove

Be accurate and grounded in the paper. Do not invent results."""

ANALYZER_MAP_SYSTEM = """You are analyzing one section of a longer research paper.
Extract only the key points from this section for a layperson brief.
Focus on: problem hints, methods, results, and claims present in this section."""

ANALYZER_REDUCE_SYSTEM = """You are merging partial analyses of a research paper
into one structured brief.

Combine the section summaries into a single coherent brief with these sections:
- Problem
- Background
- Methods
- Results
- Claims
- Limitations

Remove duplicates and resolve contradictions. Stay faithful to the source material."""


def _chunk_paper_text(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size
    return chunks


async def _analyze_chunk(chunk: str, *, part: int, total: int) -> str:
    user_prompt = f"Section {part} of {total}:\n\n{chunk}"
    return await invoke_llm(ANALYZER_MAP_SYSTEM, user_prompt, agent="analyzer")


async def _analyze_paper(text: str) -> str:
    settings = get_settings()
    chunks = _chunk_paper_text(text, settings.paper_chunk_size)

    if len(chunks) == 1:
        return await invoke_llm(
            ANALYZER_SYSTEM,
            f"Paper text:\n\n{chunks[0]}",
            agent="analyzer",
        )

    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        partial = await _analyze_chunk(chunk, part=index, total=len(chunks))
        partials.append(partial)

    combined = "\n\n---\n\n".join(partials)
    return await invoke_llm(
        ANALYZER_REDUCE_SYSTEM,
        f"Section summaries to merge:\n\n{combined}",
        agent="analyzer",
    )


async def analyzer_node(state: ExplainState) -> dict:
    paper_text = state.get("paper_text", "")
    messages = list(state.get("messages", []))

    paper_brief = await _analyze_paper(paper_text)
    messages = append_message(
        messages,
        "analyzer",
        paper_brief[:500] + ("..." if len(paper_brief) > 500 else ""),
    )

    return {
        "current_agent": "analyzer",
        "paper_brief": paper_brief,
        "messages": messages,
    }
