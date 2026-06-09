from __future__ import annotations

import json
import re

from app.config import get_settings
from app.graph.explain_state import ExplainState
from app.graph.nodes._llm import append_message, invoke_llm

READABILITY_CRITIC_SYSTEM = """You are the Readability Critic agent.
Score how understandable a research paper explanation is for a NON-EXPERT reader (0.0 to 1.0).

Evaluate:
- Plain language: minimal unexplained jargon
- Assumed background: does not require field expertise
- Clarity: logical structure and easy-to-follow sentences
- Completeness: covers key findings, methods, and limitations
- Glossary quality: technical terms are defined simply

Respond with ONLY valid JSON (no markdown fences):
{
  "quality_score": 0.0 to 1.0,
  "critique": "specific, actionable feedback for simplifying further",
  "approved": true or false
}

Set approved true only if quality_score >= 0.75."""


def _parse_critic_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


async def readability_critic_node(state: ExplainState) -> dict:
    settings = get_settings()
    paper_brief = state.get("paper_brief", "")
    draft = state.get("draft", "")
    retry_count = state.get("retry_count", 0)
    messages = list(state.get("messages", []))

    user_prompt = f"""Paper brief (for accuracy check):
{paper_brief[:3000]}

Explanation draft to review:
{draft}
"""

    raw = await invoke_llm(
        READABILITY_CRITIC_SYSTEM, user_prompt, agent="readability_critic"
    )
    messages = append_message(messages, "readability_critic", raw)

    try:
        parsed = _parse_critic_json(raw)
        quality_score = float(parsed.get("quality_score", 0.0))
        critique = str(parsed.get("critique", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        quality_score = 0.5
        critique = raw

    quality_score = max(0.0, min(1.0, quality_score))
    approved = quality_score >= settings.quality_threshold

    new_retry = retry_count
    if not approved:
        new_retry = retry_count + 1

    update: dict = {
        "current_agent": "readability_critic",
        "quality_score": quality_score,
        "critique": critique,
        "retry_count": new_retry,
        "messages": messages,
    }

    if approved:
        update["final_output"] = draft

    return update
