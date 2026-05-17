from __future__ import annotations

import json
import re

from app.config import get_settings
from app.graph.nodes._llm import append_message, invoke_llm
from app.graph.state import AgentState

CRITIC_SYSTEM = """You are the Critic agent. Score the draft report quality from 0.0 to 1.0.

Evaluate:
- Accuracy and grounding in research
- Structure and clarity
- Completeness relative to the task
- Markdown quality

Respond with ONLY valid JSON (no markdown fences):
{
  "quality_score": 0.0 to 1.0,
  "critique": "specific, actionable feedback",
  "approved": true or false
}

Set approved true only if quality_score >= 0.75."""


def _parse_critic_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


async def critic_node(state: AgentState) -> dict:
    print("[CRITIC] running...")
    settings = get_settings()
    task = state.get("task", "")
    draft = state.get("draft", "")
    research = state.get("research", "")
    retry_count = state.get("retry_count", 0)
    messages = list(state.get("messages", []))

    user_prompt = f"""Task: {task}

Research (for fact-checking):
{research[:3000]}

Draft to review:
{draft}
"""

    raw = await invoke_llm(CRITIC_SYSTEM, user_prompt)
    messages = append_message(messages, "critic", raw)

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
        "current_agent": "critic",
        "quality_score": quality_score,
        "critique": critique,
        "retry_count": new_retry,
        "messages": messages,
    }

    if approved:
        update["final_output"] = draft
        update["next_route"] = "END"

    return update
