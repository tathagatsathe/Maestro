from __future__ import annotations

import json
import re

from app.config import get_settings
from app.graph.nodes._llm import append_message, invoke_llm
from app.graph.state import AgentState

SUPERVISOR_SYSTEM = """You are the Supervisor in a multi-agent research pipeline.
Your job is to:
1. Break the user task into a clear step-by-step plan (3-6 steps).
2. Decide which worker should run next: researcher, writer, or critic.
3. When quality_score >= 0.75, set route to END and summarize final_output from the draft.

Routing rules:
- First pass (no research yet): route to researcher.
- After research, before draft: route to writer.
- After draft exists and quality unknown or low: route to critic (unless already scored high).
- If quality_score >= 0.75: route to END and copy draft to final_output.
- If retry_count >= 3 and quality still low: route to END with best available draft as final_output.

Respond with ONLY valid JSON (no markdown fences):
{
  "plan": ["step 1", "step 2", ...],
  "next_route": "researcher" | "writer" | "critic" | "END",
  "final_output": "string or empty if not ending",
  "reasoning": "brief explanation"
}"""


def _parse_supervisor_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


async def supervisor_node(state: AgentState) -> dict:
    settings = get_settings()
    task = state.get("task", "")
    quality = state.get("quality_score", 0.0)
    retry_count = state.get("retry_count", 0)
    draft = state.get("draft", "")
    research = state.get("research", "")
    messages = list(state.get("messages", []))

    if quality >= settings.quality_threshold and draft:
        return {
            "current_agent": "supervisor",
            "next_route": "END",
            "final_output": draft,
            "messages": append_message(
                messages,
                "supervisor",
                "Quality threshold met; approving final output.",
            ),
        }

    if retry_count >= settings.max_retries and draft:
        return {
            "current_agent": "supervisor",
            "next_route": "END",
            "final_output": draft,
            "messages": append_message(
                messages,
                "supervisor",
                f"Max retries ({settings.max_retries}) reached; publishing best draft.",
            ),
        }

    user_prompt = f"""Task: {task}

Current state:
- plan: {state.get("plan", [])}
- research length: {len(research)} chars
- draft length: {len(draft)} chars
- quality_score: {quality}
- retry_count: {retry_count}
- critique: {state.get("critique", "")[:500]}

Decide the next route and update the plan if needed."""

    raw = await invoke_llm(SUPERVISOR_SYSTEM, user_prompt, agent="supervisor")
    messages = append_message(messages, "supervisor", raw)

    try:
        parsed = _parse_supervisor_json(raw)
    except json.JSONDecodeError:
        next_route = "researcher"
        if research and not draft:
            next_route = "writer"
        elif draft:
            next_route = "critic"
        parsed = {
            "plan": state.get("plan") or [f"Research: {task}", "Write report", "Review"],
            "next_route": next_route,
            "final_output": "",
        }

    plan = parsed.get("plan") or state.get("plan") or []
    next_route = str(parsed.get("next_route", "researcher")).lower()
    if next_route not in {"researcher", "writer", "critic", "end"}:
        next_route = "researcher"

    update: dict = {
        "current_agent": "supervisor",
        "plan": plan,
        "next_route": next_route.upper() if next_route == "end" else next_route,
        "messages": messages,
    }

    if next_route == "end":
        update["next_route"] = "END"
        update["final_output"] = parsed.get("final_output") or draft

    return update
