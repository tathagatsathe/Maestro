from __future__ import annotations

from app.graph.nodes._llm import append_message, invoke_llm
from app.graph.state import AgentState
from app.tools.search import tavily_search, vector_store_lookup

RESEARCHER_SYSTEM = """You are the Researcher agent.
Synthesize web search and knowledge-base findings into a structured research brief.
Include:
- Key facts and figures (cite sources when available)
- Open questions or gaps
- Recommended angles for the report

Be concise but thorough. Use markdown headings where helpful."""


async def researcher_node(state: AgentState) -> dict:
    task = state.get("task", "")
    plan = state.get("plan", [])
    messages = list(state.get("messages", []))

    web_results = await tavily_search(task)
    kb_results = await vector_store_lookup(task)

    user_prompt = f"""Task: {task}

Supervisor plan:
{chr(10).join(f"- {step}" for step in plan) or "- (no plan yet)"}

--- Web search ---
{web_results}

--- Vector store ---
{kb_results}

Produce a research brief the Writer can use."""

    research = await invoke_llm(RESEARCHER_SYSTEM, user_prompt, agent="researcher")
    messages = append_message(messages, "researcher", research)

    return {
        "current_agent": "researcher",
        "research": research,
        "messages": messages,
    }
