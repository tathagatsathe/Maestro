from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.observability.langsmith_tracing import TokenUsage, traced_llm_invoke


def get_llm() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key or None,
        temperature=0.2,
        max_tokens=4096,
    )


async def invoke_llm(system: str, user: str, *, agent: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]

    text, _usage = await traced_llm_invoke(
        agent,
        lambda: llm.ainvoke(messages),
    )
    return text


async def invoke_llm_with_usage(
    system: str, user: str, *, agent: str
) -> tuple[str, TokenUsage]:
    llm = get_llm()
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    return await traced_llm_invoke(agent, lambda: llm.ainvoke(messages))


def append_message(messages: list[dict], role: str, content: str) -> list[dict]:
    return [*messages, {"role": role, "agent": role, "content": content}]
