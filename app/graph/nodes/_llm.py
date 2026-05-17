from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import get_settings


def get_llm() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key or None,
        temperature=0.2,
        max_tokens=4096,
    )


async def invoke_llm(system: str, user: str) -> str:
    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def append_message(messages: list[dict], role: str, content: str) -> list[dict]:
    return [*messages, {"role": role, "agent": role, "content": content}]
