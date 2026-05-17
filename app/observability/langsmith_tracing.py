from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, trace

from app.config import get_settings

logger = logging.getLogger(__name__)

_configured = False

TokenUsage = dict[str, int]


def configure_langsmith() -> None:
    """Apply LangSmith / LangChain tracing env vars from settings."""
    global _configured
    settings = get_settings()

    os.environ["LANGCHAIN_TRACING_V2"] = (
        "true" if settings.langchain_tracing_v2 else "false"
    )
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    _configured = True
    if settings.langchain_tracing_v2:
        logger.info(
            "LangSmith tracing enabled (project=%s)",
            settings.langchain_project,
        )


def extract_token_usage(response: Any) -> TokenUsage:
    """Read token counts from a LangChain AIMessage or compatible response."""
    usage: dict[str, Any] = {}

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        usage = dict(usage_metadata)

    if not usage:
        meta = getattr(response, "response_metadata", None) or {}
        usage = (
            meta.get("usage")
            or meta.get("token_usage")
            or meta.get("usage_metadata")
            or {}
        )
        if not isinstance(usage, dict):
            usage = {}

    input_tokens = int(
        usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    )
    output_tokens = int(
        usage.get("output_tokens") or usage.get("completion_tokens") or 0
    )
    total_tokens = int(
        usage.get("total_tokens") or (input_tokens + output_tokens)
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def record_token_usage(agent: str, usage: TokenUsage) -> None:
    """Log token usage and attach it to the active LangSmith run."""
    logger.info(
        "[%s] token_usage input=%d output=%d total=%d",
        agent.upper(),
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
    )

    run = get_current_run_tree()
    if run is None:
        return

    metadata = dict(run.metadata or {})
    steps: list[dict[str, Any]] = list(metadata.get("token_usage_steps", []))
    steps.append({"agent": agent, **usage})
    metadata["token_usage"] = usage
    metadata["token_usage_steps"] = steps
    run.metadata = metadata


def traced_agent(
    agent_name: str,
    node_fn: Callable[..., Awaitable[dict]],
) -> Callable[..., Awaitable[dict]]:
    """Wrap an agent node with a LangSmith chain trace."""
    configure_langsmith()

    @traceable(name=agent_name, run_type="chain", tags=["agent", agent_name])
    async def wrapped(state: Any) -> dict:
        print(f"[{agent_name.upper()}] running...")
        return await node_fn(state)

    wrapped.__name__ = node_fn.__name__
    wrapped.__qualname__ = node_fn.__qualname__
    return wrapped


async def traced_llm_invoke(
    agent: str,
    invoke: Callable[[], Awaitable[Any]],
) -> tuple[str, TokenUsage]:
    """Run an LLM call inside a LangSmith LLM span and record token usage."""
    configure_langsmith()

    with trace(
        name=f"{agent}_llm",
        run_type="llm",
        inputs={"agent": agent},
        tags=[agent, "llm"],
    ) as run_tree:
        response = await invoke()
        usage = extract_token_usage(response)
        record_token_usage(agent, usage)

        content = getattr(response, "content", response)
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    text_parts.append(block)
            text = "".join(text_parts)
        else:
            text = str(content)

        if run_tree is not None:
            run_tree.outputs = {
                "content_length": len(text),
                "token_usage": usage,
            }

        return text, usage
