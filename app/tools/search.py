from __future__ import annotations

import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import get_settings

_KNOWLEDGE_BASE: list[Document] = [
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-agent workflows "
            "as graphs with nodes, edges, and shared typed state."
        ),
        metadata={"source": "kb", "topic": "langgraph"},
    ),
    Document(
        page_content=(
            "Multi-agent systems route tasks between specialized agents such as "
            "researchers, writers, and critics coordinated by a supervisor."
        ),
        metadata={"source": "kb", "topic": "multi-agent"},
    ),
    Document(
        page_content=(
            "Retrieval-augmented generation combines vector search over documents "
            "with LLM synthesis for grounded answers."
        ),
        metadata={"source": "kb", "topic": "rag"},
    ),
]

_retriever: BM25Retriever | None = None


def _get_retriever() -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever.from_documents(_KNOWLEDGE_BASE, k=3)
    return _retriever


async def vector_store_lookup(query: str, k: int = 3) -> str:
    """Keyword/BM25 retrieval over a small in-process knowledge base."""
    retriever = _get_retriever()
    retriever.k = k
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in the knowledge base."
    parts = [f"- {doc.page_content}" for doc in docs]
    return "Knowledge base results:\n" + "\n".join(parts)


async def tavily_search(query: str, max_results: int = 5) -> str:
    """Run Tavily web search; returns formatted snippets or a clear error."""
    settings = get_settings()
    if not settings.tavily_api_key:
        return (
            "Tavily search unavailable (TAVILY_API_KEY not set). "
            "Proceed using knowledge base and model reasoning only."
        )

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        response: dict[str, Any] = await client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
        )
    except Exception as exc:
        return f"Tavily search failed: {exc}"

    lines: list[str] = []
    answer = response.get("answer")
    if answer:
        lines.append(f"Summary: {answer}")

    for item in response.get("results", []):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")
        snippet = re.sub(r"\s+", " ", content).strip()[:400]
        lines.append(f"- {title} ({url}): {snippet}")

    if not lines:
        return "Tavily returned no results for this query."
    return "Web search results:\n" + "\n".join(lines)
