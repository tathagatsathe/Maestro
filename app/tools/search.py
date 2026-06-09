from __future__ import annotations

import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import get_settings
from app.db.repository import DocumentRepository
from app.db.session import get_session_factory
from app.observability.metrics import SEARCH_FALLBACK_TOTAL
from app.rag.embeddings import embed_texts

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


async def _pgvector_lookup(query: str, k: int = 3) -> str | None:
    try:
        embedding = embed_texts([query])[0]
        factory = get_session_factory()
        async with factory() as session:
            repo = DocumentRepository(session)
            results = await repo.similarity_search(embedding, k=k)
        if not results:
            return None
        parts = [f"- [{title}] {content}" for content, title, _dist in results]
        return "pgvector knowledge base results:\n" + "\n".join(parts)
    except Exception:
        return None


async def vector_store_lookup(query: str, k: int = 3) -> str:
    """Semantic retrieval via pgvector, with BM25 fallback over built-in docs."""
    pg_results = await _pgvector_lookup(query, k=k)
    if pg_results:
        return pg_results

    SEARCH_FALLBACK_TOTAL.labels(source="bm25").inc()
    retriever = _get_retriever()
    retriever.k = k
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in the knowledge base."
    parts = [f"- {doc.page_content}" for doc in docs]
    return "Knowledge base results (BM25 fallback):\n" + "\n".join(parts)


async def tavily_search(query: str, max_results: int = 5) -> str:
    """Run Tavily web search; returns formatted snippets or a clear error."""
    settings = get_settings()
    if not settings.tavily_api_key:
        SEARCH_FALLBACK_TOTAL.labels(source="tavily_unconfigured").inc()
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
        SEARCH_FALLBACK_TOTAL.labels(source="tavily_error").inc()
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
        SEARCH_FALLBACK_TOTAL.labels(source="tavily_empty").inc()
        return "Tavily returned no results for this query."
    return "Web search results:\n" + "\n".join(lines)
