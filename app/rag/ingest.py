from __future__ import annotations

import re

from app.db.repository import DocumentRepository
from app.db.session import get_session_factory
from app.rag.embeddings import embed_texts


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


async def ingest_document(
    title: str,
    content: str,
    *,
    source: str | None = None,
) -> str:
    chunks = chunk_text(content)
    if not chunks:
        raise ValueError("Document content is empty after chunking")

    embeddings = embed_texts(chunks)
    factory = get_session_factory()
    async with factory() as session:
        repo = DocumentRepository(session)
        doc = await repo.create_document(
            title=title,
            source=source,
            chunks=list(zip(chunks, embeddings)),
        )
        return str(doc.id)
