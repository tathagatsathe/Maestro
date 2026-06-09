from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        settings = get_settings()
        _embedder = TextEmbedding(model_name=settings.embedding_model)
        logger.info("Loaded embedding model %s", settings.embedding_model)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    return [list(vec) for vec in embedder.embed(texts)]
