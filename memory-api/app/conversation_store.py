"""Higher-level conversation persistence API.

Wraps pgvector_client with automatic embedding computation so callers work
with plain text rather than raw vectors.
"""

from __future__ import annotations

from app.embeddings import embed_text_async
from app.pgvector_client import (
    load_thread as _load_thread,
    search_similar as _search_similar,
    store_message as _store_message,
)


async def store_message(
    thread_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> str:
    """Store a conversation message. Embedding is computed automatically."""
    embedding = await embed_text_async(content)
    return await _store_message(
        thread_id=thread_id,
        role=role,
        content=content,
        embedding=embedding,
        metadata=metadata,
    )


async def retrieve_thread(thread_id: str, limit: int = 100) -> list[dict]:
    """Load messages for a thread in chronological order."""
    return await _load_thread(thread_id, limit=limit)


async def search_similar(
    query: str,
    top_k: int = 5,
    user_id: str | None = None,
    min_similarity: float = 0.65,
) -> list[dict]:
    """ANN similarity search. Embedding is computed automatically from query text."""
    embedding = await embed_text_async(query)
    return await _search_similar(
        embedding=embedding,
        user_id=user_id,
        top_k=top_k,
        min_similarity=min_similarity,
    )
