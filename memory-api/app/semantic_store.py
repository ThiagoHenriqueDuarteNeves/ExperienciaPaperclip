"""Higher-level semantic memory API.

Wraps pgvector_client.semantic_memory operations with automatic embedding
computation. Uses upsert semantics so repeated writes to the same (user_id, key)
update the existing record.
"""

from __future__ import annotations

from app.embeddings import embed_text_async
from app.pgvector_client import (
    get_semantic_fact_by_key as _get_fact_by_key,
    get_top_semantic_facts as _get_top_semantic_facts,
    search_semantic as _search_semantic,
    search_semantic_hybrid as _search_semantic_hybrid,
    store_semantic as _store_semantic,
)


async def get_profile_facts(user_id: str | None = None, top_k: int = 5) -> list[dict]:
    """The user's durable profile: most important facts, no query embedding needed."""
    return await _get_top_semantic_facts(user_id=user_id, top_k=top_k)


async def get_identity_facts(
    user_id: str | None,
    query: str,
    top_k: int = 4,
) -> list[dict]:
    """Profile facts RELEVANT to the current message (+ the name as a fixed anchor).

    Query-aware so the prompt gets, say, the user's profession when they ask about
    work — instead of always injecting the globally most "important" (and possibly
    sensitive, irrelevant) facts into every turn.
    """
    # Hybrid (RRF) search orders by RELEVANCE to the query — unlike
    # search_semantic_memories, which orders by importance first.
    relevant = await search_semantic_memories_hybrid(
        query=query, user_id=user_id, top_k=top_k
    )
    # Always anchor the user's name, even when it isn't query-relevant.
    if user_id and not any(f.get("key") == "nome" for f in relevant):
        name = await _get_fact_by_key(user_id=user_id, key="nome")
        if name:
            relevant = [name, *relevant]
    return relevant


async def store_semantic_memory(
    user_id: str,
    key: str,
    content: str,
    importance: float = 0.0,
) -> str:
    """Store a semantic memory fact. Embedding is computed automatically.

    Uses upsert: if (user_id, key) already exists, the content and embedding
    are updated in-place (idempotent).
    """
    embedding = await embed_text_async(content)
    return await _store_semantic(
        user_id=user_id,
        key=key,
        content=content,
        embedding=embedding,
        importance=importance,
    )


async def search_semantic_memories(
    query: str,
    user_id: str | None = None,
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> list[dict]:
    """Semantic search over stored facts. Embedding is computed automatically."""
    embedding = await embed_text_async(query, is_query=True)
    return await _search_semantic(
        embedding=embedding,
        user_id=user_id,
        top_k=top_k,
        min_similarity=min_similarity,
    )


async def search_semantic_memories_hybrid(
    query: str,
    user_id: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Hybrid dense + BM25 search over semantic facts with RRF fusion."""
    from app.config import settings
    embedding = await embed_text_async(query, is_query=True)
    return await _search_semantic_hybrid(
        embedding=embedding,
        query_text=query,
        user_id=user_id,
        top_k=top_k,
        rrf_k=settings.rrf_k,
        fts_language=settings.fts_language,
    )
