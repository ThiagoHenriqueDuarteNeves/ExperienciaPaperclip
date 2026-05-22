"""Async PostgreSQL + pgvector client.

Connection pool with health checks and helper functions for vector operations.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import asyncpg
from asyncpg.pool import Pool

from app.config import settings

_pool: Pool | None = None


async def get_pool() -> Pool:
    """Return the singleton asyncpg connection pool, creating it if needed."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=settings.pgvector_host,
            port=settings.pgvector_port,
            user=settings.pgvector_user,
            password=settings.pgvector_password,
            database=settings.pgvector_database,
            min_size=settings.pgvector_min_pool,
            max_size=settings.pgvector_max_pool,
        )
    return _pool


async def close_pool() -> None:
    """Close the connection pool cleanly."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def health() -> bool:
    """Check whether the database is reachable."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False


async def connection() -> AsyncIterator[asyncpg.Connection]:
    """Async context manager yielding a connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


def _format_vector(embedding: list[float]) -> str:
    """Format a Python float list as a pgvector-compatible string literal."""
    inner = ",".join(str(v) for v in embedding)
    return f"[{inner}]"


def vector_str(embedding: list[float]) -> str:
    """Convert embedding list to pgvector literal string."""
    return _format_vector(embedding)


def vector_from_str(s: str) -> list[float]:
    """Parse a pgvector string representation back to a float list."""
    return [float(v) for v in s.strip("[]").split(",")]


# ---------------------------------------------------------------------------
# Conversation message CRUD
# ---------------------------------------------------------------------------


async def store_message(
    thread_id: str,
    role: str,
    content: str,
    embedding: list[float] | None = None,
    metadata: dict | None = None,
) -> str:
    pool = await get_pool()
    vec = _format_vector(embedding) if embedding else None
    meta = json.dumps(metadata or {})
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO conversation_messages (thread_id, role, content, embedding, metadata)
               VALUES ($1, $2, $3, $4::vector, $5::jsonb)
               RETURNING id""",
            thread_id, role, content, vec, meta,
        )
        return str(row["id"])


async def load_thread(
    thread_id: str, limit: int = 100
) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, thread_id, role, content, metadata, created_at
               FROM conversation_messages
               WHERE thread_id = $1
               ORDER BY created_at ASC
               LIMIT $2""",
            thread_id, limit,
        )
        return [dict(r) for r in rows]


async def search_similar(
    embedding: list[float],
    user_id: str | None = None,
    top_k: int = 5,
    min_similarity: float = 0.65,
) -> list[dict]:
    pool = await get_pool()
    vec = _format_vector(embedding)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, content, metadata,
                      1 - (embedding <=> $1::vector) AS similarity
               FROM conversation_messages
               WHERE ($2::text IS NULL OR metadata->>'user_id' = $2)
                 AND embedding IS NOT NULL
                 AND 1 - (embedding <=> $1::vector) >= $3
               ORDER BY similarity DESC
               LIMIT $4""",
            vec, user_id, min_similarity, top_k,
        )
        return [dict(r) for r in rows]


async def delete_message(message_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM conversation_messages WHERE id = $1", message_id,
        )


# ---------------------------------------------------------------------------
# Semantic memory CRUD
# ---------------------------------------------------------------------------


async def store_semantic(
    user_id: str,
    key: str,
    content: str,
    embedding: list[float] | None = None,
    importance: float = 0.0,
) -> str:
    pool = await get_pool()
    vec = _format_vector(embedding) if embedding else None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO semantic_memory (user_id, key, content, embedding, importance)
               VALUES ($1, $2, $3, $4::vector, $5)
               ON CONFLICT (user_id, key)
               DO UPDATE SET content = $3, embedding = $4::vector,
                             importance = $5, updated_at = NOW()
               RETURNING id""",
            user_id, key, content, vec, importance,
        )
        return str(row["id"])


async def search_semantic(
    embedding: list[float],
    user_id: str | None = None,
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> list[dict]:
    pool = await get_pool()
    vec = _format_vector(embedding)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, user_id, key, content, importance,
                      1 - (embedding <=> $1::vector) AS similarity
               FROM semantic_memory
               WHERE ($2::text IS NULL OR user_id = $2)
                 AND embedding IS NOT NULL
                 AND 1 - (embedding <=> $1::vector) >= $3
               ORDER BY importance DESC, similarity DESC
               LIMIT $4""",
            vec, user_id, min_similarity, top_k,
        )
        return [dict(r) for r in rows]
