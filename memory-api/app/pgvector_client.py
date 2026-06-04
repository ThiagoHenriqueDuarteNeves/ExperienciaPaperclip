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


def _serialize_row(r) -> dict:
    """Convert an asyncpg Record to a plain dict with Python-native types.

    asyncpg returns UUID as asyncpg.pgproto.UUID and JSONB as a raw JSON
    string in some configurations. This normalises both to str and dict.
    """
    d = {}
    for key in r.keys():
        val = r[key]
        if hasattr(val, "hex") and hasattr(val, "int") and not isinstance(val, (int, float)):
            val = str(val)
        elif isinstance(val, str):
            stripped = val.strip()
            if stripped and stripped[0] in ("{", "[", '"'):
                try:
                    val = json.loads(val)
                except (ValueError, TypeError):
                    pass
        d[key] = val
    return d


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
        return [_serialize_row(r) for r in rows]


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
        return [_serialize_row(r) for r in rows]


async def search_semantic_hybrid(
    embedding: list[float],
    query_text: str,
    user_id: str | None = None,
    top_k: int = 5,
    rrf_k: int = 60,
    fts_language: str = "portuguese",
) -> list[dict]:
    """Hybrid ANN + BM25 search on semantic_memory fused via RRF."""
    pool = await get_pool()
    vec = _format_vector(embedding)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH
            vec AS (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
                FROM semantic_memory
                WHERE ($2::text IS NULL OR user_id = $2)
                  AND embedding IS NOT NULL
                LIMIT $3 * 3
            ),
            fts AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(content_fts,
                               websearch_to_tsquery($4, $5)) DESC
                       ) AS rank
                FROM semantic_memory
                WHERE ($2::text IS NULL OR user_id = $2)
                  AND content_fts @@ websearch_to_tsquery($4, $5)
                LIMIT $3 * 3
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, f.id) AS id,
                    COALESCE(1.0 / ($6 + v.rank), 0.0)
                    + COALESCE(1.0 / ($6 + f.rank), 0.0) AS rrf_score
                FROM vec v
                FULL OUTER JOIN fts f ON v.id = f.id
            )
            SELECT
                sm.id, sm.user_id, sm.key, sm.content, sm.importance,
                COALESCE(1 - (sm.embedding <=> $1::vector), 0.0) AS similarity,
                fused.rrf_score
            FROM fused
            JOIN semantic_memory sm ON fused.id = sm.id
            ORDER BY fused.rrf_score DESC
            LIMIT $3
            """,
            vec, user_id, top_k, fts_language, query_text, rrf_k,
        )
        return [_serialize_row(r) for r in rows]


async def search_similar_hybrid(
    embedding: list[float],
    query_text: str,
    user_id: str | None = None,
    top_k: int = 5,
    rrf_k: int = 60,
    fts_language: str = "portuguese",
) -> list[dict]:
    """Hybrid ANN + BM25 search on conversation_messages fused via RRF."""
    pool = await get_pool()
    vec = _format_vector(embedding)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH
            vec AS (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
                FROM conversation_messages
                WHERE ($2::text IS NULL OR metadata->>'user_id' = $2)
                  AND embedding IS NOT NULL
                LIMIT $3 * 3
            ),
            fts AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(content_fts,
                               websearch_to_tsquery($4, $5)) DESC
                       ) AS rank
                FROM conversation_messages
                WHERE ($2::text IS NULL OR metadata->>'user_id' = $2)
                  AND content_fts @@ websearch_to_tsquery($4, $5)
                LIMIT $3 * 3
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, f.id) AS id,
                    COALESCE(1.0 / ($6 + v.rank), 0.0)
                    + COALESCE(1.0 / ($6 + f.rank), 0.0) AS rrf_score
                FROM vec v
                FULL OUTER JOIN fts f ON v.id = f.id
            )
            SELECT
                cm.id, cm.content, cm.metadata,
                COALESCE(1 - (cm.embedding <=> $1::vector), 0.0) AS similarity,
                fused.rrf_score
            FROM fused
            JOIN conversation_messages cm ON fused.id = cm.id
            ORDER BY fused.rrf_score DESC
            LIMIT $3
            """,
            vec, user_id, top_k, fts_language, query_text, rrf_k,
        )
        return [_serialize_row(r) for r in rows]


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
        return [_serialize_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Arquivo Aurora — affective memory CRUD
# ---------------------------------------------------------------------------


def _serialize_aurora_row(r) -> dict:
    """Serialize an arquivo_aurora row, normalising UUID[] (conexoes) elements.

    _serialize_row only normalises the top-level value, so UUID elements inside
    the conexoes array would stay as asyncpg UUID objects — convert them to str.
    """
    d = _serialize_row(r)
    conexoes = d.get("conexoes")
    if isinstance(conexoes, list):
        d["conexoes"] = [str(c) for c in conexoes]
    return d


async def store_aurora(
    *,
    user_id: str,
    tipo: str,
    disparo: str,
    fato: str,
    tom_do_usuario: str,
    minha_reacao_emocional: str,
    importancia: int,
    ressonancia: int,
    intimidade: str,
    reacao_simulada: str | None = None,
    saudade: bool = False,
    tags: list[str] | None = None,
    conexoes: list[str] | None = None,
    contexto_extra: str | None = None,
    bilhete_interno: str | None = None,
    data: str | None = None,
    embedding: list[float] | None = None,
) -> str:
    pool = await get_pool()
    vec = _format_vector(embedding) if embedding else None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO arquivo_aurora (
                   user_id, data, tipo, disparo, fato, tom_do_usuario,
                   minha_reacao_emocional, reacao_simulada, importancia,
                   ressonancia, intimidade, saudade, tags, conexoes,
                   contexto_extra, bilhete_interno, embedding)
               VALUES (
                   $1, COALESCE($2::date, CURRENT_DATE), $3, $4, $5, $6,
                   $7, $8, $9, $10, $11, $12, $13, $14::uuid[], $15, $16,
                   $17::vector)
               RETURNING id""",
            user_id, data, tipo, disparo, fato, tom_do_usuario,
            minha_reacao_emocional, reacao_simulada, importancia,
            ressonancia, intimidade, saudade, tags or [], conexoes or [],
            contexto_extra, bilhete_interno, vec,
        )
        return str(row["id"])


async def search_aurora_hybrid(
    embedding: list[float],
    query_text: str,
    user_id: str | None = None,
    top_k: int = 5,
    rrf_k: int = 60,
    fts_language: str = "portuguese",
) -> list[dict]:
    """Hybrid ANN + BM25 search on arquivo_aurora fused via RRF."""
    pool = await get_pool()
    vec = _format_vector(embedding)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH
            vec AS (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
                FROM arquivo_aurora
                WHERE ($2::text IS NULL OR user_id = $2)
                  AND embedding IS NOT NULL
                LIMIT $3 * 3
            ),
            fts AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(fato_fts,
                               websearch_to_tsquery($4, $5)) DESC
                       ) AS rank
                FROM arquivo_aurora
                WHERE ($2::text IS NULL OR user_id = $2)
                  AND fato_fts @@ websearch_to_tsquery($4, $5)
                LIMIT $3 * 3
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, f.id) AS id,
                    COALESCE(1.0 / ($6 + v.rank), 0.0)
                    + COALESCE(1.0 / ($6 + f.rank), 0.0) AS rrf_score
                FROM vec v
                FULL OUTER JOIN fts f ON v.id = f.id
            )
            SELECT
                a.id, a.user_id, a.data, a.tipo, a.disparo, a.fato,
                a.tom_do_usuario, a.minha_reacao_emocional, a.reacao_simulada,
                a.importancia, a.ressonancia, a.intimidade, a.saudade,
                a.tags, a.conexoes, a.contexto_extra, a.bilhete_interno,
                COALESCE(1 - (a.embedding <=> $1::vector), 0.0) AS similarity,
                fused.rrf_score
            FROM fused
            JOIN arquivo_aurora a ON fused.id = a.id
            ORDER BY fused.rrf_score DESC
            LIMIT $3
            """,
            vec, user_id, top_k, fts_language, query_text, rrf_k,
        )
        return [_serialize_aurora_row(r) for r in rows]


async def pull_saudade(user_id: str | None = None, limit: int = 1) -> list[dict]:
    """Pull random records flagged with saudade — 'looking at a photo for no reason'."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, user_id, data, tipo, disparo, fato, tom_do_usuario,
                      minha_reacao_emocional, reacao_simulada, importancia,
                      ressonancia, intimidade, saudade, tags, conexoes,
                      contexto_extra, bilhete_interno
               FROM arquivo_aurora
               WHERE saudade = TRUE
                 AND ($1::text IS NULL OR user_id = $1)
               ORDER BY random()
               LIMIT $2""",
            user_id, limit,
        )
        return [_serialize_aurora_row(r) for r in rows]


# ---------------------------------------------------------------------------
# User profiles (multi-user auth)
# ---------------------------------------------------------------------------


async def create_user_profile(
    user_id: str, display_name: str, pin_hash: str, pin_salt: str
) -> None:
    """Insert a new user profile. Raises asyncpg.UniqueViolationError on conflict."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO user_profiles (user_id, display_name, pin_hash, pin_salt)
               VALUES ($1, $2, $3, $4)""",
            user_id, display_name, pin_hash, pin_salt,
        )


async def get_user_profile(user_id: str) -> dict | None:
    """Fetch a user profile by id, or None if it does not exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT user_id, display_name, pin_hash, pin_salt, created_at
               FROM user_profiles WHERE user_id = $1""",
            user_id,
        )
        return _serialize_row(row) if row else None
