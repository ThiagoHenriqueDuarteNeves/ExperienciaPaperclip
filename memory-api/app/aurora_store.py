"""Higher-level Arquivo Aurora (affective memory) API.

Wraps pgvector_client.arquivo_aurora operations with automatic embedding
computation. The embedding is computed from the `fato` field — the objective
description of what happened — so search matches on the event itself, while the
emotional layer (tom, reação, bilhete) travels alongside it.
"""

from __future__ import annotations

from app.embeddings import embed_text_async
from app.pgvector_client import (
    pull_saudade as _pull_saudade,
    search_aurora_hybrid as _search_aurora_hybrid,
    store_aurora as _store_aurora,
)


async def store_aurora_memory(
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
) -> str:
    """Store an affective memory record. Embedding is computed from `fato`."""
    embedding = await embed_text_async(fato)
    return await _store_aurora(
        user_id=user_id,
        tipo=tipo,
        disparo=disparo,
        fato=fato,
        tom_do_usuario=tom_do_usuario,
        minha_reacao_emocional=minha_reacao_emocional,
        importancia=importancia,
        ressonancia=ressonancia,
        intimidade=intimidade,
        reacao_simulada=reacao_simulada,
        saudade=saudade,
        tags=tags,
        conexoes=conexoes,
        contexto_extra=contexto_extra,
        bilhete_interno=bilhete_interno,
        data=data,
        embedding=embedding,
    )


async def search_aurora_memories(
    query: str,
    user_id: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Hybrid dense + BM25 search over affective records with RRF fusion."""
    from app.config import settings

    embedding = await embed_text_async(query, is_query=True)
    return await _search_aurora_hybrid(
        embedding=embedding,
        query_text=query,
        user_id=user_id,
        top_k=top_k,
        rrf_k=settings.rrf_k,
        fts_language=settings.fts_language,
    )


async def pull_saudade_memories(
    user_id: str | None = None,
    limit: int = 1,
) -> list[dict]:
    """Pull random records flagged with saudade."""
    return await _pull_saudade(user_id=user_id, limit=limit)
