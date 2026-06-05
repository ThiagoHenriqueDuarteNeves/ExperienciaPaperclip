"""Memory inspector — a local debug tool.

Lets you (1) preview the EXACT recall + system prompt that the chat pipeline
would send to the LLM for a given prompt, and (2) freely browse what is stored
in every memory bank (pgvector conversations/semantic/aurora, ChromaDB episodic,
Neo4j graph).

Gated by settings.inspect_enabled (MEMORY_INSPECT_ENABLED=true) so it is never
reachable through the public zrok share unless explicitly turned on locally.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.chat_pipeline import recall_and_build_system
from app.config import settings
from app.pgvector_client import _serialize_row, get_pool

router = APIRouter(prefix="/inspect", tags=["inspect"])

_PAGE = os.path.join(os.path.dirname(__file__), "static", "inspect.html")


def _guard() -> None:
    # getattr default keeps this safe even if the `inspect_enabled` config field
    # isn't present yet (e.g. committed separately from this feature).
    if not getattr(settings, "inspect_enabled", False):
        # 404 (not 403) so the feature's existence isn't advertised when off.
        raise HTTPException(status_code=404, detail="Not Found")


# ---------------------------------------------------------------------------
# Web page
# ---------------------------------------------------------------------------


@router.get("", include_in_schema=False)
def inspect_page():
    _guard()
    return FileResponse(_PAGE)


# ---------------------------------------------------------------------------
# Overview — counts per bank + user list
# ---------------------------------------------------------------------------


@router.get("/overview")
async def overview():
    _guard()
    pool = await get_pool()
    async with pool.acquire() as conn:
        counts = await conn.fetchrow(
            """SELECT
                 (SELECT count(*) FROM conversation_messages) AS conversations,
                 (SELECT count(*) FROM semantic_memory)       AS semantic,
                 (SELECT count(*) FROM arquivo_aurora)        AS aurora,
                 (SELECT count(*) FROM user_profiles)         AS profiles"""
        )
        profiles = await conn.fetch(
            "SELECT user_id, display_name FROM user_profiles ORDER BY user_id"
        )

    banks = {
        "conversations": counts["conversations"],
        "semantic": counts["semantic"],
        "aurora": counts["aurora"],
        "profiles": counts["profiles"],
    }
    # ChromaDB + Neo4j are best-effort (may be down).
    try:
        from app.chroma_client import get_or_create_collection

        banks["episodic"] = get_or_create_collection().count()
    except Exception:
        banks["episodic"] = None
    try:
        from app.neo4j_client import get_driver

        with get_driver().session() as s:
            banks["graph_entities"] = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            banks["graph_edges"] = s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
    except Exception:
        banks["graph_entities"] = banks["graph_edges"] = None

    return {
        "banks": banks,
        "users": [dict(p) for p in profiles],
    }


# ---------------------------------------------------------------------------
# Recall preview — the heart of the tool
# ---------------------------------------------------------------------------


@router.post("/recall")
async def recall_preview(body: dict):
    """Given {prompt, user_id}, return the exact recall + system payload that the
    chat pipeline would send to the LLM — WITHOUT calling the LLM."""
    _guard()
    prompt = (body.get("prompt") or "").strip()
    user_id = body.get("user_id") or None
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    layers, memory_context, system = await recall_and_build_system(user_id, prompt)

    # This mirrors the chat request body (minus history, which comes from the client).
    llm_payload = {
        "model": settings.claude_model,
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    return {
        "user_id": user_id,
        "prompt": prompt,
        "layers": layers,
        "memory_context": memory_context,
        "system": system,
        "llm_payload": llm_payload,
        "counts": {k: len(v) for k, v in layers.items()},
    }


# ---------------------------------------------------------------------------
# Free browsing of each bank
# ---------------------------------------------------------------------------


@router.get("/bank/conversations")
async def bank_conversations(
    user_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    _guard()
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, thread_id::text, role, content, metadata, created_at
               FROM conversation_messages
               WHERE ($1::text IS NULL OR metadata->>'user_id' = $1)
                 AND ($2::text IS NULL OR content ILIKE '%'||$2||'%')
               ORDER BY created_at DESC
               LIMIT $3""",
            user_id, q, limit,
        )
    return {"rows": [_serialize_row(r) for r in rows], "count": len(rows)}


@router.get("/bank/semantic")
async def bank_semantic(
    user_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    _guard()
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, user_id, key, content, importance, created_at, updated_at
               FROM semantic_memory
               WHERE ($1::text IS NULL OR user_id = $1)
                 AND ($2::text IS NULL OR content ILIKE '%'||$2||'%' OR key ILIKE '%'||$2||'%')
               ORDER BY updated_at DESC
               LIMIT $3""",
            user_id, q, limit,
        )
    return {"rows": [_serialize_row(r) for r in rows], "count": len(rows)}


@router.get("/bank/aurora")
async def bank_aurora(
    user_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    _guard()
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, user_id, data::text AS data, tipo, disparo, tom_do_usuario,
                      importancia, ressonancia, intimidade, saudade, tags, fato,
                      minha_reacao_emocional, reacao_simulada, contexto_extra,
                      bilhete_interno, created_at
               FROM arquivo_aurora
               WHERE ($1::text IS NULL OR user_id = $1)
                 AND ($2::text IS NULL OR fato ILIKE '%'||$2||'%'
                      OR contexto_extra ILIKE '%'||$2||'%')
               ORDER BY created_at DESC
               LIMIT $3""",
            user_id, q, limit,
        )
    return {"rows": [_serialize_row(r) for r in rows], "count": len(rows)}


@router.get("/bank/episodic")
async def bank_episodic(
    user_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    _guard()
    try:
        from app.chroma_client import get_or_create_collection

        collection = get_or_create_collection()
        where = {"user_id": user_id} if user_id else None
        where_doc = {"$contains": q} if q else None
        res = collection.get(
            where=where,
            where_document=where_doc,
            limit=limit,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ChromaDB unavailable: {exc}")

    rows = [
        {"id": i, "content": d, "metadata": m}
        for i, d, m in zip(
            res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
        )
    ]
    return {"rows": rows, "count": len(rows)}


@router.get("/bank/graph")
async def bank_graph(
    q: str | None = Query(default=None),
    entity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    _guard()
    try:
        from app.neo4j_client import get_entity_graph, search_entities
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}")

    # Drill into one entity's neighborhood.
    if entity:
        graph = get_entity_graph(entity, depth=2)
        if graph is None:
            raise HTTPException(status_code=404, detail=f"entity '{entity}' not found")
        return graph

    entities = search_entities(query=q or "", limit=limit)
    return {"rows": entities, "count": len(entities)}
