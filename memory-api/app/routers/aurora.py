"""Arquivo Aurora — affective-memory routes.

Owns the Aurora optional-import block: if aurora_store / aurora_extractor (or
their deps) are unavailable, the functions degrade to safe stubs so the rest of
the API keeps working. The chat pipeline (main.py) reuses `extract_aurora` and
`store_aurora_memory` from here to persist affective records per turn.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.rate_limit import limiter
from app.schemas import AuroraSearchRequest, AuroraStoreRequest

try:
    from app.aurora_store import (
        pull_saudade_memories,
        search_aurora_memories,
        store_aurora_memory,
    )
    from app.aurora_extractor import extract_aurora

    _AURORA_AVAILABLE = True
except ImportError:
    _AURORA_AVAILABLE = False

    async def store_aurora_memory(*args, **kwargs) -> str:
        raise RuntimeError("Arquivo Aurora not available")

    async def search_aurora_memories(*args, **kwargs) -> list:
        return []

    async def pull_saudade_memories(*args, **kwargs) -> list:
        return []

    def extract_aurora(*args, **kwargs) -> dict | None:
        return None


logger = logging.getLogger(__name__)
router = APIRouter(tags=["aurora"])


@router.post("/memory/aurora", status_code=201)
@limiter.limit("30/minute")
async def store_aurora_endpoint(request: Request, req: AuroraStoreRequest):
    logger.info("memory/aurora: user_id=%s tipo=%s", req.user_id, req.tipo)
    aurora_id = await store_aurora_memory(
        user_id=req.user_id,
        tipo=req.tipo,
        disparo=req.disparo,
        fato=req.fato,
        tom_do_usuario=req.tom_do_usuario,
        minha_reacao_emocional=req.minha_reacao_emocional,
        reacao_simulada=req.reacao_simulada,
        importancia=req.importancia,
        ressonancia=req.ressonancia,
        intimidade=req.intimidade,
        saudade=req.saudade,
        tags=req.tags,
        conexoes=req.conexoes,
        contexto_extra=req.contexto_extra,
        bilhete_interno=req.bilhete_interno,
        data=req.data,
    )
    return {"aurora_id": aurora_id, "status": "stored"}


@router.post("/memory/aurora/search")
@limiter.limit("60/minute")
async def search_aurora_endpoint(request: Request, req: AuroraSearchRequest):
    results = await search_aurora_memories(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return {"results": results, "query": req.query}


@router.get("/memory/aurora/saudade")
@limiter.limit("60/minute")
async def aurora_saudade_endpoint(request: Request, user_id: str | None = None, limit: int = 1):
    results = await pull_saudade_memories(user_id=user_id, limit=limit)
    return {"results": results, "count": len(results)}
