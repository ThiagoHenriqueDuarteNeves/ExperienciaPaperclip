"""Episodic memory routes (ChromaDB) — store / retrieve / delete.

Thin HTTP layer over app.retrieval. Lives behind the shared limiter so routers
apply @limiter.limit without importing main.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.rate_limit import limiter
from app.retrieval import delete_memory, retrieve_similar, store_conversation
from app.schemas import (
    MemoryItem,
    RetrieveRequest,
    RetrieveResponse,
    StoreRequest,
    StoreResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["episodic"])


@router.post("/store", response_model=StoreResponse)
@limiter.limit("10/minute")
def store(request: Request, req: StoreRequest):
    logger.info("store: user_id=%s conversation_id=%s", req.user_id, req.conversation_id)
    memory_id = store_conversation(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        content=req.content,
        metadata=req.metadata,
        extract_knowledge_graph=req.extract_kg,
    )
    return StoreResponse(memory_id=memory_id)


@router.post("/retrieve", response_model=RetrieveResponse)
@limiter.limit("30/minute")
def retrieve(request: Request, req: RetrieveRequest):
    results = retrieve_similar(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return RetrieveResponse(memories=[MemoryItem(**m) for m in results])


@router.delete("/memories/{memory_id}", status_code=204)
def delete(memory_id: str):
    delete_memory(memory_id)
