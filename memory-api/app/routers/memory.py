"""pgvector-backed memory routes: conversation messages, semantic facts, and the
confidence scorer.

Thin HTTP layer over app.conversation_store / app.semantic_store. Behind the
shared limiter.
"""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Request

from app.conversation_store import (
    retrieve_thread,
    search_hybrid as search_conversations,
    store_message,
)
from app.rate_limit import limiter
from app.schemas import (
    ConfidenceRequest,
    MessageSearchRequest,
    MessageStoreRequest,
    SemanticSearchRequest,
    SemanticStoreRequest,
)
from app.semantic_store import search_semantic_memories, store_semantic_memory

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memory"])


# ---------------------------------------------------------------------------
# Conversation messages
# ---------------------------------------------------------------------------


@router.post("/memory/messages", status_code=201)
@limiter.limit("60/minute")
async def store_message_endpoint(request: Request, req: MessageStoreRequest):
    logger.info("memory/messages: thread_id=%s role=%s", req.thread_id, req.role)
    message_id = await store_message(
        thread_id=req.thread_id,
        role=req.role,
        content=req.content,
        metadata=req.metadata,
    )
    return {"message_id": message_id, "status": "stored"}


@router.get("/memory/threads/{thread_id}")
async def get_thread_endpoint(thread_id: str, limit: int = 100):
    messages = await retrieve_thread(thread_id, limit=limit)
    return {"thread_id": thread_id, "messages": messages, "count": len(messages)}


@router.post("/memory/search")
@limiter.limit("60/minute")
async def search_messages_endpoint(request: Request, req: MessageSearchRequest):
    # NOTE: hybrid (RRF) search ranks by fused rank, not a cosine threshold, so
    # req.min_similarity does not apply here (passing it raised TypeError -> 500;
    # pre-existing bug surfaced by the router split). Kept on the schema for the
    # semantic-search endpoint, which does use it.
    results = await search_conversations(
        query=req.query,
        top_k=req.top_k,
        user_id=req.user_id,
    )
    return {"results": results, "query": req.query}


# ---------------------------------------------------------------------------
# Semantic facts
# ---------------------------------------------------------------------------


@router.post("/memory/semantic", status_code=201)
@limiter.limit("30/minute")
async def store_semantic_endpoint(request: Request, req: SemanticStoreRequest):
    logger.info("memory/semantic: user_id=%s key=%s", req.user_id, req.key)
    fact_id = await store_semantic_memory(
        user_id=req.user_id,
        key=req.key,
        content=req.content,
        importance=req.importance,
    )
    return {"fact_id": fact_id, "status": "stored"}


@router.post("/memory/semantic/search")
@limiter.limit("60/minute")
async def search_semantic_endpoint(request: Request, req: SemanticSearchRequest):
    results = await search_semantic_memories(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
    )
    return {"results": results, "query": req.query}


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

# Phrases that signal the assistant has no real answer to store.
_UNCERTAINTY_PHRASES = [
    "I don't know", "I don't have that information", "I'm not sure",
    "Let me check", "I need to look that up", "I don't remember",
    "não tenho essa informação", "não sei", "não lembro",
    "vou verificar", "deixa eu pesquisar", "preciso consultar minha memória",
    "realizando busca", "pesquisando nos bancos",
]

# Lazy cache: populated on first call to avoid embedding overhead at startup.
_uncertainty_embeddings: list[list[float]] | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


@router.post("/memory/confidence")
@limiter.limit("120/minute")
async def confidence_endpoint(request: Request, req: ConfidenceRequest):
    """Return whether an assistant response is confident enough to be stored.

    Embeds the response text and compares it against known uncertainty phrases
    using cosine similarity. A high similarity to uncertainty phrases means the
    response should not be persisted.
    """
    from app.embeddings import embed_texts_async

    global _uncertainty_embeddings
    if _uncertainty_embeddings is None:
        _uncertainty_embeddings = await embed_texts_async(_UNCERTAINTY_PHRASES)

    text_vec = await embed_texts_async([req.text])
    max_similarity = max(
        _cosine_similarity(text_vec[0], phrase_vec)
        for phrase_vec in _uncertainty_embeddings
    )
    confident = max_similarity < req.threshold
    return {"confident": confident, "score": round(1.0 - max_similarity, 4)}
