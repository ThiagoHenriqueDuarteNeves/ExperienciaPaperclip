import asyncio
import json
import logging
import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.chat_pipeline import build_system, recall_and_build_system
from app.chroma_client import health as chroma_health
from app.llm_client import stream_text
from app.logging_config import configure_logging
from app.retrieval import delete_memory, retrieve_similar, store_conversation

# Phase 1: pgvector-backed memory
from app.conversation_store import (
    retrieve_thread,
    search_hybrid as search_conversations,
    store_message,
)
from app.pgvector_client import health as pgvector_health
from app.semantic_store import (
    search_semantic_memories,
    store_semantic_memory,
)

# Phase 3: Arquivo Aurora — affective memory
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

# Phase 4: multi-user auth (auth routes live in app.routers.auth; chat needs this one)
from app.auth import user_from_authorization

configure_logging()
logger = logging.getLogger(__name__)

# Phase 1: LangGraph agent
try:
    from app.langgraph_agent import run_agent as run_langgraph_agent

    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False

    async def run_langgraph_agent(*args, **kwargs) -> dict:
        raise RuntimeError("Agent not available — check langgraph dependencies")


async def run_agent(user_message: str, thread_id: str | None = None, user_id: str | None = None) -> dict:
    """Run the LangGraph agent for a single turn."""
    import uuid

    thread_id = thread_id or str(uuid.uuid4())
    user_id = user_id or "default"

    result = await run_langgraph_agent(
        thread_id=thread_id,
        user_id=user_id,
        user_message=user_message,
    )
    return result


async def load_thread_history(thread_id: str) -> list:
    """Load full conversation history for a thread."""
    return await retrieve_thread(thread_id)

try:
    from app.neo4j_client import health as neo4j_health, init_schema
    from app.kg_retrieval import (
        augment_with_graph_context,
        get_entity_graph,
        query_graph,
        search_graph,
    )
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

    def neo4j_health() -> bool:
        return False

    def init_schema() -> None:
        pass

    def search_graph(*args, **kwargs) -> list:
        return []

    def query_graph(*args, **kwargs) -> dict:
        return {"entity": None}

    def get_entity_graph(*args, **kwargs) -> dict | None:
        return None

    def augment_with_graph_context(*args, **kwargs) -> list:
        return []

# Letta routes + their optional-import handling live in app.routers.letta.

# Conversation loop is not yet fully implemented — routes return 501.
_CONVERSATION_LOOP_AVAILABLE = False


from app.rate_limit import limiter  # noqa: E402

app = FastAPI(title="Episodic, Semantic & Procedural Memory API", version="0.4.0-phase1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Routers split by domain (Fase 2.2). Inspector self-gates on settings.inspect_enabled.
from app.inspect_api import router as inspect_router  # noqa: E402
from app.routers.auth import router as auth_router  # noqa: E402
from app.routers.letta import letta_health, router as letta_router  # noqa: E402

app.include_router(inspect_router)
app.include_router(auth_router)
app.include_router(letta_router)


# Request / Response models live in app.schemas (extracted for SRP).
from app.schemas import (  # noqa: E402
    AgentChatRequest,
    AuroraSearchRequest,
    AuroraStoreRequest,
    ConfidenceRequest,
    ConversationRequest,
    EnrichedSearchRequest,
    GraphQueryRequest,
    GraphSearchRequest,
    MemoryItem,
    MessageSearchRequest,
    MessageStoreRequest,
    RetrieveRequest,
    RetrieveResponse,
    SemanticSearchRequest,
    SemanticStoreRequest,
    StoreRequest,
    StoreResponse,
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup():
    """Initialize schema, run migrations, and validate API key on startup."""
    from app.config import settings

    # Log embedding mode
    if settings.effective_embedding_api_key and settings.effective_embedding_api_base:
        logger.info(
            "startup: embeddings — API mode, base=%s model=%s",
            settings.effective_embedding_api_base,
            settings.embedding_api_model,
        )
    else:
        logger.info(
            "startup: embeddings — local model '%s' (%d-dim)",
            settings.embedding_model,
            settings.embedding_dimensions,
        )

    # Validate LLM API key for entity extraction
    if not settings.effective_claude_api_key:
        logger.warning(
            "startup: LLM API key not configured — entity extraction will be skipped"
        )
    else:
        logger.info("startup: LLM API key present — provider=%s", settings.llm_provider)

    try:
        init_schema()
    except Exception:
        pass  # Neo4j may not be connected yet during first startup

    try:
        from app.migrations import run_migrations
        await run_migrations()
    except Exception:
        pass  # DB may not be ready yet during first startup


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def get_health():
    chromadb_ok = chroma_health()
    try:
        neo4j_ok = neo4j_health()
    except Exception:
        neo4j_ok = False
    try:
        letta_ok = letta_health()
    except Exception:
        letta_ok = False
    try:
        pg_ok = await pgvector_health()
    except Exception:
        pg_ok = False
    if not chromadb_ok:
        raise HTTPException(status_code=503, detail="ChromaDB unreachable")
    return {
        "status": "ok",
        "chromadb": chromadb_ok,
        "neo4j": neo4j_ok,
        "letta": letta_ok,
        "pgvector": pg_ok,
    }


# ---------------------------------------------------------------------------
# Episodic memory (ChromaDB)
# ---------------------------------------------------------------------------


@app.post("/store", response_model=StoreResponse)
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


@app.post("/retrieve", response_model=RetrieveResponse)
@limiter.limit("30/minute")
def retrieve(request: Request, req: RetrieveRequest):
    results = retrieve_similar(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return RetrieveResponse(memories=[MemoryItem(**m) for m in results])


@app.delete("/memories/{memory_id}", status_code=204)
def delete(memory_id: str):
    delete_memory(memory_id)


# ---------------------------------------------------------------------------
# Knowledge graph (Neo4j)
# ---------------------------------------------------------------------------


@app.post("/graph/search")
@limiter.limit("30/minute")
def graph_search(request: Request, req: GraphSearchRequest):
    results = search_graph(query=req.query, type_filter=req.type_filter, limit=req.limit)
    return {"entities": results}


@app.post("/graph/query")
@limiter.limit("30/minute")
def graph_query(request: Request, req: GraphQueryRequest):
    result = query_graph(entity_name=req.entity_name, depth=req.depth)
    if result["entity"] is None:
        raise HTTPException(status_code=404, detail=f"Entity '{req.entity_name}' not found")
    return result


@app.get("/graph/entity/{name}")
def graph_entity(name: str, depth: int = 2):
    result = get_entity_graph(name, depth=depth)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return result


@app.post("/search/enriched")
@limiter.limit("30/minute")
def enriched_search(request: Request, req: EnrichedSearchRequest):
    results = augment_with_graph_context(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return {"memories": results}


# ---------------------------------------------------------------------------
# pgvector conversation memory
# ---------------------------------------------------------------------------


@app.post("/memory/messages", status_code=201)
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


@app.get("/memory/threads/{thread_id}")
async def get_thread_endpoint(thread_id: str, limit: int = 100):
    messages = await retrieve_thread(thread_id, limit=limit)
    return {"thread_id": thread_id, "messages": messages, "count": len(messages)}


@app.post("/memory/search")
@limiter.limit("60/minute")
async def search_messages_endpoint(request: Request, req: MessageSearchRequest):
    results = await search_conversations(
        query=req.query,
        top_k=req.top_k,
        user_id=req.user_id,
        min_similarity=req.min_similarity,
    )
    return {"results": results, "query": req.query}


@app.post("/memory/semantic", status_code=201)
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


@app.post("/memory/semantic/search")
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
# Arquivo Aurora — affective memory
# ---------------------------------------------------------------------------


@app.post("/memory/aurora", status_code=201)
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


@app.post("/memory/aurora/search")
@limiter.limit("60/minute")
async def search_aurora_endpoint(request: Request, req: AuroraSearchRequest):
    results = await search_aurora_memories(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return {"results": results, "query": req.query}


@app.get("/memory/aurora/saudade")
@limiter.limit("60/minute")
async def aurora_saudade_endpoint(request: Request, user_id: str | None = None, limit: int = 1):
    results = await pull_saudade_memories(user_id=user_id, limit=limit)
    return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Confidence scoring endpoint
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


@app.post("/memory/confidence")
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


# ---------------------------------------------------------------------------
# LangGraph agent endpoints — NOT YET IMPLEMENTED (501)
# ---------------------------------------------------------------------------


@app.post("/agent/chat")
async def agent_chat_endpoint(req: AgentChatRequest):
    return JSONResponse(
        status_code=501,
        content={"detail": "LangGraph agent not yet implemented"},
    )


@app.get("/agent/threads/{thread_id}")
async def agent_thread_endpoint(thread_id: str):
    return JSONResponse(
        status_code=501,
        content={"detail": "LangGraph agent not yet implemented"},
    )


# ---------------------------------------------------------------------------
# MCP server info
# ---------------------------------------------------------------------------


@app.get("/mcp/info")
def mcp_info_endpoint():
    return {
        "server_name": "memory-server",
        "version": "0.1.0",
        "tools": [
            {
                "name": "store_message",
                "description": "Store a conversation message with its embedding for future retrieval.",
            },
            {
                "name": "search_memories",
                "description": "Search stored memories semantically across conversations and facts.",
            },
        ],
        "entry_point": "python -m app.mcp_server",
        "transport": "stdio",
    }


# ---------------------------------------------------------------------------
# Conversation loop — NOT YET IMPLEMENTED (501)
# ---------------------------------------------------------------------------


@app.post("/conversation")
async def conversation_endpoint(req: ConversationRequest):
    return JSONResponse(
        status_code=501,
        content={"detail": "Conversation loop not yet implemented"},
    )


# ---------------------------------------------------------------------------
# Chat endpoint — full pipeline: recall → LLM stream → persist
# Recall + system assembly live in app.chat_pipeline (shared with the inspector).
# ---------------------------------------------------------------------------


@app.post("/api/chat")
@limiter.limit("20/minute")
async def api_chat_endpoint(request: Request):
    """Full chat pipeline: recall memories, stream LLM response, persist turn."""
    body = await request.json()
    messages = body.get("messages", [])
    # Authenticated multi-user: user_id is derived from the signed token, not a
    # trusted header — this is what actually protects one user's memories from
    # another. No token → 401 (no x-user-id fallback, by design).
    user_id = user_from_authorization(request.headers.get("authorization"))
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    conversation_id = request.headers.get("x-conversation-id", "default")

    latest_user_message: str | None = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )

    async def generate():
        final_text = ""
        try:
            # 1. Recall memories + 2. build system — shared with the inspector
            #    (app.chat_pipeline) so what is sent here == what /inspect shows,
            #    including the relevance gating / dedup / budget curation.
            system = build_system("")
            if latest_user_message:
                try:
                    _, _, system = await recall_and_build_system(
                        user_id, latest_user_message
                    )
                except Exception as mem_err:
                    logger.warning("chat: memory recall failed: %s", mem_err)
                    system = build_system("")

            # 3. Fire-and-forget: store user message
            if latest_user_message:
                asyncio.create_task(
                    store_message(
                        thread_id=conversation_id,
                        role="user",
                        content=latest_user_message,
                        metadata={"user_id": user_id},
                    )
                )

            # 4. Stream from LLM via the unified client; re-wrap each chunk into
            #    our browser-facing SSE protocol.
            async for chunk in stream_text(
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                system=system,
                max_tokens=4096,
            ):
                final_text += chunk
                yield f"data: {json.dumps({'event': 'text', 'content': chunk})}\n\n"

            yield f"data: {json.dumps({'event': 'done', 'text': final_text})}\n\n"

            # 5. Fire-and-forget: persist assistant reply
            final_reply = final_text.strip()
            if latest_user_message and final_reply:
                async def _persist():
                    try:
                        await store_message(
                            thread_id=conversation_id,
                            role="assistant",
                            content=final_reply,
                            metadata={"user_id": user_id},
                        )
                    except Exception as e:
                        logger.warning("chat: store assistant msg failed: %s", e)
                    try:
                        await asyncio.to_thread(
                            store_conversation,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            content=f"User: {latest_user_message}\nAssistant: {final_reply}",
                            metadata={"source": "chat"},
                            extract_knowledge_graph=True,
                        )
                    except Exception as e:
                        logger.warning("chat: episodic store failed: %s", e)
                    try:
                        record = await asyncio.to_thread(
                            extract_aurora, latest_user_message, final_reply
                        )
                        if record:
                            await store_aurora_memory(user_id=user_id, **record)
                            logger.info("chat: aurora record stored (tipo=%s)", record.get("tipo"))
                    except Exception as e:
                        logger.warning("chat: aurora extraction failed: %s", e)
                    try:
                        from app.semantic_extractor import extract_semantic_facts

                        facts = await asyncio.to_thread(
                            extract_semantic_facts, latest_user_message, final_reply
                        )
                        for f in facts:
                            await store_semantic_memory(
                                user_id=user_id,
                                key=f["key"],
                                content=f["content"],
                                importance=f["importance"],
                            )
                        if facts:
                            logger.info("chat: %d semantic fact(s) stored", len(facts))
                    except Exception as e:
                        logger.warning("chat: semantic extraction failed: %s", e)

                asyncio.create_task(_persist())

        except Exception as exc:
            logger.exception("chat endpoint error")
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
