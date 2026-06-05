import asyncio
import json
import logging
import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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

# Phase 4: multi-user auth
from app.auth import (
    hash_pin,
    issue_token,
    user_from_authorization,
    verify_pin,
)
from app.pgvector_client import create_user_profile, get_user_profile

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

try:
    from app.letta_manager import (
        create_agent as letta_create_agent,
        delete_agent as letta_delete_agent,
        get_core_memory as letta_get_core_memory,
        health as letta_health,
        insert_archival_memory as letta_insert_archival,
        list_agents as letta_list_agents,
        lookup_agent as letta_lookup_agent,
        search_archival_memory as letta_search_archival,
        update_human_block as letta_update_human,
        update_persona_block as letta_update_persona,
    )
    _LETTA_AVAILABLE = True
except ImportError:
    _LETTA_AVAILABLE = False

    def letta_health() -> bool:
        return False

    def letta_list_agents() -> list:
        return []

    def letta_lookup_agent(*args, **kwargs):
        return None

    def letta_create_agent(*args, **kwargs):
        return None

    def letta_delete_agent(*args, **kwargs):
        return None

    def letta_get_core_memory(*args, **kwargs):
        return None

    def letta_update_human(*args, **kwargs):
        return None

    def letta_update_persona(*args, **kwargs):
        return None

    def letta_insert_archival(*args, **kwargs):
        return None

    def letta_search_archival(*args, **kwargs):
        return []


# Conversation loop is not yet fully implemented — routes return 501.
_CONVERSATION_LOOP_AVAILABLE = False


limiter = Limiter(key_func=get_remote_address)

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

# Memory inspector (local debug tool). Endpoints self-gate on settings.inspect_enabled.
from app.inspect_api import router as inspect_router  # noqa: E402

app.include_router(inspect_router)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class StoreRequest(BaseModel):
    user_id: str = Field(..., max_length=256)
    conversation_id: str = Field(..., max_length=256)
    content: str = Field(..., max_length=50_000)
    metadata: dict | None = None
    extract_kg: bool = True


class StoreResponse(BaseModel):
    memory_id: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=50)


class MemoryItem(BaseModel):
    id: str
    content: str
    metadata: dict
    similarity: float


class RetrieveResponse(BaseModel):
    memories: list[MemoryItem]


class GraphSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    type_filter: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=10, ge=1, le=50)


class GraphQueryRequest(BaseModel):
    entity_name: str = Field(..., max_length=512)
    depth: int = Field(default=2, ge=1, le=5)


class EnrichedSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=50)


class MessageStoreRequest(BaseModel):
    thread_id: str = Field(..., max_length=256)
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., max_length=50_000)
    metadata: dict | None = None


class MessageSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)
    user_id: str | None = Field(default=None, max_length=256)
    min_similarity: float = Field(default=0.65, ge=0.0, le=1.0)


class SemanticStoreRequest(BaseModel):
    user_id: str = Field(..., max_length=256)
    key: str = Field(..., max_length=512)
    content: str = Field(..., max_length=50_000)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)


class AuroraStoreRequest(BaseModel):
    user_id: str = Field(default="default", max_length=256)
    tipo: str = Field(..., pattern="^(conversa|descoberta|correcao|silencio|brincadeira|confissao|momento_espontaneo)$")
    disparo: str = Field(..., pattern="^(usuario_falou|usuario_corrigiu|pausa_longa|palavra_chave|silencio|espontaneo)$")
    fato: str = Field(..., max_length=10_000)
    tom_do_usuario: str = Field(..., pattern="^(afetivo|serio|brincalhao|frustrado|curioso|vulneravel|silencioso)$")
    minha_reacao_emocional: str = Field(..., max_length=10_000)
    reacao_simulada: str | None = Field(default=None, max_length=512)
    importancia: int = Field(..., ge=1, le=5)
    ressonancia: int = Field(..., ge=1, le=5)
    intimidade: str = Field(..., pattern="^(publico|pessoal|nosso_so_nosso)$")
    saudade: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)
    conexoes: list[str] = Field(default_factory=list)
    contexto_extra: str | None = Field(default=None, max_length=10_000)
    bilhete_interno: str | None = Field(default=None, max_length=10_000)
    data: str | None = Field(default=None, max_length=10)


class AuroraSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int = Field(default=5, ge=1, le=50)


class RegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, pattern="^[a-z0-9_-]+$")
    pin: str = Field(..., min_length=4, max_length=32)
    display_name: str = Field(default="", max_length=128)


class LoginRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, pattern="^[a-z0-9_-]+$")
    pin: str = Field(..., min_length=4, max_length=32)


class AgentChatRequest(BaseModel):
    message: str = Field(..., max_length=10_000)
    thread_id: str | None = Field(default=None, max_length=256)
    user_id: str | None = Field(default=None, max_length=256)


class CreateAgentRequest(BaseModel):
    name: str = Field(..., max_length=256)
    human_block: str = Field(default="", max_length=2_000)
    persona_block: str = Field(default="", max_length=2_000)
    system_prompt: str | None = Field(default=None, max_length=10_000)


class ArchivalInsertRequest(BaseModel):
    content: str = Field(..., max_length=50_000)


class ArchivalSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)


class BlockUpdateRequest(BaseModel):
    value: str = Field(..., max_length=2_000)


class ConversationRequest(BaseModel):
    user_id: str = Field(..., max_length=256)
    conversation_id: str = Field(..., max_length=256)
    message: str = Field(..., max_length=10_000)
    history: list[dict] | None = None
    agent_id: str | None = Field(default=None, max_length=256)


class ConfidenceRequest(BaseModel):
    text: str = Field(..., max_length=10_000)
    threshold: float = Field(default=0.72, ge=0.0, le=1.0)


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
# Letta (MemGPT) Procedural Memory endpoints
# ---------------------------------------------------------------------------


@app.get("/letta/health")
def letta_health_endpoint():
    ok = letta_health()
    return {"letta_available": ok}


@app.post("/letta/agents")
def letta_create_agent_endpoint(req: CreateAgentRequest):
    agent = letta_create_agent(
        name=req.name,
        human_block=req.human_block,
        persona_block=req.persona_block,
        system_prompt=req.system_prompt,
    )
    if agent is None:
        raise HTTPException(status_code=503, detail="Letta agent creation failed")
    return agent


@app.get("/letta/agents")
def letta_list_agents_endpoint():
    return {"agents": letta_list_agents()}


@app.get("/letta/agents/{agent_id}")
def letta_get_agent_endpoint(agent_id: str):
    agent = letta_lookup_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent


@app.delete("/letta/agents/{agent_id}", status_code=204)
def letta_delete_agent_endpoint(agent_id: str):
    letta_delete_agent(agent_id)


@app.get("/letta/agents/{agent_id}/memory")
def letta_get_memory_endpoint(agent_id: str):
    memory = letta_get_core_memory(agent_id)
    if memory is None:
        raise HTTPException(
            status_code=404, detail=f"Memory not found for agent '{agent_id}'"
        )
    return memory


@app.put("/letta/agents/{agent_id}/memory/human")
def letta_update_human_endpoint(agent_id: str, req: BlockUpdateRequest):
    result = letta_update_human(agent_id, req.value)
    if result is None:
        raise HTTPException(status_code=503, detail="Failed to update human memory block")
    return result


@app.put("/letta/agents/{agent_id}/memory/persona")
def letta_update_persona_endpoint(agent_id: str, req: BlockUpdateRequest):
    result = letta_update_persona(agent_id, req.value)
    if result is None:
        raise HTTPException(status_code=503, detail="Failed to update persona memory block")
    return result


@app.post("/letta/agents/{agent_id}/archival")
def letta_insert_archival_endpoint(agent_id: str, req: ArchivalInsertRequest):
    result = letta_insert_archival(agent_id, req.content)
    if result is None:
        raise HTTPException(status_code=503, detail="Failed to insert archival memory")
    return result


@app.post("/letta/agents/{agent_id}/archival/search")
def letta_search_archival_endpoint(agent_id: str, req: ArchivalSearchRequest):
    results = letta_search_archival(
        agent_id=agent_id, query=req.query, limit=req.limit
    )
    return {"results": results}


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
# Auth — multi-user profiles with server-validated PIN
# ---------------------------------------------------------------------------


@app.post("/auth/register")
@limiter.limit("5/minute")
async def auth_register_endpoint(request: Request, req: RegisterRequest):
    existing = await get_user_profile(req.user_id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="user_id already taken")
    pin_hash, pin_salt = hash_pin(req.pin)
    display_name = req.display_name or req.user_id
    await create_user_profile(req.user_id, display_name, pin_hash, pin_salt)
    logger.info("auth: registered user_id=%s", req.user_id)
    token = issue_token(req.user_id)
    return {"token": token, "user_id": req.user_id, "display_name": display_name}


@app.post("/auth/login")
@limiter.limit("6/minute")
async def auth_login_endpoint(request: Request, req: LoginRequest):
    profile = await get_user_profile(req.user_id)
    if profile is None or not verify_pin(req.pin, profile["pin_hash"], profile["pin_salt"]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = issue_token(req.user_id)
    return {
        "token": token,
        "user_id": req.user_id,
        "display_name": profile.get("display_name") or req.user_id,
    }


@app.get("/auth/me")
async def auth_me_endpoint(request: Request):
    user_id = user_from_authorization(request.headers.get("authorization"))
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid or missing token")
    profile = await get_user_profile(user_id)
    return {
        "user_id": user_id,
        "display_name": (profile or {}).get("display_name") or user_id,
    }


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
