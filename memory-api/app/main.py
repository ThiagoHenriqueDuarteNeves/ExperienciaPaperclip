from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi.responses import StreamingResponse

from app.chroma_client import health as chroma_health
from app.retrieval import delete_memory, retrieve_similar, store_conversation

# Phase 1: pgvector-backed memory
from app.conversation_store import (
    retrieve_thread,
    search_similar as search_conversations,
    store_message,
)
from app.pgvector_client import health as pgvector_health
from app.semantic_store import search_semantic_memories, store_semantic_memory

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


try:
    from app.conversation_loop import run_conversation_loop
    _CONVERSATION_LOOP_AVAILABLE = True
except ImportError:
    _CONVERSATION_LOOP_AVAILABLE = False

    async def run_conversation_loop(*args, **kwargs):
        yield {"event": "error", "message": "Conversation loop not available"}


app = FastAPI(title="Episodic, Semantic & Procedural Memory API", version="0.3.0")


class StoreRequest(BaseModel):
    user_id: str
    conversation_id: str
    content: str
    metadata: dict | None = None
    extract_kg: bool = True


class StoreResponse(BaseModel):
    memory_id: str


class RetrieveRequest(BaseModel):
    query: str
    user_id: str | None = None
    top_k: int | None = None


class MemoryItem(BaseModel):
    id: str
    content: str
    metadata: dict
    similarity: float


class RetrieveResponse(BaseModel):
    memories: list[MemoryItem]


class GraphSearchRequest(BaseModel):
    query: str
    type_filter: str | None = None
    limit: int = 10


class GraphQueryRequest(BaseModel):
    entity_name: str
    depth: int = 2


class EnrichedSearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    top_k: int | None = None


@app.on_event("startup")
async def on_startup():
    """Initialize schema and run migrations on startup."""
    try:
        init_schema()
    except Exception:
        pass  # May not be connected yet during first startup

    # Phase 1: run pgvector migrations
    try:
        from app.migrations import run_migrations

        await run_migrations()
    except Exception:
        pass  # DB may not be ready yet during first startup


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


@app.post("/store", response_model=StoreResponse)
def store(req: StoreRequest):
    memory_id = store_conversation(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        content=req.content,
        metadata=req.metadata,
        extract_knowledge_graph=req.extract_kg,
    )
    return StoreResponse(memory_id=memory_id)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    results = retrieve_similar(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return RetrieveResponse(memories=[MemoryItem(**m) for m in results])


@app.delete("/memories/{memory_id}", status_code=204)
def delete(memory_id: str):
    delete_memory(memory_id)


# --- Semantic / Knowledge Graph endpoints ---


@app.post("/graph/search")
def graph_search(req: GraphSearchRequest):
    """Search entities in the knowledge graph by name/description."""
    results = search_graph(query=req.query, type_filter=req.type_filter, limit=req.limit)
    return {"entities": results}


@app.post("/graph/query")
def graph_query(req: GraphQueryRequest):
    """Query the knowledge graph centered on an entity."""
    result = query_graph(entity_name=req.entity_name, depth=req.depth)
    if result["entity"] is None:
        raise HTTPException(status_code=404, detail=f"Entity '{req.entity_name}' not found")
    return result


@app.get("/graph/entity/{name}")
def graph_entity(name: str, depth: int = 2):
    """Full graph neighborhood for an entity."""
    result = get_entity_graph(name, depth=depth)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return result


@app.post("/search/enriched")
def enriched_search(req: EnrichedSearchRequest):
    """Hybrid retrieval: vector similarity + graph context."""
    results = augment_with_graph_context(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return {"memories": results}


# ---------------------------------------------------------------------------
# Phase 1: pgvector memory endpoints (async)
# ---------------------------------------------------------------------------


class MessageStoreRequest(BaseModel):
    thread_id: str
    role: str  # user, assistant, system
    content: str
    metadata: dict | None = None


class MessageSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    user_id: str | None = None
    min_similarity: float = 0.65


class SemanticStoreRequest(BaseModel):
    user_id: str
    key: str
    content: str
    importance: float = 0.0


class SemanticSearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    top_k: int = 5
    min_similarity: float = 0.5


@app.post("/memory/messages", status_code=201)
async def store_message_endpoint(req: MessageStoreRequest):
    """Store a conversation message with embedding in pgvector."""
    if req.role not in ("user", "assistant", "system"):
        raise HTTPException(status_code=400, detail="role must be user, assistant, or system")
    message_id = await store_message(
        thread_id=req.thread_id,
        role=req.role,
        content=req.content,
        metadata=req.metadata,
    )
    return {"message_id": message_id, "status": "stored"}


@app.get("/memory/threads/{thread_id}")
async def get_thread_endpoint(thread_id: str, limit: int = 100):
    """Load full thread history from pgvector in chronological order."""
    messages = await retrieve_thread(thread_id, limit=limit)
    return {"thread_id": thread_id, "messages": messages, "count": len(messages)}


@app.post("/memory/search")
async def search_messages_endpoint(req: MessageSearchRequest):
    """ANN similarity search over conversation messages using pgvector."""
    results = await search_conversations(
        query=req.query,
        top_k=req.top_k,
        user_id=req.user_id,
        min_similarity=req.min_similarity,
    )
    return {"results": results, "query": req.query}


@app.post("/memory/semantic", status_code=201)
async def store_semantic_endpoint(req: SemanticStoreRequest):
    """Store a semantic memory fact with embedding (upsert)."""
    fact_id = await store_semantic_memory(
        user_id=req.user_id,
        key=req.key,
        content=req.content,
        importance=req.importance,
    )
    return {"fact_id": fact_id, "status": "stored"}


@app.post("/memory/semantic/search")
async def search_semantic_endpoint(req: SemanticSearchRequest):
    """Semantic search over stored facts using pgvector ANN."""
    results = await search_semantic_memories(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
    )
    return {"results": results, "query": req.query}


# ---------------------------------------------------------------------------
# Phase 1: LangGraph agent endpoints
# ---------------------------------------------------------------------------


class AgentChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str | None = None


@app.post("/agent/chat")
async def agent_chat_endpoint(req: AgentChatRequest):
    """Run the LangGraph agent for a single turn.

    Stores the exchange in conversation_messages and checkpoints agent state.
    Returns the assistant response with any tool calls made.
    """
    if not _AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent not available")
    try:
        result = await run_agent(
            user_message=req.message,
            thread_id=req.thread_id,
            user_id=req.user_id,
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/agent/threads/{thread_id}")
async def agent_thread_endpoint(thread_id: str):
    """Load full conversation history for an agent thread."""
    if not _AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent not available")
    messages = await load_thread_history(thread_id)
    return {"thread_id": thread_id, "messages": messages, "count": len(messages)}


# ---------------------------------------------------------------------------
# Phase 1: MCP server info
# ---------------------------------------------------------------------------


@app.get("/mcp/info")
def mcp_info_endpoint():
    """Information about the MCP memory server."""
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


class CreateAgentRequest(BaseModel):
    name: str
    human_block: str = ""
    persona_block: str = ""
    system_prompt: str | None = None


class ArchivalInsertRequest(BaseModel):
    content: str


class ArchivalSearchRequest(BaseModel):
    query: str
    limit: int = 10


class BlockUpdateRequest(BaseModel):
    value: str


@app.get("/letta/health")
def letta_health_endpoint():
    """Check Letta server connectivity."""
    ok = letta_health()
    return {"letta_available": ok}


@app.post("/letta/agents")
def letta_create_agent_endpoint(req: CreateAgentRequest):
    """Create a new Letta agent with core memory blocks."""
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
    """List all Letta agents."""
    return {"agents": letta_list_agents()}


@app.get("/letta/agents/{agent_id}")
def letta_get_agent_endpoint(agent_id: str):
    """Get agent details by ID."""
    agent = letta_lookup_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent


@app.delete("/letta/agents/{agent_id}", status_code=204)
def letta_delete_agent_endpoint(agent_id: str):
    """Delete a Letta agent and its memories."""
    letta_delete_agent(agent_id)


@app.get("/letta/agents/{agent_id}/memory")
def letta_get_memory_endpoint(agent_id: str):
    """Get core memory blocks (human + persona) for an agent."""
    memory = letta_get_core_memory(agent_id)
    if memory is None:
        raise HTTPException(
            status_code=404, detail=f"Memory not found for agent '{agent_id}'"
        )
    return memory


@app.put("/letta/agents/{agent_id}/memory/human")
def letta_update_human_endpoint(agent_id: str, req: BlockUpdateRequest):
    """Update the human block of an agent's core memory."""
    result = letta_update_human(agent_id, req.value)
    if result is None:
        raise HTTPException(
            status_code=503, detail="Failed to update human memory block"
        )
    return result


@app.put("/letta/agents/{agent_id}/memory/persona")
def letta_update_persona_endpoint(agent_id: str, req: BlockUpdateRequest):
    """Update the persona block of an agent's core memory."""
    result = letta_update_persona(agent_id, req.value)
    if result is None:
        raise HTTPException(
            status_code=503, detail="Failed to update persona memory block"
        )
    return result


@app.post("/letta/agents/{agent_id}/archival")
def letta_insert_archival_endpoint(agent_id: str, req: ArchivalInsertRequest):
    """Insert a passage into the agent's archival memory."""
    result = letta_insert_archival(agent_id, req.content)
    if result is None:
        raise HTTPException(
            status_code=503, detail="Failed to insert archival memory"
        )
    return result


@app.post("/letta/agents/{agent_id}/archival/search")
def letta_search_archival_endpoint(agent_id: str, req: ArchivalSearchRequest):
    """Semantic search over the agent's archival memory."""
    results = letta_search_archival(
        agent_id=agent_id, query=req.query, limit=req.limit
    )
    return {"results": results}


# ---------------------------------------------------------------------------
# Conversation loop endpoint (SSE streaming)
# ---------------------------------------------------------------------------


class ConversationRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    history: list[dict] | None = None
    agent_id: str | None = None


@app.post("/conversation")
async def conversation_endpoint(req: ConversationRequest):
    """Run the full conversation loop and stream results via SSE.

    Events emitted: context, text, tool_call, done, error.
    """
    async def event_stream():
        async for event in run_conversation_loop(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            user_message=req.message,
            history=req.history,
            agent_id=req.agent_id,
        ):
            import json as _json
            yield f"data: {_json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
