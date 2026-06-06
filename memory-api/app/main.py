import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.chroma_client import health as chroma_health
from app.logging_config import configure_logging

# pgvector health for /health; retrieve_thread for the (experimental) LangGraph
# helper below. The /memory/* routes live in app.routers.memory.
from app.conversation_store import retrieve_thread
from app.pgvector_client import health as pgvector_health

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

# Neo4j optional-import block + graph routes live in app.routers.graph.
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
# Each domain router owns its own optional-import block (graph→neo4j, aurora,
# letta); main.py reuses neo4j_health/init_schema and letta_health for /health
# and startup.
from app.inspect_api import router as inspect_router  # noqa: E402
from app.routers.aurora import router as aurora_router  # noqa: E402
from app.routers.auth import router as auth_router  # noqa: E402
from app.routers.chat import router as chat_router  # noqa: E402
from app.routers.episodic import router as episodic_router  # noqa: E402
from app.routers.graph import (  # noqa: E402
    init_schema,
    neo4j_health,
    router as graph_router,
)
from app.routers.letta import letta_health, router as letta_router  # noqa: E402
from app.routers.memory import router as memory_router  # noqa: E402
from app.routers.system import router as system_router  # noqa: E402

app.include_router(inspect_router)
app.include_router(aurora_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(episodic_router)
app.include_router(graph_router)
app.include_router(letta_router)
app.include_router(memory_router)
app.include_router(system_router)


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
