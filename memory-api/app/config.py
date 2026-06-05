import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    embedding_model: str = "intfloat/multilingual-e5-large"
    collection_name: str = "episodic_memories"
    similarity_top_k: int = 5
    min_similarity_threshold: float = 0.65

    chromadb_use_local: bool = True
    chromadb_local_persist_dir: str = "./chroma_data"

    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "semantic-memory-local"

    # LLM Configuration (Anthropic or Deepseek)
    llm_provider: str = "anthropic"  # 'anthropic' or 'deepseek'
    # Explicit override (MEMORY_LLM_API_BASE). When empty, a per-provider default
    # is used (see effective_llm_api_base). Set this to point at any
    # Anthropic-Messages-compatible endpoint (e.g. a local proxy).
    llm_api_base: str = ""
    # Default base for the 'deepseek' provider — overridable via
    # MEMORY_DEEPSEEK_API_BASE without changing provider/code.
    deepseek_api_base: str = "https://api.deepseek.com/anthropic"
    anthropic_api_base: str = "https://api.anthropic.com/v1"
    claude_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250506"
    max_extraction_retries: int = 2

    pgvector_host: str = "memory-db"
    pgvector_port: int = 5432
    pgvector_user: str = "memory"
    pgvector_password: str = "memory-local"
    pgvector_database: str = "memory_db"
    pgvector_min_pool: int = 2
    pgvector_max_pool: int = 10

    # Embedding Configuration — local model (intfloat/multilingual-e5-large, 1024-dim).
    # Supports API override: set MEMORY_EMBEDDING_API_BASE + MEMORY_EMBEDDING_API_KEY
    # to use OpenAI or another compat provider instead.
    # Changing between local↔API requires recreating the schema if dimensions differ:
    #   docker compose down -v && docker compose up
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_api_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024

    # Deepseek key (LLM only — not used for embeddings)
    deepseek_api_key: str = ""

    rrf_k: int = 60
    fts_language: str = "portuguese"

    # Recall curation (chat_pipeline). Smaller top_k + a relevance gate + a char
    # budget keep the injected memory compact instead of dumping every hit.
    recall_episodic_k: int = 4
    recall_convo_k: int = 4
    recall_identity_k: int = 4
    recall_aurora_k: int = 3
    recall_min_similarity: float = 0.7
    recall_char_budget: int = 6000
    # Fase C — cross-encoder re-ranking (opt-in; heavy extra model, default off).
    recall_rerank_enabled: bool = False
    recall_rerank_top_n: int = 6
    recall_rerank_candidate_k: int = 12
    recall_rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    letta_base_url: str = "http://letta:8283"
    letta_api_key: str = ""

    # Auth (multi-user, PIN validated server-side). Tokens are HMAC-signed with
    # this secret — set MEMORY_AUTH_SECRET in production. Empty falls back to a
    # dev secret (logged as a warning on startup).
    auth_secret: str = ""
    auth_token_ttl_hours: int = 720  # 30 days

    # Memory inspector (local debug tool). OFF by default so the endpoints are
    # never exposed via the zrok share unless explicitly enabled for local use.
    # Enable with MEMORY_INSPECT_ENABLED=true.
    inspect_enabled: bool = False

    @property
    def effective_auth_secret(self) -> str:
        return self.auth_secret or "dev-insecure-auth-secret-change-me"

    @property
    def effective_letta_api_key(self) -> str:
        return self.letta_api_key or os.environ.get("LETTA_API_KEY", "")

    @property
    def effective_claude_api_key(self) -> str:
        return self.claude_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def effective_llm_api_base(self) -> str:
        """Get LLM API base URL.

        Resolution order:
        1. MEMORY_LLM_API_BASE — explicit override, always wins (any provider).
        2. per-provider default — deepseek -> MEMORY_DEEPSEEK_API_BASE,
           otherwise MEMORY_ANTHROPIC_API_BASE.
        """
        if self.llm_api_base:
            return self.llm_api_base
        if self.llm_provider == "deepseek":
            return self.deepseek_api_base
        return self.anthropic_api_base

    @property
    def effective_embedding_api_base(self) -> str:
        """Embedding base URL — must be set explicitly via MEMORY_EMBEDDING_API_BASE.
        Deepseek does not offer an embeddings API; use OpenAI or another provider.
        """
        return self.embedding_api_base

    @property
    def effective_embedding_api_key(self) -> str:
        """Embedding API key — reads MEMORY_EMBEDDING_API_KEY explicitly."""
        return self.embedding_api_key

    model_config = {"env_prefix": "MEMORY_"}


settings = Settings()
