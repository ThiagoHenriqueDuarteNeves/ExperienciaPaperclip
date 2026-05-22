import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    embedding_model: str = "all-MiniLM-L6-v2"
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
    llm_api_base: str = "https://api.anthropic.com/v1"
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

    # Embedding Configuration
    # Set MEMORY_EMBEDDING_API_BASE + MEMORY_EMBEDDING_API_KEY to use an external API.
    # When empty and provider is 'deepseek', defaults to Deepseek's embedding API.
    # When empty and provider is 'anthropic', falls back to local SentenceTransformer model.
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_api_model: str = "text-embedding-3-large"

    # Deepseek-specific key (used for embeddings when provider is 'deepseek')
    deepseek_api_key: str = ""

    letta_base_url: str = "http://letta:8283"
    letta_api_key: str = ""

    @property
    def effective_letta_api_key(self) -> str:
        return self.letta_api_key or os.environ.get("LETTA_API_KEY", "")

    @property
    def effective_claude_api_key(self) -> str:
        return self.claude_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def effective_llm_api_base(self) -> str:
        """Get LLM API base URL, supporting both Anthropic and Deepseek."""
        if self.llm_provider == "deepseek":
            return "https://api.deepseek.com/anthropic"
        return self.llm_api_base or "https://api.anthropic.com/v1"

    @property
    def effective_embedding_api_base(self) -> str:
        """Get the effective embedding API base URL based on the provider.

        - If explicitly set via MEMORY_EMBEDDING_API_BASE, use it.
        - If provider is 'deepseek', use Deepseek's embedding endpoint.
        - If provider is 'anthropic', return empty (use local model).
        """
        if self.embedding_api_base:
            return self.embedding_api_base
        if self.llm_provider == "deepseek":
            return "https://api.deepseek.com/v1"
        return ""

    @property
    def effective_embedding_api_key(self) -> str:
        """Get the effective embedding API key based on the provider.

        - If explicitly set via MEMORY_EMBEDDING_API_KEY, use it.
        - If provider is 'deepseek', use the Deepseek API key.
        - If provider is 'anthropic', return empty (use local model).
        """
        if self.embedding_api_key:
            return self.embedding_api_key
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        return ""

    model_config = {"env_prefix": "MEMORY_"}


settings = Settings()
