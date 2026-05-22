"""MCP Memory Server — standardized tool interface for memory operations.

Exposes `store_message` and `search_memories` tools via the Model Context Protocol.
Decouples memory infrastructure from agent logic so memory backends can be
swapped without touching agent code.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import settings

mcp = FastMCP("Memory Server")


@mcp.tool()
async def store_message(
    thread_id: str,
    role: str,
    content: str,
    user_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Store a message in the conversation memory with embedding.

    Args:
        thread_id: The conversation thread identifier.
        role: Message role (user, assistant, tool, system).
        content: The message text content.
        user_id: Optional user identifier for cross-thread retrieval.
        metadata: Optional additional metadata.

    Returns:
        dict with stored message id.
    """
    from app.pgvector_client import store_message as pg_store
    from app.embeddings import embed_text_async

    embedding = await embed_text_async(content)
    meta = metadata or {}
    if user_id:
        meta["user_id"] = user_id

    msg_id = await pg_store(
        thread_id=thread_id,
        role=role,
        content=content,
        embedding=embedding,
        metadata=meta,
    )
    return {"id": msg_id}


@mcp.tool()
async def search_memories(
    query: str,
    user_id: str = "",
    top_k: int = 5,
    min_similarity: float = 0.65,
) -> list[dict[str, Any]]:
    """Search conversation memories by semantic similarity.

    Args:
        query: The search query text.
        user_id: Optional filter by user.
        top_k: Maximum number of results to return.
        min_similarity: Minimum cosine similarity threshold (0-1).

    Returns:
        List of matching memory items with id, content, metadata, and similarity.
    """
    from app.pgvector_client import search_similar
    from app.embeddings import embed_text_async

    embedding = await embed_text_async(query)
    results = await search_similar(
        embedding=embedding,
        user_id=user_id or None,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    return results


@mcp.tool()
async def get_context(
    thread_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Load thread conversation history.

    Args:
        thread_id: The conversation thread identifier.
        limit: Maximum number of messages to return.

    Returns:
        List of messages in chronological order.
    """
    from app.pgvector_client import load_thread

    return await load_thread(thread_id, limit=limit)


@mcp.tool()
async def store_semantic_fact(
    user_id: str,
    key: str,
    content: str,
    importance: float = 0.0,
) -> dict[str, str]:
    """Store a semantic fact in long-term memory.

    Args:
        user_id: The user this fact belongs to.
        key: Unique key for this fact (upserted).
        content: The fact content.
        importance: Importance score (0-1, higher = more important).

    Returns:
        dict with stored fact id.
    """
    from app.pgvector_client import store_semantic
    from app.embeddings import embed_text_async

    embedding = await embed_text_async(content)
    fact_id = await store_semantic(
        user_id=user_id,
        key=key,
        content=content,
        embedding=embedding,
        importance=importance,
    )
    return {"id": fact_id}


def run() -> None:
    """Run the MCP memory server (stdio transport)."""
    mcp.run(transport="stdio")
