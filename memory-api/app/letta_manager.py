"""Letta (MemGPT) integration for autonomous memory management.

Provides the procedural memory tier on top of the existing episodic
(ChromaDB) and semantic (Neo4j) stores:

  - Agent lifecycle: create, lookup, list, delete
  - Core memory block CRUD (human, persona blocks)
  - Archival memory with autonomous insert and semantic search

Letta handles its own archival memory compaction and eviction autonomously
based on access patterns and the system-configured storage policy.
"""

from __future__ import annotations

from typing import Any

from letta_client import Letta
from letta_client.types import CreateBlockParam

from app.config import settings

_client: Letta | None = None


def get_client() -> Letta:
    """Return a singleton Letta client connected to the configured server."""
    global _client
    if _client is None:
        api_key = settings.effective_letta_api_key or None
        _client = Letta(base_url=settings.letta_base_url, api_key=api_key)
    return _client


def health() -> bool:
    """Check whether the Letta server is reachable."""
    try:
        get_client().agents.list()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


def create_agent(
    name: str,
    human_block: str = "",
    persona_block: str = "",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Create a new Letta agent with core memory blocks."""
    client = get_client()
    memory_blocks = [
        CreateBlockParam(label="human", value=human_block or "", limit=2000),
        CreateBlockParam(label="persona", value=persona_block or "", limit=2000),
    ]
    kwargs: dict[str, Any] = {
        "name": name,
        "memory_blocks": memory_blocks,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    return client.agents.create(**kwargs).model_dump()


def lookup_agent(agent_id: str) -> dict[str, Any] | None:
    """Return agent metadata, or *None* if the agent does not exist."""
    try:
        client = get_client()
        return client.agents.retrieve(agent_id).model_dump()
    except Exception:
        return None


def list_agents() -> list[dict[str, Any]]:
    """List all agents visible to the configured API key."""
    client = get_client()
    return [a.model_dump() for a in client.agents.list()]


def delete_agent(agent_id: str) -> None:
    """Permanently delete an agent and all of its memories."""
    client = get_client()
    client.agents.delete(agent_id)


# ---------------------------------------------------------------------------
# Core memory block CRUD
# ---------------------------------------------------------------------------


def get_core_memory(agent_id: str) -> dict[str, Any] | None:
    """Return both ``human`` and ``persona`` core memory blocks."""
    try:
        client = get_client()
        blocks = client.agents.blocks.list(agent_id)
        return {"blocks": [b.model_dump() for b in blocks]}
    except Exception:
        return None


def update_human_block(agent_id: str, value: str) -> dict[str, Any]:
    """Replace the ``human`` block content for the given agent."""
    return _update_block(agent_id, "human", value)


def update_persona_block(agent_id: str, value: str) -> dict[str, Any]:
    """Replace the ``persona`` block content for the given agent."""
    return _update_block(agent_id, "persona", value)


def _update_block(agent_id: str, label: str, value: str) -> dict[str, Any]:
    """Internal helper: update a single core memory block by label."""
    client = get_client()
    result = client.agents.blocks.update(
        block_label=label,
        agent_id=agent_id,
        value=value,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Archival memory
# ---------------------------------------------------------------------------


def insert_archival_memory(agent_id: str, content: str) -> dict[str, Any]:
    """Insert a passage into the agent's archival memory.

    Letta autonomously manages archival memory compaction and eviction,
    so no explicit delete/compact step is required here.
    """
    client = get_client()
    return client.agents.passages.create(agent_id=agent_id, text=content).model_dump()


def search_archival_memory(
    agent_id: str,
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Semantic search over the agent's archival memory."""
    client = get_client()
    result = client.agents.passages.search(
        agent_id=agent_id, query=query, top_k=limit
    )
    return [p.model_dump() for p in result.passages] if hasattr(result, "passages") else []
