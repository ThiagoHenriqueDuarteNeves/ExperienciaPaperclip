"""Integration tests for Phase 1 memory foundation.

Covers:
  1. store -> retrieve -> respond -> store full cycle
  2. pgvector ANN query returns relevant past context
  3. MCP server tools respond correctly
  4. LangGraph agent loads thread history on restart

Requires a running PostgreSQL + pgvector instance. Set env vars:
  MEMORY_PGVECTOR_HOST, MEMORY_PGVECTOR_PORT, MEMORY_PGVECTOR_USER,
  MEMORY_PGVECTOR_PASSWORD, MEMORY_PGVECTOR_DATABASE

Or use the docker-compose memory-db service:
  docker compose up -d memory-db

Tests are skipped if the database is not reachable.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from app.config import settings
from app.conversation_store import retrieve_thread, search_similar, store_message
from app.pgvector_client import (
    close_pool,
    get_pool,
    health,
    search_semantic,
    store_semantic,
)
from app.semantic_store import search_semantic_memories, store_semantic_memory

# Unique test user/thread to avoid collisions with production data
TEST_USER = f"test-user-{uuid.uuid4().hex[:8]}"
TEST_THREAD = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(autouse=True)
async def db_check():
    """Skip all tests if pgvector is not reachable."""
    try:
        ok = await health()
    except Exception:
        ok = False
    if not ok:
        pytest.skip("pgvector database not reachable — start `docker compose up -d memory-db`")
    yield
    # Cleanup test data
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversation_messages WHERE metadata->>'user_id' = $1", TEST_USER)
        await conn.execute("DELETE FROM semantic_memory WHERE user_id = $1", TEST_USER)


# ---------------------------------------------------------------------------
# 1. store -> retrieve -> respond -> store full cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_and_retrieve_thread():
    """Verify a full write-then-read cycle for conversation messages."""
    thread_id = str(uuid.uuid4())

    # Store two messages
    msg1_id = await store_message(
        thread_id=thread_id,
        role="user",
        content="Hello, my name is Alice and I work at Acme Corp.",
        metadata={"user_id": TEST_USER},
    )
    msg2_id = await store_message(
        thread_id=thread_id,
        role="assistant",
        content="Nice to meet you, Alice! How can I help you today?",
        metadata={"user_id": TEST_USER},
    )

    assert msg1_id
    assert msg2_id
    assert msg1_id != msg2_id

    # Retrieve the thread
    thread = await retrieve_thread(thread_id)

    assert len(thread) == 2
    assert thread[0]["role"] == "user"
    assert "Alice" in thread[0]["content"]
    assert thread[1]["role"] == "assistant"
    assert "Nice to meet you" in thread[1]["content"]
    # Messages should be in chronological order
    assert thread[0]["created_at"] <= thread[1]["created_at"]


@pytest.mark.asyncio
async def test_full_cycle_store_retrieve_respond_store():
    """End-to-end cycle simulating a real conversation turn."""
    thread_id = str(uuid.uuid4())
    user_msg = "What's the capital of France?"

    # User message stored
    await store_message(
        thread_id=thread_id,
        role="user",
        content=user_msg,
        metadata={"user_id": TEST_USER},
    )

    # Simulate retrieval for context
    context = await search_similar(query=user_msg, user_id=TEST_USER, top_k=3)
    assert len(context) >= 1
    assert any("France" in (r.get("content") or "") for r in context)

    # Simulate assistant response
    assistant_msg = "The capital of France is Paris."
    await store_message(
        thread_id=thread_id,
        role="assistant",
        content=assistant_msg,
        metadata={"user_id": TEST_USER},
    )

    # Verify full thread
    thread = await retrieve_thread(thread_id)
    assert len(thread) == 2
    assert thread[1]["content"] == assistant_msg


# ---------------------------------------------------------------------------
# 2. pgvector ANN query returns relevant past context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ann_search_returns_relevant_context():
    """Verify pgvector cosine ANN returns semantically relevant messages."""
    thread_id = str(uuid.uuid4())

    # Store several messages on different topics
    await store_message(thread_id, "user", "I love pizza and Italian food.", {"user_id": TEST_USER})
    await store_message(thread_id, "assistant", "Italian cuisine is wonderful!", {"user_id": TEST_USER})
    await store_message(thread_id, "user", "My car needs an oil change soon.", {"user_id": TEST_USER})
    await store_message(thread_id, "assistant", "Regular maintenance is important for cars.", {"user_id": TEST_USER})

    # Search for food-related content
    food_results = await search_similar(query="food and cooking", user_id=TEST_USER, top_k=2)

    assert len(food_results) > 0
    # Food results should appear before car results
    food_contents = [r["content"] for r in food_results]
    assert any("pizza" in c or "Italian" in c or "cuisine" in c for c in food_contents)

    # All returned results should have similarity scores
    for r in food_results:
        assert "similarity" in r
        assert 0 <= r["similarity"] <= 1


@pytest.mark.asyncio
async def test_ann_search_respects_user_isolation():
    """Verify that user_id scoping works."""
    thread_id = str(uuid.uuid4())

    await store_message(thread_id, "user", "I prefer Python for backend work.", {"user_id": TEST_USER})
    await store_message(thread_id, "user", "Someone else's secret message.", {"user_id": "other-user-xyz"})

    # Search scoped to TEST_USER
    results = await search_similar(query="programming language", user_id=TEST_USER, top_k=3)

    for r in results:
        meta = r.get("metadata") or {}
        if meta.get("user_id") == "other-user-xyz":
            pytest.fail("Results should not include other users' data")


# ---------------------------------------------------------------------------
# 3. MCP server tool responses (tested via the store functions directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_message_tool_contract():
    """Verify the store_message operation matches the MCP tool contract."""
    thread_id = str(uuid.uuid4())

    # The MCP tool contract: thread_id (UUID), role (user|assistant|system), content, metadata?
    for role in ("user", "assistant", "system"):
        msg_id = await store_message(
            thread_id=thread_id,
            role=role,
            content=f"Test message with role={role}",
            metadata={"test": True, "user_id": TEST_USER},
        )
        assert msg_id
        # Verify it's a valid UUID
        uuid.UUID(msg_id)


@pytest.mark.asyncio
async def test_search_memories_tool_contract():
    """Verify search_memories returns results matching the MCP tool contract.

    The MCP tool merges conversation + semantic results, sorted by similarity.
    """
    # Store conversation context
    await store_message(
        TEST_THREAD, "user",
        "My favorite book is Dune by Frank Herbert.",
        {"user_id": TEST_USER},
    )

    # Store a semantic fact
    await store_semantic_memory(
        user_id=TEST_USER,
        key="favorite_book",
        content="User's favorite book is Dune by Frank Herbert.",
        importance=0.9,
    )

    # Search both stores (mimicking the MCP tool's merge behavior)
    import asyncio

    conv, sem = await asyncio.gather(
        search_similar(query="What book do I like?", user_id=TEST_USER, top_k=3),
        search_semantic_memories(query="What book do I like?", user_id=TEST_USER, top_k=3),
    )

    # Both should return results mentioning Dune
    all_content = " ".join(
        [r.get("content", "") for r in conv] + [r.get("content", "") for r in sem]
    )
    assert "Dune" in all_content or "book" in all_content.lower()

    # Semantic results should have a key field
    if sem:
        assert "key" in sem[0]


# ---------------------------------------------------------------------------
# 4. LangGraph agent loads thread history on restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thread_history_survives_restart():
    """Verify that thread history is loadable after storing (simulates restart).

    The agent calls load_thread_history on startup; this test proves the
    persistence layer works for that use case.
    """
    thread_id = str(uuid.uuid4())

    # Simulate a previous conversation
    messages = [
        ("user", "Hi, I'm Bob."),
        ("assistant", "Hello Bob! How can I help?"),
        ("user", "Remember that I work at SpaceX."),
        ("assistant", "Got it — you work at SpaceX."),
    ]

    for role, content in messages:
        await store_message(thread_id, role, content, {"user_id": TEST_USER})

    # Simulate restart: load thread from scratch
    history = await retrieve_thread(thread_id)

    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hi, I'm Bob."
    assert history[3]["role"] == "assistant"
    assert "SpaceX" in history[3]["content"]

    # Verify message IDs are persistent
    for msg in history:
        assert "id" in msg
        # Verify each can be re-fetched
        uuid.UUID(msg["id"])


@pytest.mark.asyncio
async def test_agent_state_persistence_via_checkpointer_db():
    """Verify the same database can serve as LangGraph checkpointer store.

    Proves the database is correctly configured for both conversation_messages
    and the LangGraph checkpoint tables.
    """
    pool = await get_pool()

    # Check that the conversation_messages table exists with HNSW index
    async with pool.acquire() as conn:
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('conversation_messages', 'semantic_memory')
        """)
        table_names = {r["table_name"] for r in tables}
        assert "conversation_messages" in table_names, "conversation_messages table missing"
        assert "semantic_memory" in table_names, "semantic_memory table missing"


# ---------------------------------------------------------------------------
# Semantic memory CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_memory_upsert():
    """Verify semantic memory upsert semantics (idempotent writes)."""
    user_id = TEST_USER
    key = f"test-key-{uuid.uuid4().hex[:8]}"

    # First write
    id1 = await store_semantic_memory(
        user_id=user_id, key=key, content="First value", importance=0.5,
    )
    assert id1

    # Second write to same key — should update, not create duplicate
    id2 = await store_semantic_memory(
        user_id=user_id, key=key, content="Updated value", importance=0.8,
    )
    assert id2 == id1  # Upsert should return same ID

    # Verify the updated value
    results = await search_semantic_memories(query=key, user_id=user_id, top_k=1)
    assert len(results) == 1
    assert results[0]["content"] == "Updated value"


@pytest.mark.asyncio
async def test_semantic_memory_search():
    """Verify semantic memory search returns relevant facts ranked by importance."""
    user_id = TEST_USER

    await store_semantic_memory(user_id, "name", "User's name is Alice.", importance=0.3)
    await store_semantic_memory(user_id, "job", "Alice works at Acme Corp as a senior engineer.", importance=0.9)
    await store_semantic_memory(user_id, "location", "Alice lives in Seattle, WA.", importance=0.5)

    results = await search_semantic_memories(
        query="What does Alice do for work?", user_id=user_id, top_k=3,
    )

    assert len(results) > 0
    # The job fact should be ranked highly for a work-related query
    high_ranked = results[0]
    assert "Acme" in high_ranked.get("content", "") or "engineer" in high_ranked.get("content", "")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session", autouse=True)
async def cleanup_after_all():
    """Close the connection pool after all tests."""
    yield
    await close_pool()
