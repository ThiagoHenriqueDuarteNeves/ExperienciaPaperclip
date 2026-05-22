"""LangGraph agent skeleton with Postgres checkpointer and Claude API tool use."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from app.config import settings


class AgentState(TypedDict):
    thread_id: str
    user_id: str
    messages: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    agent_response: str | None


_graph: CompiledStateGraph | None = None
_checkpointer: AsyncPostgresSaver | None = None
_store: InMemoryStore | None = None


def _pg_config() -> dict[str, Any]:
    return {
        "host": settings.pgvector_host,
        "port": settings.pgvector_port,
        "user": settings.pgvector_user,
        "password": settings.pgvector_password,
        "database": settings.pgvector_database,
    }


async def get_checkpointer() -> AsyncPostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        conn_str = (
            f"postgresql://{settings.pgvector_user}:{settings.pgvector_password}"
            f"@{settings.pgvector_host}:{settings.pgvector_port}/{settings.pgvector_database}"
        )
        _checkpointer = AsyncPostgresSaver.from_conn_string(conn_str)
        await _checkpointer.setup()
    return _checkpointer


def get_store() -> InMemoryStore:
    global _store
    if _store is None:
        _store = InMemoryStore()
    return _store


def _build_graph() -> CompiledStateGraph:
    builder = StateGraph(AgentState)

    async def retrieve_context(state: AgentState) -> dict[str, Any]:
        from app.pgvector_client import load_thread, search_similar
        from app.embeddings import embed_text_async

        thread_history = await load_thread(state["thread_id"])
        last_msg = state["messages"][-1]["content"] if state["messages"] else ""
        retrieved = []
        if last_msg:
            embedding = await embed_text_async(last_msg)
            retrieved = await search_similar(embedding, top_k=settings.similarity_top_k)
        return {
            "retrieved_context": retrieved,
            "messages": thread_history,
        }

    async def generate_response(state: AgentState) -> dict[str, Any]:
        import httpx

        api_key = settings.effective_claude_api_key
        if not api_key:
            return {"agent_response": "Error: Claude API key not configured"}

        context_str = "\n".join(
            m.get("content", "") for m in state.get("retrieved_context", [])
        )
        system_prompt = (
            f"You are a helpful AI assistant with memory.\n"
            f"Relevant past context:\n{context_str}\n"
            f"Thread history:\n"
            "; ".join(f"{m['role']}: {m['content']}" for m in state.get("messages", []))
        )

        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.claude_model,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": state["messages"][-1]["content"]}
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return {"agent_response": data["content"][0]["text"]}

    async def store_turn(state: AgentState) -> dict[str, Any]:
        if state.get("agent_response"):
            from app.pgvector_client import store_message
            from app.embeddings import embed_text_async

            msg_id = str(uuid.uuid4())
            embedding = await embed_text_async(state["agent_response"])
            await store_message(
                thread_id=state["thread_id"],
                role="assistant",
                content=state["agent_response"],
                embedding=embedding,
                metadata={"user_id": state["user_id"], "msg_id": msg_id},
            )
        return {}

    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_response", generate_response)
    builder.add_node("store_turn", store_turn)

    builder.set_entry_point("retrieve_context")
    builder.add_edge("retrieve_context", "generate_response")
    builder.add_edge("generate_response", "store_turn")
    builder.add_edge("store_turn", END)

    return builder.compile()


async def get_graph() -> CompiledStateGraph:
    global _graph
    if _graph is None:
        _graph = _build_graph()
        _graph.checkpointer = await get_checkpointer()
        _graph.store = get_store()
    return _graph


async def run_agent(
    thread_id: str,
    user_id: str,
    user_message: str,
) -> dict[str, Any]:
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    from app.pgvector_client import store_message
    from app.embeddings import embed_text_async

    embedding = await embed_text_async(user_message)
    await store_message(
        thread_id=thread_id, role="user", content=user_message,
        embedding=embedding, metadata={"user_id": user_id},
    )

    result = await graph.ainvoke(
        {"thread_id": thread_id, "user_id": user_id,
         "messages": [{"role": "user", "content": user_message}],
         "retrieved_context": [], "agent_response": None},
        config,
    )
    return {
        "thread_id": thread_id,
        "response": result.get("agent_response", ""),
        "retrieved_context": result.get("retrieved_context", []),
    }


async def close() -> None:
    global _checkpointer, _graph, _store
    if _checkpointer is not None:
        await _checkpointer.close()
        _checkpointer = None
    _graph = None
    _store = None
