"""Conversation loop orchestrator with Claude API tool-use.

Full pipeline:
  1. Context retrieval (episodic via ChromaDB + semantic via Neo4j)
  2. Claude API call with memory tool definitions and prompt caching
  3. Tool execution (store, retrieve, graph operations, Letta archival)
  4. Result storage and entity extraction

The orchestrator runs a tool-use loop: Claude may call multiple tools
in sequence before producing a final text response.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx

from app.chroma_client import get_or_create_collection
from app.config import settings
from app.embeddings import embed_text
from app.entity_extraction import extract_knowledge
from app.retrieval import retrieve_similar, store_conversation as store_episodic

try:
    from app.kg_retrieval import augment_with_graph_context, search_graph
    _KG_AVAILABLE = True
except ImportError:
    _KG_AVAILABLE = False

    def search_graph(**kwargs) -> list:
        return []

    def augment_with_graph_context(**kwargs) -> list:
        return []

try:
    from app.letta_manager import (
        get_client as get_letta_client,
        insert_archival_memory,
        search_archival_memory,
    )
    _LETTA_AVAILABLE = True
except ImportError:
    _LETTA_AVAILABLE = False

    def insert_archival_memory(*args, **kwargs):
        return None

    def search_archival_memory(*args, **kwargs):
        return []


# ---------------------------------------------------------------------------
# Memory tool definitions (Claude tool-use schema)
# ---------------------------------------------------------------------------

MEMORY_TOOLS = [
    {
        "name": "remember_conversation",
        "description": (
            "Store a conversation or important fact in episodic memory. "
            "Use this when the user shares something worth remembering — "
            "preferences, decisions, personal facts, project context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or conversation snippet to remember.",
                },
                "category": {
                    "type": "string",
                    "enum": ["fact", "preference", "decision", "context", "general"],
                    "description": "Category for the stored memory.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_memories",
        "description": (
            "Search episodic memory for semantically similar past conversations "
            "or facts. Use this to recall what the user previously mentioned "
            "about a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_knowledge_graph",
        "description": (
            "Search the semantic knowledge graph for entities, concepts, "
            "or relationships. Use this to look up structured knowledge "
            "about people, technologies, projects, or organizations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Entity or concept to look up.",
                },
                "type_filter": {
                    "type": "string",
                    "description": "Optional entity type filter (person, technology, etc.).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "store_archival_memory",
        "description": (
            "Store a detailed passage in the Letta procedural / archival memory "
            "for an agent. Use this for long-term knowledge that should persist "
            "across sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Detailed passage to archive.",
                },
            },
            "required": ["content"],
        },
    },
]

SYSTEM_PROMPT = """You are a human-like conversational AI with a 4-tier memory system.

## Your Memory Architecture

1. **Working Memory** — The current conversation (provided as messages).
2. **Episodic Memory** (ChromaDB) — Past conversations retrieved by semantic similarity. Use `recall_memories` to search and `remember_conversation` to store.
3. **Semantic Memory** (Neo4j Knowledge Graph) — Structured entities and relationships. Use `search_knowledge_graph` to look up structured knowledge.
4. **Procedural Memory** (Letta/MemGPT) — Long-term archival storage managed autonomously by Letta. Use `store_archival_memory` for facts that should persist indefinitely.

## Guidelines

- Before answering, check if relevant memories exist using `recall_memories`.
- When the user shares a fact, preference, or decision, store it with `remember_conversation`.
- Use `search_knowledge_graph` for entity-specific questions.
- Maintain a natural conversational tone. Do not mention your memory system unless asked.
- Only use tool calls when genuinely useful — don't force them.
- If you don't recall something, be honest rather than fabricating."""


def _build_tool_cache_breakpoint() -> list[dict]:
    """Return Claude-compatible tool definitions with cache_control breakpoints.

    The last tool in the array carries an ephemeral cache_control breakpoint
    so the tools block is cached across requests.
    """
    tools = [dict(t) for t in MEMORY_TOOLS]
    if tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _build_system_message() -> dict:
    """Build the system message with ephemeral cache breakpoint."""
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------


async def _execute_remember(user_id: str, conversation_id: str, args: dict) -> str:
    """Execute the remember_conversation tool."""
    content = args["content"]
    category = args.get("category", "general")
    memory_id = store_episodic(
        user_id=user_id,
        conversation_id=conversation_id,
        content=content,
        metadata={"category": category, "source": "tool_call"},
    )
    return json.dumps({"stored": True, "memory_id": memory_id})


async def _execute_recall(user_id: str, args: dict) -> str:
    """Execute the recall_memories tool."""
    query = args["query"]
    top_k = args.get("top_k", 5)
    results = retrieve_similar(query=query, user_id=user_id, top_k=top_k)
    return json.dumps({"memories": results, "count": len(results)})


async def _execute_graph_search(args: dict) -> str:
    """Execute the search_knowledge_graph tool."""
    query = args["query"]
    type_filter = args.get("type_filter")
    results = search_graph(query=query, type_filter=type_filter)
    return json.dumps({"entities": results, "count": len(results)})


async def _execute_store_archival(agent_id: str | None, args: dict) -> str:
    """Execute the store_archival_memory tool."""
    if not _LETTA_AVAILABLE or agent_id is None:
        return json.dumps({"stored": False, "error": "Letta is not available"})
    content = args["content"]
    result = insert_archival_memory(agent_id=agent_id, content=content)
    return json.dumps({"stored": result is not None})


_TOOL_EXECUTORS = {
    "remember_conversation": _execute_remember,
    "recall_memories": _execute_recall,
    "search_knowledge_graph": _execute_graph_search,
    "store_archival_memory": _execute_store_archival,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_conversation_loop(
    user_id: str,
    conversation_id: str,
    user_message: str,
    history: list[dict] | None = None,
    agent_id: str | None = None,
    max_tool_rounds: int = 5,
) -> AsyncGenerator[dict, None]:
    """Run the full conversation loop and yield SSE-style events.

    Parameters
    ----------
    user_id : str
        The user identifier for memory scoping.
    conversation_id : str
        The conversation session identifier.
    user_message : str
        The latest user message.
    history : list[dict] | None
        Prior conversation turns as ``[{"role": "user"|"assistant", "content": "..."}]``.
    agent_id : str | None
        Optional Letta agent ID for archival memory operations.
    max_tool_rounds : int
        Maximum number of back-and-forth tool-calling rounds.

    Yields
    ------
    dict
        SSE events: ``{"event": "context", ...}``, ``{"event": "text", ...}``,
        ``{"event": "tool_call", ...}``, ``{"event": "done", ...}``,
        or ``{"event": "error", ...}``.
    """
    api_key = settings.effective_claude_api_key
    if not api_key:
        yield {"event": "error", "message": "Claude API key not configured"}
        return

    # 1. Retrieve context
    episodic_context = _retrieve_episodic(query=user_message, user_id=user_id)
    graph_context = _retrieve_graph_context(query=user_message, user_id=user_id)

    yield {
        "event": "context",
        "episodic": episodic_context,
        "graph": graph_context,
    }

    # 2. Build messages for Claude
    system_msg = _build_system_message()
    tools = _build_tool_cache_breakpoint()

    messages: list[dict] = []
    for ctx in episodic_context:
        messages.append({
            "role": "system",
            "content": (
                f"[Relevant memory — similarity {ctx.get('similarity', '?.??')}]: "
                f"{ctx['content']}"
            ),
        })
    for entity in graph_context:
        messages.append({
            "role": "system",
            "content": (
                f"[Knowledge graph entity: {entity.get('name', 'unknown')} "
                f"({entity.get('type', 'unknown')})]: {entity.get('description', '')}"
            ),
        })

    # Add conversation history (last 20 turns)
    for turn in (history or [])[-20:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_message})

    # 3. Tool-use loop
    tool_round = 0
    final_text = ""
    all_tool_calls: list[dict] = []

    async with httpx.AsyncClient(timeout=60) as http:
        while tool_round < max_tool_rounds:
            tool_round += 1
            response = await _call_claude(http, api_key, system_msg, tools, messages)

            if response is None:
                yield {"event": "error", "message": "Claude API call failed"}
                return

            stop_reason = response.get("stop_reason", "")
            content_blocks = response.get("content", [])

            # Process content blocks
            text_parts: list[str] = []
            tool_requests: list[dict] = []

            for block in content_blocks:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    text_parts.append(text)
                    yield {"event": "text", "content": text}
                elif block.get("type") == "tool_use":
                    tool_requests.append(block)

            # If Claude produced a final text response, we're done
            if stop_reason == "end_turn":
                final_text = "".join(text_parts)
                break

            # If Claude wants to use tools, execute them
            if stop_reason == "tool_use" and tool_requests:
                # Add assistant message
                messages.append({"role": "assistant", "content": content_blocks})

                tool_results: list[dict] = []
                for tool_req in tool_requests:
                    tool_name = tool_req.get("name", "")
                    tool_input = tool_req.get("input", {})
                    tool_id = tool_req.get("id", "")

                    yield {
                        "event": "tool_call",
                        "name": tool_name,
                        "input": tool_input,
                    }

                    executor = _TOOL_EXECUTORS.get(tool_name)
                    if executor:
                        if tool_name in ("remember_conversation", "recall_memories"):
                            result = await executor(user_id, conversation_id, tool_input)
                        elif tool_name == "store_archival_memory":
                            result = await executor(agent_id, tool_input)
                        else:
                            result = await executor(tool_input)
                    else:
                        result = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                    })
                    all_tool_calls.append({
                        "name": tool_name,
                        "input": tool_input,
                        "result": result,
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool requests and no end_turn — unexpected
                break

    # 4. Store the conversation
    try:
        store_episodic(
            user_id=user_id,
            conversation_id=conversation_id,
            content=f"User: {user_message}\nAssistant: {final_text}",
            metadata={"source": "conversation_loop"},
        )
    except Exception:
        pass

    # 5. Extract knowledge graph entities (best-effort)
    try:
        extract_knowledge(f"User: {user_message}\nAssistant: {final_text}")
    except Exception:
        pass

    yield {
        "event": "done",
        "text": final_text,
        "tool_calls": all_tool_calls,
        "tool_rounds": tool_round,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieve_episodic(query: str, user_id: str) -> list[dict]:
    """Retrieve relevant episodic memories."""
    try:
        return retrieve_similar(query=query, user_id=user_id, top_k=5)
    except Exception:
        return []


def _retrieve_graph_context(query: str, user_id: str | None) -> list[dict]:
    """Retrieve relevant semantic graph context."""
    if not _KG_AVAILABLE:
        return []
    try:
        return search_graph(query=query, limit=5)
    except Exception:
        return []


async def _call_claude(
    http: httpx.AsyncClient,
    api_key: str,
    system_msg: dict,
    tools: list[dict],
    messages: list[dict],
) -> dict | None:
    """Send a request to the Claude API and return the parsed response."""
    try:
        api_url = f"{settings.effective_llm_api_base}/messages"
        resp = await http.post(
            api_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.claude_model,
                "max_tokens": 4096,
                "system": [system_msg],
                "tools": tools,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPStatusError, httpx.TimeoutException, json.JSONDecodeError):
        return None
