"""System / meta routes: MCP server info and the not-yet-implemented stubs
(LangGraph agent, conversation loop) that return 501.

These carry no business logic — they're the API's self-description and explicit
placeholders for future work, grouped out of main.py.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas import AgentChatRequest, ConversationRequest

router = APIRouter(tags=["system"])


# --- MCP server info ---


@router.get("/mcp/info")
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


# --- LangGraph agent endpoints — NOT YET IMPLEMENTED (501) ---


@router.post("/agent/chat")
async def agent_chat_endpoint(req: AgentChatRequest):
    return JSONResponse(
        status_code=501,
        content={"detail": "LangGraph agent not yet implemented"},
    )


@router.get("/agent/threads/{thread_id}")
async def agent_thread_endpoint(thread_id: str):
    return JSONResponse(
        status_code=501,
        content={"detail": "LangGraph agent not yet implemented"},
    )


# --- Conversation loop — NOT YET IMPLEMENTED (501) ---


@router.post("/conversation")
async def conversation_endpoint(req: ConversationRequest):
    return JSONResponse(
        status_code=501,
        content={"detail": "Conversation loop not yet implemented"},
    )
