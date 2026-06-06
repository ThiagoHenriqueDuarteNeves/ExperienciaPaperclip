"""Chat endpoint — the full pipeline: recall → LLM stream → persist.

Recall + system assembly live in app.chat_pipeline (shared with the inspector),
so what is streamed here == what /inspect shows. Affective (Aurora) persistence
reuses extract_aurora / store_aurora_memory from app.routers.aurora.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.auth import user_from_authorization
from app.chat_pipeline import build_system, recall_and_build_system
from app.conversation_store import store_message
from app.llm_client import stream_text
from app.rate_limit import limiter
from app.retrieval import store_conversation
from app.routers.aurora import extract_aurora, store_aurora_memory
from app.semantic_store import store_semantic_memory

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/api/chat")
@limiter.limit("20/minute")
async def api_chat_endpoint(request: Request):
    """Full chat pipeline: recall memories, stream LLM response, persist turn."""
    body = await request.json()
    messages = body.get("messages", [])
    # Authenticated multi-user: user_id is derived from the signed token, not a
    # trusted header — this is what actually protects one user's memories from
    # another. No token → 401 (no x-user-id fallback, by design).
    user_id = user_from_authorization(request.headers.get("authorization"))
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    conversation_id = request.headers.get("x-conversation-id", "default")

    latest_user_message: str | None = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )

    async def generate():
        final_text = ""
        try:
            # 1. Recall memories + 2. build system — shared with the inspector
            #    (app.chat_pipeline) so what is sent here == what /inspect shows,
            #    including the relevance gating / dedup / budget curation.
            system = build_system("")
            if latest_user_message:
                try:
                    _, _, system = await recall_and_build_system(
                        user_id, latest_user_message
                    )
                except Exception as mem_err:
                    logger.warning("chat: memory recall failed: %s", mem_err)
                    system = build_system("")

            # 3. Fire-and-forget: store user message
            if latest_user_message:
                asyncio.create_task(
                    store_message(
                        thread_id=conversation_id,
                        role="user",
                        content=latest_user_message,
                        metadata={"user_id": user_id},
                    )
                )

            # 4. Stream from LLM via the unified client; re-wrap each chunk into
            #    our browser-facing SSE protocol.
            async for chunk in stream_text(
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                system=system,
                max_tokens=4096,
            ):
                final_text += chunk
                yield f"data: {json.dumps({'event': 'text', 'content': chunk})}\n\n"

            yield f"data: {json.dumps({'event': 'done', 'text': final_text})}\n\n"

            # 5. Fire-and-forget: persist assistant reply
            final_reply = final_text.strip()
            if latest_user_message and final_reply:
                async def _persist():
                    try:
                        await store_message(
                            thread_id=conversation_id,
                            role="assistant",
                            content=final_reply,
                            metadata={"user_id": user_id},
                        )
                    except Exception as e:
                        logger.warning("chat: store assistant msg failed: %s", e)
                    try:
                        await asyncio.to_thread(
                            store_conversation,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            content=f"User: {latest_user_message}\nAssistant: {final_reply}",
                            metadata={"source": "chat"},
                            extract_knowledge_graph=True,
                        )
                    except Exception as e:
                        logger.warning("chat: episodic store failed: %s", e)
                    try:
                        record = await asyncio.to_thread(
                            extract_aurora, latest_user_message, final_reply
                        )
                        if record:
                            await store_aurora_memory(user_id=user_id, **record)
                            logger.info("chat: aurora record stored (tipo=%s)", record.get("tipo"))
                    except Exception as e:
                        logger.warning("chat: aurora extraction failed: %s", e)
                    try:
                        from app.config import settings
                        from app.semantic_extractor import extract_semantic_facts

                        # Skip trivial turns ("ok", "kkkk", "valeu"…) — they rarely
                        # carry durable facts, so don't spend an LLM call on them.
                        min_chars = getattr(settings, "chat_semantic_min_chars", 30)
                        if len((latest_user_message or "").strip()) >= min_chars:
                            facts = await asyncio.to_thread(
                                extract_semantic_facts, latest_user_message, final_reply
                            )
                            for f in facts:
                                await store_semantic_memory(
                                    user_id=user_id,
                                    key=f["key"],
                                    content=f["content"],
                                    importance=f["importance"],
                                )
                            if facts:
                                logger.info("chat: %d semantic fact(s) stored", len(facts))
                    except Exception as e:
                        logger.warning("chat: semantic extraction failed: %s", e)

                asyncio.create_task(_persist())

        except Exception as exc:
            logger.exception("chat endpoint error")
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
