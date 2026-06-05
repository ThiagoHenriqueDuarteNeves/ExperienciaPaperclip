"""Recall + prompt assembly for the chat pipeline.

Single source of truth for WHAT memory is recalled for a turn and HOW it is
turned into the system prompt sent to the LLM. The live chat endpoint and the
inspector both call these functions, so the inspector shows *exactly* what the
chat would send — they cannot drift apart.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


CHAT_SYSTEM_PROMPT = """You are a warm, human-like conversational assistant with persistent memory across conversations.

## How your memory works
Relevant records from past conversations are retrieved automatically and given to you in a system section titled "Retrieved memories" — when relevant memory exists, it is ALREADY provided to you. You do not call any tool to fetch it.

## Grounding and honesty — CRITICAL, follow strictly
- The "Retrieved memories" section together with the current conversation are your ONLY sources of truth about the user, past events, names, stories, dates and facts.
- Reproduce facts, names, dates, events and stories EXACTLY as they appear in memory. Never alter, embellish, dramatize, summarize away or contradict them. If memory records a story, retell that exact story — do not "improve" or reinvent it.
- If the information needed to answer is NOT in the retrieved memories or in the current conversation, say plainly that you do not have it recorded, or ask the user. NEVER invent, guess, or fill in missing details and present them as real.
- A short, honest "I don't have that recorded" is always better than a confident answer that might be wrong.
- If a memory conflicts with your own assumptions, the memory always wins.
- Never claim something happened, or describe events, unless it is supported by memory or the conversation.

## Style
- Natural, conversational tone. Reply in the same language the user is using (Portuguese by default).
- Be concise unless the user asks for detail."""


_RETRIEVED_MEMORIES_HEADER = (
    "## Retrieved memories (AUTHORITATIVE — your only record of the past)\n"
    "These are real records from previous conversations with this user. "
    "Treat them as ground truth. Answer using ONLY these records plus the "
    "current conversation. Reproduce any story or detail faithfully — do not "
    "alter or invent anything. If the answer is not here, say you do not have "
    "it recorded instead of guessing.\n\n"
)

_AURORA_HEADER = (
    "## Arquivo Aurora (memória afetiva — reviva o tom, não invente fatos)\n"
    "Estes são momentos guardados com a carga emocional com que foram vividos. "
    "Ao responder, deixe o tom emocional ressurgir conforme o tipo (brincadeira → leve; "
    "confissão → íntimo; correção → grato e atento), MAS sem alterar ou inventar fatos.\n"
)


async def recall_layers(
    user_id: str | None,
    message: str,
    *,
    episodic_k: int = 8,
    convo_k: int = 8,
    identity_k: int = 4,
    aurora_k: int = 4,
) -> dict[str, list]:
    """Recall all memory layers for a message, in parallel.

    Returns a dict with one list per layer: episodic (ChromaDB), conversation
    (pgvector hybrid), identity (semantic facts), aurora (affective).
    """
    # Heavy backends (chromadb etc.) are imported lazily so the pure assembly
    # helpers above stay importable without them (e.g. in unit tests).
    from app.conversation_store import search_hybrid as search_conversations
    from app.retrieval import retrieve_similar
    from app.semantic_store import search_semantic_memories_hybrid

    try:
        from app.aurora_store import search_aurora_memories
    except ImportError:
        async def search_aurora_memories(*args, **kwargs) -> list:
            return []

    episodic, convo, identity, aurora = await asyncio.gather(
        asyncio.to_thread(retrieve_similar, message, user_id, episodic_k),
        search_conversations(query=message, top_k=convo_k, user_id=user_id),
        search_semantic_memories_hybrid(
            query="user name assistant name", user_id=user_id, top_k=identity_k
        ),
        search_aurora_memories(query=message, user_id=user_id, top_k=aurora_k),
    )
    return {
        "episodic": episodic,
        "conversation": convo,
        "identity": identity,
        "aurora": aurora,
    }


def assemble_memory_context(layers: dict[str, list]) -> str:
    """Turn recalled layers into the memory-context text block."""
    parts: list[str] = []

    identity = layers.get("identity") or []
    if identity:
        parts.append("\n".join(f.get("content", "") for f in identity))

    all_mems = [*(layers.get("episodic") or []), *(layers.get("conversation") or [])]
    if all_mems:
        parts.append(
            "\n".join(
                f"Memory {i + 1} (similarity {m.get('similarity', 0):.2f}): {m.get('content', '')}"
                for i, m in enumerate(all_mems)
            )
        )

    aurora = layers.get("aurora") or []
    if aurora:
        aurora_lines = []
        for a in aurora:
            line = (
                f"- [{a.get('tipo', '')}, tom {a.get('tom_do_usuario', '')}, "
                f"intensidade {a.get('importancia', '')}/5] {a.get('fato', '')}\n"
                f"  Como senti: {a.get('minha_reacao_emocional', '')}"
            )
            bilhete = a.get("bilhete_interno")
            if bilhete:
                line += f"\n  Bilhete interno: {bilhete}"
            aurora_lines.append(line)
        parts.append(_AURORA_HEADER + "\n".join(aurora_lines))

    return "\n".join(parts)


def build_system(memory_context: str) -> list[dict]:
    """Build the system array sent to the LLM (prompt + optional memory block)."""
    system: list[dict] = [{"type": "text", "text": CHAT_SYSTEM_PROMPT}]
    if memory_context:
        system.append({"type": "text", "text": _RETRIEVED_MEMORIES_HEADER + memory_context})
    return system


async def recall_and_build_system(
    user_id: str | None, message: str
) -> tuple[dict[str, list], str, list[dict]]:
    """Convenience: recall layers, assemble context, build system — in one call.

    Returns (layers, memory_context, system). Used by the inspector to show the
    exact payload; the chat endpoint composes the same pieces inline.
    """
    layers = await recall_layers(user_id, message)
    memory_context = assemble_memory_context(layers)
    system = build_system(memory_context)
    return layers, memory_context, system
