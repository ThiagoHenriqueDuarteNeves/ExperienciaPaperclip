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


# Curation defaults — overridable per-deploy via MEMORY_RECALL_* env (read through
# getattr so the code is safe even when those config fields aren't present yet).
_RECALL_DEFAULTS = {
    "episodic_k": 4,
    "convo_k": 4,
    "identity_k": 4,
    "aurora_k": 3,
    "min_similarity": 0.7,
    "char_budget": 6000,
    # Fase C — cross-encoder re-ranking (opt-in).
    "rerank_enabled": False,
    "rerank_top_n": 6,
    "rerank_candidate_k": 12,
}


def _cfg(name: str):
    from app.config import settings

    return getattr(settings, f"recall_{name}", _RECALL_DEFAULTS[name])


def _gate(mems: list[dict], min_similarity: float) -> list[dict]:
    """Drop memories whose similarity is below the relevance threshold."""
    return [m for m in mems if float(m.get("similarity", 0) or 0) >= min_similarity]


def _near_duplicate(a: str, b: str, *, jaccard: float = 0.8) -> bool:
    """True if two texts are near-duplicates (containment or high word overlap)."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return False
    inter = len(sa & sb)
    union = len(sa | sb)
    return bool(union) and inter / union >= jaccard


def merge_and_dedup(episodic: list[dict], conversation: list[dict]) -> list[dict]:
    """Merge episodic + conversation, sort by similarity desc, drop near-duplicates.

    Episodic and conversation cover the same turns in different shapes, so this
    collapses the overlap and keeps the strongest representative of each memory.
    """
    merged = sorted(
        [*(episodic or []), *(conversation or [])],
        key=lambda m: float(m.get("similarity", 0) or 0),
        reverse=True,
    )
    kept: list[dict] = []
    for m in merged:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if any(_near_duplicate(content, k.get("content", "")) for k in kept):
            continue
        kept.append(m)
    return kept


def apply_budget(mems: list[dict], char_budget: int | None) -> list[dict]:
    """Keep memories (already ordered by relevance) until the char budget is hit."""
    if not char_budget:
        return mems
    out: list[dict] = []
    used = 0
    for m in mems:
        size = len(m.get("content", "") or "")
        if out and used + size > char_budget:
            break
        out.append(m)
        used += size
    return out


async def recall_layers(
    user_id: str | None,
    message: str,
    *,
    episodic_k: int | None = None,
    convo_k: int | None = None,
    identity_k: int | None = None,
    aurora_k: int | None = None,
) -> dict[str, list]:
    """Recall all memory layers for a message, in parallel.

    Returns a dict with one list per layer: episodic (ChromaDB), conversation
    (pgvector hybrid), identity (semantic facts), aurora (affective).
    """
    # Heavy backends (chromadb etc.) are imported lazily so the pure assembly
    # helpers above stay importable without them (e.g. in unit tests).
    from app.conversation_store import search_hybrid as search_conversations
    from app.retrieval import retrieve_similar
    from app.semantic_store import get_profile_facts

    try:
        from app.aurora_store import search_aurora_memories
    except ImportError:
        async def search_aurora_memories(*args, **kwargs) -> list:
            return []

    episodic_k = episodic_k if episodic_k is not None else _cfg("episodic_k")
    convo_k = convo_k if convo_k is not None else _cfg("convo_k")
    identity_k = identity_k if identity_k is not None else _cfg("identity_k")
    aurora_k = aurora_k if aurora_k is not None else _cfg("aurora_k")

    episodic, convo, identity, aurora = await asyncio.gather(
        asyncio.to_thread(retrieve_similar, message, user_id, episodic_k),
        search_conversations(query=message, top_k=convo_k, user_id=user_id),
        get_profile_facts(user_id=user_id, top_k=identity_k),
        search_aurora_memories(query=message, user_id=user_id, top_k=aurora_k),
    )
    return {
        "episodic": episodic,
        "conversation": convo,
        "identity": identity,
        "aurora": aurora,
    }


def assemble_memory_context(
    layers: dict[str, list],
    *,
    min_similarity: float = 0.0,
    char_budget: int | None = None,
    query: str | None = None,
    rerank: bool = False,
    rerank_top_n: int = 6,
) -> str:
    """Turn recalled layers into the memory-context text block.

    Curates rather than dumps: gates episodic/conversation by relevance, merges
    and de-duplicates the overlap, optionally cross-encoder re-ranks against the
    query (Fase C), and trims to a character budget. Identity and Aurora are kept
    (they are small / already top-k bounded).
    """
    parts: list[str] = []

    identity = layers.get("identity") or []
    if identity:
        parts.append("\n".join(f.get("content", "") for f in identity))

    episodic = _gate(layers.get("episodic") or [], min_similarity)
    conversation = _gate(layers.get("conversation") or [], min_similarity)
    all_mems = merge_and_dedup(episodic, conversation)
    if rerank and query:
        from app.reranker import rerank_memories

        all_mems = rerank_memories(query, all_mems, rerank_top_n)
    all_mems = apply_budget(all_mems, char_budget)
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
    rerank = bool(_cfg("rerank_enabled"))
    if rerank:
        # over-fetch candidates so the re-ranker has a real pool to choose from
        k = _cfg("rerank_candidate_k")
        layers = await recall_layers(user_id, message, episodic_k=k, convo_k=k)
    else:
        layers = await recall_layers(user_id, message)
    memory_context = assemble_memory_context(
        layers,
        min_similarity=_cfg("min_similarity"),
        char_budget=_cfg("char_budget"),
        query=message,
        rerank=rerank,
        rerank_top_n=_cfg("rerank_top_n"),
    )
    system = build_system(memory_context)
    return layers, memory_context, system
