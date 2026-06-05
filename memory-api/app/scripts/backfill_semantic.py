"""Backfill the semantic profile from existing episodic memories.

Phase B wires semantic-fact extraction into the chat going forward, but existing
users' profiles stay empty until they chat again. This re-extracts durable facts
from the ChromaDB episodic documents and upserts them into semantic_memory.

Idempotent: facts use canonical snake_case keys, so repeated/duplicate facts
collapse via upsert (the profile stays compact, not bloated).

Usage (inside the container):
  docker compose exec memory-api python -m app.scripts.backfill_semantic --dry-run --limit 5
  docker compose exec memory-api python -m app.scripts.backfill_semantic --user thiago
  docker compose exec memory-api python -m app.scripts.backfill_semantic            # all users
"""

from __future__ import annotations

import argparse
import asyncio

from app.pgvector_client import get_pool
from app.semantic_extractor import extract_semantic_facts
from app.semantic_store import store_semantic_memory


def _split_turn(doc: str) -> tuple[str, str]:
    """Split a stored episodic doc ('User: X\\nAssistant: Y') into (user, assistant).

    Avoids feeding the whole doc as the user message with an empty reply, which
    would double the speaker labels in the extraction prompt.
    """
    marker = "\nAssistant:"
    if marker in doc:
        user_part, assistant_part = doc.split(marker, 1)
        user_part = user_part.split("User:", 1)[-1].strip()
        return user_part, assistant_part.strip()
    return doc.strip(), ""


def _fetch(user_id: str | None, limit: int | None) -> list[tuple[str, str]]:
    from app.chroma_client import get_or_create_collection

    collection = get_or_create_collection()
    where = {"user_id": user_id} if user_id else None
    res = collection.get(where=where, include=["documents", "metadatas"])
    items = [
        ((m or {}).get("user_id"), doc)
        for (m, doc) in zip(res.get("metadatas", []), res.get("documents", []))
        if doc
    ]
    return items[:limit] if limit else items


async def _count(user_id: str | None) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM semantic_memory WHERE ($1::text IS NULL OR user_id=$1)",
            user_id,
        )


async def backfill(dry_run: bool, limit: int | None, user_id: str | None, timeout: float) -> int:
    items = _fetch(user_id, limit)
    before = await _count(user_id)
    scope = f" (user={user_id})" if user_id else ""
    print(f"episodic memories to process{scope}: {len(items)}")
    print(f"semantic facts before: {before}")
    print("mode:", "DRY-RUN (no writes)" if dry_run else "APPLY")
    print("-" * 60)

    seen_facts = 0
    failed = 0
    for i, (uid, text) in enumerate(items, 1):
        # Facts are about the user who owns the memory.
        owner = user_id or uid
        if not owner:
            continue
        try:
            user_part, assistant_part = _split_turn(text)
            facts = await asyncio.to_thread(extract_semantic_facts, user_part, assistant_part)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(items)}] EXTRACTION FAILED: {exc}")
            continue
        seen_facts += len(facts)
        if dry_run:
            label = ", ".join(f"{f['key']}={f['content'][:30]}" for f in facts) or "—"
            print(f"[{i}/{len(items)}] {owner}: {len(facts)} fato(s)  {label}")
        else:
            for f in facts:
                await store_semantic_memory(
                    user_id=owner, key=f["key"], content=f["content"], importance=f["importance"]
                )
            print(f"[{i}/{len(items)}] {owner}: +{len(facts)} fato(s)")

    after = await _count(user_id)
    print("-" * 60)
    print(f"docs processed: {len(items)} (failures: {failed}) | facts seen: {seen_facts}")
    if dry_run:
        print("DRY-RUN — nothing written.")
    else:
        print(f"semantic facts: {before} -> {after} (canonical keys collapse duplicates)")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill semantic profile from episodic memory.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--timeout", type=float, default=90.0)
    a = p.parse_args()
    raise SystemExit(asyncio.run(backfill(a.dry_run, a.limit, a.user, a.timeout)))


if __name__ == "__main__":
    main()
