"""Rebuild the Neo4j relationship graph from existing episodic memories.

Background: entity nodes were stored, but relationships never persisted (the
map-as-property bug in neo4j_client, now fixed). The graph ended up with N
entity nodes and ZERO edges. This script re-extracts entities + relationships
from the original conversation text (stored in the ChromaDB episodic collection)
and writes the edges that were lost.

How it works:
  1. Read every episodic document from ChromaDB (id + text + metadata).
  2. Re-run LLM extraction (extract_knowledge) on each text.
  3. upsert_entity + upsert_relationship for each result.

Safety:
  - Idempotent: entity/relationship writes use MERGE, so re-running does not
    duplicate. Existing entities keep their original source_id.
  - --dry-run extracts and reports what WOULD be written, touching nothing.
  - Each extraction calls the LLM, so a full run can be slow; use --limit to
    sample first.

Usage (inside the container):
  docker compose exec memory-api python -m app.scripts.reprocess_kg --dry-run --limit 5
  docker compose exec memory-api python -m app.scripts.reprocess_kg --dry-run
  docker compose exec memory-api python -m app.scripts.reprocess_kg            # apply
  docker compose exec memory-api python -m app.scripts.reprocess_kg --user thiago
"""

from __future__ import annotations

import argparse

from app.chroma_client import get_or_create_collection
from app.entity_extraction import extract_knowledge
from app.neo4j_client import (
    get_driver,
    health as neo4j_health,
    upsert_entity,
    upsert_relationship,
)


def _edge_count() -> int:
    """Total RELATES_TO edges currently in the graph."""
    with get_driver().session() as session:
        rec = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()
        return rec["c"] if rec else 0


def _fetch_memories(user_id: str | None, limit: int | None) -> list[tuple[str, str]]:
    """Return (memory_id, text) pairs from the ChromaDB episodic collection."""
    collection = get_or_create_collection()
    where = {"user_id": user_id} if user_id else None
    res = collection.get(where=where, include=["documents"])
    items = [
        (mid, doc)
        for mid, doc in zip(res.get("ids", []), res.get("documents", []))
        if doc
    ]
    return items[:limit] if limit else items


def reprocess(dry_run: bool, limit: int | None, user_id: str | None, timeout: float) -> int:
    if not neo4j_health():
        print("ERROR: Neo4j unreachable — start `docker compose up -d neo4j`")
        return 1

    items = _fetch_memories(user_id, limit)
    before = _edge_count()
    scope = f" (user={user_id})" if user_id else ""
    print(f"episodic memories to process{scope}: {len(items)}")
    print(f"edges before: {before}")
    print("mode:", "DRY-RUN (no writes)" if dry_run else "APPLY")
    print("-" * 60)

    total_ents = total_rels = created_rels = 0
    failed_docs = 0

    for i, (mem_id, text) in enumerate(items, 1):
        try:
            data = extract_knowledge(text, timeout=timeout)
        except Exception as exc:  # extraction is best-effort; never abort the run
            failed_docs += 1
            print(f"[{i}/{len(items)}] {mem_id[:8]} EXTRACTION FAILED: {exc}")
            continue

        ents = data["entities"]
        rels = data["relationships"]
        total_ents += len(ents)
        total_rels += len(rels)

        if dry_run:
            print(f"[{i}/{len(items)}] {mem_id[:8]} -> {len(ents)} ents, {len(rels)} rels")
            for r in rels[:4]:
                print(f"        {r['source']} -[{r['type']}]-> {r['target']}  props={r.get('properties') or {}}")
        else:
            for e in ents:
                upsert_entity(e["name"], e["type"], e.get("description", ""), source_id=mem_id)
            for r in rels:
                res = upsert_relationship(
                    r["source"], r["target"], r["type"], r.get("properties")
                )
                if res is not None:
                    created_rels += 1
            print(f"[{i}/{len(items)}] {mem_id[:8]} -> +{len(rels)} rels")

    print("-" * 60)
    print(f"docs processed: {len(items)}  (extraction failures: {failed_docs})")
    print(f"entities seen: {total_ents} | relationships seen: {total_rels}")
    if dry_run:
        print("DRY-RUN complete — nothing written to Neo4j.")
    else:
        after = _edge_count()
        print(f"relationships upserted OK: {created_rels}")
        print(f"edges: {before} -> {after}  (+{after - before})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Neo4j relationships from episodic memories."
    )
    parser.add_argument("--dry-run", action="store_true", help="extract and report only; no writes")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N memories")
    parser.add_argument("--user", default=None, help="restrict to a single user_id")
    parser.add_argument(
        "--timeout", type=float, default=90.0,
        help="per-extraction HTTP timeout in seconds (default 90; raise for slow local models)",
    )
    args = parser.parse_args()
    raise SystemExit(
        reprocess(dry_run=args.dry_run, limit=args.limit, user_id=args.user, timeout=args.timeout)
    )


if __name__ == "__main__":
    main()
