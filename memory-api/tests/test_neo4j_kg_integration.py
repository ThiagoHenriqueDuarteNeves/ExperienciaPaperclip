"""Integration test: relationships with map properties actually persist.

Reproduces the original failure (a relationship carrying a nested-map property)
against a live Neo4j and proves the edge is now created and round-trips.

Requires Neo4j up:  docker compose up -d neo4j
Run with:           pytest -m integration
"""

from __future__ import annotations

import uuid

import pytest

from app.neo4j_client import (
    get_relationships,
    health,
    upsert_entity,
    upsert_relationship,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def neo4j_check():
    if not health():
        pytest.skip("Neo4j not reachable — start `docker compose up -d neo4j`")
    yield


def _cleanup(driver_names: list[str]):
    from app.neo4j_client import get_driver

    with get_driver().session() as s:
        for name in driver_names:
            s.run("MATCH (e:Entity {name: $name}) DETACH DELETE e", name=name)


def test_relationship_with_map_properties_persists():
    src = f"TestPerson-{uuid.uuid4().hex[:8]}"
    tgt = f"TestCompany-{uuid.uuid4().hex[:8]}"
    try:
        upsert_entity(src, "person", "a test person")
        upsert_entity(tgt, "organization", "a test company")

        # The nested-map property that used to crash Neo4j.
        result = upsert_relationship(
            source_name=src,
            target_name=tgt,
            rel_type="works_at",
            properties={"duration": "12 anos", "years_experience": 5},
        )

        # The call now returns the created relationship, not None.
        assert result is not None
        assert result["type"] == "works_at"
        assert result["properties"]["duration"] == "12 anos"
        assert result["properties"]["years_experience"] == 5

        # And the edge is actually queryable from the graph.
        rels = get_relationships(src, direction="outgoing")
        assert any(
            r["rel_type"] == "works_at"
            and r["target_name"] == tgt
            and r["properties"].get("duration") == "12 anos"
            for r in rels
        ), "relationship not found in graph after upsert"
    finally:
        _cleanup([src, tgt])


def test_relationship_without_properties_also_persists():
    src = f"TestA-{uuid.uuid4().hex[:8]}"
    tgt = f"TestB-{uuid.uuid4().hex[:8]}"
    try:
        upsert_entity(src, "concept", "")
        upsert_entity(tgt, "concept", "")
        result = upsert_relationship(src, tgt, "relates_to", properties=None)
        assert result is not None
        assert result["properties"] == {}
        rels = get_relationships(src, direction="outgoing")
        assert any(r["target_name"] == tgt for r in rels)
    finally:
        _cleanup([src, tgt])
