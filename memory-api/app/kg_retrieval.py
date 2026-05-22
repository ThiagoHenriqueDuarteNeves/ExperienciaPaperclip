"""Knowledge graph retrieval and hybrid search layer."""

from app.chroma_client import get_or_create_collection
from app.config import settings
from app.embeddings import embed_text
from app.entity_extraction import extract_knowledge
from app.neo4j_client import (
    find_connected_entities,
    get_entity,
    get_relationships,
    search_entities,
    upsert_entity,
    upsert_relationship,
)


def store_knowledge(
    text: str,
    source_id: str | None = None,
) -> dict:
    """Extract entities and relationships from text and store in Neo4j."""
    extracted = extract_knowledge(text)
    entity_count = 0
    rel_count = 0

    for ent in extracted["entities"]:
        upsert_entity(
            name=ent["name"],
            type=ent["type"],
            description=ent["description"],
            source_id=source_id,
        )
        entity_count += 1

    for rel in extracted["relationships"]:
        upsert_relationship(
            source_name=rel["source"],
            target_name=rel["target"],
            rel_type=rel["type"],
            properties=rel.get("properties"),
        )
        rel_count += 1

    return {
        "entities_extracted": entity_count,
        "relationships_extracted": rel_count,
    }


def augment_with_graph_context(
    query: str,
    top_k: int | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """Retrieve similar memories from ChromaDB and augment them with graph context.

    For each retrieved memory, finds related entities in Neo4j and attaches
    the graph neighborhood as context.
    """
    embedding = embed_text(query)
    collection = get_or_create_collection()

    where = None
    if user_id:
        where = {"user_id": user_id}

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k or settings.similarity_top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    enriched = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        if similarity < settings.min_similarity_threshold:
            continue

        content = results["documents"][0][i]
        memory = {
            "id": results["ids"][0][i],
            "content": content,
            "metadata": results["metadatas"][0][i],
            "similarity": round(similarity, 4),
        }

        # Find entities mentioned in this memory content
        try:
            entities = search_entities(query=content[:200])
            graph_context = []
            for entity in entities[:3]:
                connected = find_connected_entities(
                    entity["name"], max_depth=1
                )
                if connected:
                    graph_context.append({
                        "entity": entity["name"],
                        "entity_type": entity["type"],
                        "connected_to": [
                            {"name": c["name"], "type": c["type"]}
                            for c in connected[:5]
                        ],
                    })
            if graph_context:
                memory["graph_context"] = graph_context
        except Exception:
            memory["graph_context"] = []

        enriched.append(memory)

    return enriched


def query_graph(
    entity_name: str,
    depth: int = 2,
) -> dict:
    """Query the knowledge graph centered on an entity."""
    entity = get_entity(entity_name)
    if not entity:
        return {"entity": None, "relationships": [], "connected": []}

    relationships = get_relationships(entity_name, direction="both")
    connected = find_connected_entities(entity_name, max_depth=depth)

    return {
        "entity": entity,
        "relationships": relationships,
        "connected": connected,
    }


def search_graph(
    query: str,
    type_filter: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search entities in the knowledge graph by name/description."""
    return search_entities(query=query, type_filter=type_filter, limit=limit)


def get_entity_graph(name: str, depth: int = 2) -> dict | None:
    """Full graph neighborhood for an entity: its data + relationships +
    connected entities with their types."""
    entity = get_entity(name)
    if not entity:
        return None

    rels = get_relationships(name, direction="both")
    connected = find_connected_entities(name, max_depth=depth)

    return {
        "entity": entity,
        "relationships": rels,
        "connected_entities": connected,
    }
