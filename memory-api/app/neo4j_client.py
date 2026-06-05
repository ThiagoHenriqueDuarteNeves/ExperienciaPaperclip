import json

from neo4j import GraphDatabase, Driver

from app.config import settings

_driver: Driver | None = None


def _props_to_storage(properties: dict | None) -> str:
    """Serialize relationship properties for Neo4j storage.

    Neo4j property values must be primitives or arrays of primitives — a nested
    map (e.g. {"duration": "12 anos"}) is rejected with a 22N01 type error. We
    therefore store the whole properties bag as a JSON string.
    """
    return json.dumps(properties or {})


def _props_from_storage(value) -> dict:
    """Inverse of _props_to_storage. Tolerant of legacy/plain values."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return {}
    if isinstance(value, dict):
        return value
    return {}


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def health() -> bool:
    try:
        driver = get_driver()
        driver.verify_connectivity()
        return True
    except Exception:
        return False


def init_schema() -> None:
    """Create constraints and indexes for the knowledge graph."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)"
        )


def upsert_entity(
    name: str,
    type: str,
    description: str | None = None,
    source_id: str | None = None,
) -> dict:
    """Create or update an entity node. Returns the entity dict."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MERGE (e:Entity {name: $name})
            ON CREATE SET
                e.id = randomUUID(),
                e.type = $type,
                e.description = $description,
                e.source_id = $source_id,
                e.created_at = datetime(),
                e.updated_at = datetime()
            ON MATCH SET
                e.type = CASE WHEN $type IS NOT NULL THEN $type ELSE e.type END,
                e.description = CASE WHEN $description IS NOT NULL
                    THEN $description ELSE e.description END,
                e.updated_at = datetime()
            RETURN e.id AS id, e.name AS name, e.type AS type,
                   e.description AS description, e.source_id AS source_id
            """,
            name=name,
            type=type,
            description=description,
            source_id=source_id,
        )
        record = result.single()
        return dict(record) if record else {}


def get_entity(name: str) -> dict | None:
    """Look up an entity by name."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.id AS id, e.name AS name, e.type AS type,
                   e.description AS description, e.source_id AS source_id,
                   e.created_at AS created_at, e.updated_at AS updated_at
            """,
            name=name,
        )
        record = result.single()
        return dict(record) if record else None


def search_entities(
    query: str,
    type_filter: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search entities by name or description (substring)."""
    driver = get_driver()
    with driver.session() as session:
        cypher = """
            MATCH (e:Entity)
            WHERE (e.name CONTAINS $query OR e.description CONTAINS $query)
        """
        params = {"query": query, "limit": limit}
        if type_filter:
            cypher += " AND e.type = $type_filter"
        cypher += """
            RETURN e.id AS id, e.name AS name, e.type AS type,
                   e.description AS description
            ORDER BY e.name
            LIMIT $limit
        """
        params["type_filter"] = type_filter
        result = session.run(cypher, params)
        return [dict(r) for r in result]


def upsert_relationship(
    source_name: str,
    target_name: str,
    rel_type: str,
    properties: dict | None = None,
) -> dict | None:
    """Create or update a directed relationship between two entities."""
    driver = get_driver()
    properties_json = _props_to_storage(properties)
    with driver.session() as session:
        for name in (source_name, target_name):
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET
                    e.id = randomUUID(),
                    e.type = 'unknown',
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                """,
                name=name,
            )

        result = session.run(
            """
            MATCH (a:Entity {name: $source})
            MATCH (b:Entity {name: $target})
            MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
            ON CREATE SET r.created_at = datetime(), r.properties = $properties_json
            ON MATCH SET r.properties = $properties_json
            RETURN r.type AS type, r.properties AS properties
            """,
            source=source_name,
            target=target_name,
            rel_type=rel_type,
            properties_json=properties_json,
        )
        record = result.single()
        if record is None:
            return None
        out = dict(record)
        out["properties"] = _props_from_storage(out.get("properties"))
        return out


def get_relationships(
    entity_name: str,
    direction: str = "both",
    rel_type: str | None = None,
    depth: int = 1,
) -> list[dict]:
    """Get relationships for an entity. Direction: outgoing, incoming, or both."""
    driver = get_driver()
    with driver.session() as session:
        if direction == "outgoing":
            match = "(a:Entity {name: $name})-[r:RELATES_TO]->(b:Entity)"
        elif direction == "incoming":
            match = "(b:Entity)<-[r:RELATES_TO]-(a:Entity {name: $name})"
        else:
            match = "(a:Entity {name: $name})-[r:RELATES_TO]-(b:Entity)"

        cypher = f"MATCH {match}"
        params = {"name": entity_name}
        if rel_type:
            cypher += " WHERE r.type = $rel_type"
        cypher += """
            RETURN r.type AS rel_type,
                   a.name AS source_name,
                   b.name AS target_name,
                   r.properties AS properties
            LIMIT 50
        """
        params["rel_type"] = rel_type
        result = session.run(cypher, params)
        rels = []
        for r in result:
            d = dict(r)
            d["properties"] = _props_from_storage(d.get("properties"))
            rels.append(d)
        return rels


def find_connected_entities(
    entity_name: str,
    max_depth: int = 2,
) -> list[dict]:
    """BFS traversal to find entities connected within max_depth hops."""
    # Neo4j forbids a parameter as the upper bound of a variable-length pattern,
    # so the (validated, clamped) int is inlined as a literal.
    depth = max(1, min(int(max_depth), 10))
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH path = (a:Entity {{name: $name}})
                        -[r:RELATES_TO*1..{depth}]-(b:Entity)
            WHERE a <> b
            RETURN DISTINCT
                b.name AS name,
                b.type AS type,
                b.description AS description,
                length(path) AS depth
            ORDER BY depth, b.name
            LIMIT 100
            """,
            name=entity_name,
        )
        return [dict(r) for r in result]


def associative_retrieval(
    query_entities: list[str],
    max_depth: int = 3,
) -> list[dict]:
    """Find subgraphs connecting multiple query entities."""
    depth = max(1, min(int(max_depth), 10))  # literal upper bound (see above)
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH path = (a:Entity)-[r:RELATES_TO*1..{depth}]-(b:Entity)
            WHERE a.name IN $entities
              AND b.name IN $entities
              AND a <> b
            RETURN [n IN nodes(path) | n.name] AS path_nodes,
                   [rel IN relationships(path) | rel.type] AS path_rels,
                   length(path) AS depth
            ORDER BY depth
            LIMIT 50
            """,
            entities=query_entities,
        )
        return [dict(r) for r in result]
