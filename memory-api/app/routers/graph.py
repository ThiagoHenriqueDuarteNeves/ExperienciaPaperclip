"""Knowledge-graph routes (Neo4j).

Owns the Neo4j optional-import block: if neo4j_client / kg_retrieval (or their
deps) are unavailable, the functions degrade to safe stubs so the rest of the API
keeps working. main.py reuses `neo4j_health` (for /health) and `init_schema`
(for startup) from here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.rate_limit import limiter
from app.schemas import EnrichedSearchRequest, GraphQueryRequest, GraphSearchRequest

try:
    from app.neo4j_client import health as neo4j_health, init_schema
    from app.kg_retrieval import (
        augment_with_graph_context,
        get_entity_graph,
        query_graph,
        search_graph,
    )

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

    def neo4j_health() -> bool:
        return False

    def init_schema() -> None:
        pass

    def search_graph(*args, **kwargs) -> list:
        return []

    def query_graph(*args, **kwargs) -> dict:
        return {"entity": None}

    def get_entity_graph(*args, **kwargs) -> dict | None:
        return None

    def augment_with_graph_context(*args, **kwargs) -> list:
        return []


router = APIRouter(tags=["graph"])


@router.post("/graph/search")
@limiter.limit("30/minute")
def graph_search(request: Request, req: GraphSearchRequest):
    results = search_graph(query=req.query, type_filter=req.type_filter, limit=req.limit)
    return {"entities": results}


@router.post("/graph/query")
@limiter.limit("30/minute")
def graph_query(request: Request, req: GraphQueryRequest):
    result = query_graph(entity_name=req.entity_name, depth=req.depth)
    if result["entity"] is None:
        raise HTTPException(status_code=404, detail=f"Entity '{req.entity_name}' not found")
    return result


@router.get("/graph/entity/{name}")
def graph_entity(name: str, depth: int = 2):
    result = get_entity_graph(name, depth=depth)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return result


@router.post("/search/enriched")
@limiter.limit("30/minute")
def enriched_search(request: Request, req: EnrichedSearchRequest):
    results = augment_with_graph_context(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    return {"memories": results}
