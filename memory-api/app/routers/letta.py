"""Letta (MemGPT) procedural-memory routes.

Owns its optional-import handling: if letta_manager (or its deps) is unavailable,
the functions degrade to safe stubs so the rest of the API keeps working. main.py
reuses `letta_health` here for the aggregate /health endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import (
    ArchivalInsertRequest,
    ArchivalSearchRequest,
    BlockUpdateRequest,
    CreateAgentRequest,
)

try:
    from app.letta_manager import (
        create_agent as letta_create_agent,
        delete_agent as letta_delete_agent,
        get_core_memory as letta_get_core_memory,
        health as letta_health,
        insert_archival_memory as letta_insert_archival,
        list_agents as letta_list_agents,
        lookup_agent as letta_lookup_agent,
        search_archival_memory as letta_search_archival,
        update_human_block as letta_update_human,
        update_persona_block as letta_update_persona,
    )

    _LETTA_AVAILABLE = True
except ImportError:
    _LETTA_AVAILABLE = False

    def letta_health() -> bool:
        return False

    def letta_list_agents() -> list:
        return []

    def letta_lookup_agent(*args, **kwargs):
        return None

    def letta_create_agent(*args, **kwargs):
        return None

    def letta_delete_agent(*args, **kwargs):
        return None

    def letta_get_core_memory(*args, **kwargs):
        return None

    def letta_update_human(*args, **kwargs):
        return None

    def letta_update_persona(*args, **kwargs):
        return None

    def letta_insert_archival(*args, **kwargs):
        return None

    def letta_search_archival(*args, **kwargs):
        return []


router = APIRouter(tags=["letta"])


@router.get("/letta/health")
def letta_health_endpoint():
    ok = letta_health()
    return {"letta_available": ok}


@router.post("/letta/agents")
def letta_create_agent_endpoint(req: CreateAgentRequest):
    agent = letta_create_agent(
        name=req.name,
        human_block=req.human_block,
        persona_block=req.persona_block,
        system_prompt=req.system_prompt,
    )
    if agent is None:
        raise HTTPException(status_code=503, detail="Letta agent creation failed")
    return agent


@router.get("/letta/agents")
def letta_list_agents_endpoint():
    return {"agents": letta_list_agents()}


@router.get("/letta/agents/{agent_id}")
def letta_get_agent_endpoint(agent_id: str):
    agent = letta_lookup_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent


@router.delete("/letta/agents/{agent_id}", status_code=204)
def letta_delete_agent_endpoint(agent_id: str):
    letta_delete_agent(agent_id)


@router.get("/letta/agents/{agent_id}/memory")
def letta_get_memory_endpoint(agent_id: str):
    memory = letta_get_core_memory(agent_id)
    if memory is None:
        raise HTTPException(
            status_code=404, detail=f"Memory not found for agent '{agent_id}'"
        )
    return memory


@router.put("/letta/agents/{agent_id}/memory/human")
def letta_update_human_endpoint(agent_id: str, req: BlockUpdateRequest):
    result = letta_update_human(agent_id, req.value)
    if result is None:
        raise HTTPException(status_code=503, detail="Failed to update human memory block")
    return result


@router.put("/letta/agents/{agent_id}/memory/persona")
def letta_update_persona_endpoint(agent_id: str, req: BlockUpdateRequest):
    result = letta_update_persona(agent_id, req.value)
    if result is None:
        raise HTTPException(status_code=503, detail="Failed to update persona memory block")
    return result


@router.post("/letta/agents/{agent_id}/archival")
def letta_insert_archival_endpoint(agent_id: str, req: ArchivalInsertRequest):
    result = letta_insert_archival(agent_id, req.content)
    if result is None:
        raise HTTPException(status_code=503, detail="Failed to insert archival memory")
    return result


@router.post("/letta/agents/{agent_id}/archival/search")
def letta_search_archival_endpoint(agent_id: str, req: ArchivalSearchRequest):
    results = letta_search_archival(agent_id=agent_id, query=req.query, limit=req.limit)
    return {"results": results}
