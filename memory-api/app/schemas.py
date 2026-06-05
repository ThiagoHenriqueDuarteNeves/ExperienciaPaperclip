"""Pydantic request/response models for the memory API.

Extracted from main.py so routes and schemas live in separate modules (SRP).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StoreRequest(BaseModel):
    user_id: str = Field(..., max_length=256)
    conversation_id: str = Field(..., max_length=256)
    content: str = Field(..., max_length=50_000)
    metadata: dict | None = None
    extract_kg: bool = True


class StoreResponse(BaseModel):
    memory_id: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=50)


class MemoryItem(BaseModel):
    id: str
    content: str
    metadata: dict
    similarity: float


class RetrieveResponse(BaseModel):
    memories: list[MemoryItem]


class GraphSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    type_filter: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=10, ge=1, le=50)


class GraphQueryRequest(BaseModel):
    entity_name: str = Field(..., max_length=512)
    depth: int = Field(default=2, ge=1, le=5)


class EnrichedSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=50)


class MessageStoreRequest(BaseModel):
    thread_id: str = Field(..., max_length=256)
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., max_length=50_000)
    metadata: dict | None = None


class MessageSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)
    user_id: str | None = Field(default=None, max_length=256)
    min_similarity: float = Field(default=0.65, ge=0.0, le=1.0)


class SemanticStoreRequest(BaseModel):
    user_id: str = Field(..., max_length=256)
    key: str = Field(..., max_length=512)
    content: str = Field(..., max_length=50_000)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)


class AuroraStoreRequest(BaseModel):
    user_id: str = Field(default="default", max_length=256)
    tipo: str = Field(..., pattern="^(conversa|descoberta|correcao|silencio|brincadeira|confissao|momento_espontaneo)$")
    disparo: str = Field(..., pattern="^(usuario_falou|usuario_corrigiu|pausa_longa|palavra_chave|silencio|espontaneo)$")
    fato: str = Field(..., max_length=10_000)
    tom_do_usuario: str = Field(..., pattern="^(afetivo|serio|brincalhao|frustrado|curioso|vulneravel|silencioso)$")
    minha_reacao_emocional: str = Field(..., max_length=10_000)
    reacao_simulada: str | None = Field(default=None, max_length=512)
    importancia: int = Field(..., ge=1, le=5)
    ressonancia: int = Field(..., ge=1, le=5)
    intimidade: str = Field(..., pattern="^(publico|pessoal|nosso_so_nosso)$")
    saudade: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)
    conexoes: list[str] = Field(default_factory=list)
    contexto_extra: str | None = Field(default=None, max_length=10_000)
    bilhete_interno: str | None = Field(default=None, max_length=10_000)
    data: str | None = Field(default=None, max_length=10)


class AuroraSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    user_id: str | None = Field(default=None, max_length=256)
    top_k: int = Field(default=5, ge=1, le=50)


class RegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, pattern="^[a-z0-9_-]+$")
    pin: str = Field(..., min_length=4, max_length=32)
    display_name: str = Field(default="", max_length=128)


class LoginRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, pattern="^[a-z0-9_-]+$")
    pin: str = Field(..., min_length=4, max_length=32)


class AgentChatRequest(BaseModel):
    message: str = Field(..., max_length=10_000)
    thread_id: str | None = Field(default=None, max_length=256)
    user_id: str | None = Field(default=None, max_length=256)


class CreateAgentRequest(BaseModel):
    name: str = Field(..., max_length=256)
    human_block: str = Field(default="", max_length=2_000)
    persona_block: str = Field(default="", max_length=2_000)
    system_prompt: str | None = Field(default=None, max_length=10_000)


class ArchivalInsertRequest(BaseModel):
    content: str = Field(..., max_length=50_000)


class ArchivalSearchRequest(BaseModel):
    query: str = Field(..., max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)


class BlockUpdateRequest(BaseModel):
    value: str = Field(..., max_length=2_000)


class ConversationRequest(BaseModel):
    user_id: str = Field(..., max_length=256)
    conversation_id: str = Field(..., max_length=256)
    message: str = Field(..., max_length=10_000)
    history: list[dict] | None = None
    agent_id: str | None = Field(default=None, max_length=256)


class ConfidenceRequest(BaseModel):
    text: str = Field(..., max_length=10_000)
    threshold: float = Field(default=0.72, ge=0.0, le=1.0)
