"""Embedding provider: local SentenceTransformer (primary) with optional API override.

Default: intfloat/multilingual-e5-large — 1024-dim, multilingual, runs inside the container.
API override: set MEMORY_EMBEDDING_API_BASE + MEMORY_EMBEDDING_API_KEY in .env to use
  OpenAI (text-embedding-3-large, 1536-dim) or any OpenAI-compat provider instead.

WARNING: mixing local↔API corrupts ANN search if dimensions differ.
To switch providers: docker compose down -v && docker compose up --build
"""

from __future__ import annotations

import hashlib
import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_local_model = None
_cache: dict[str, tuple[list[float], float]] = {}
_MAX_CACHE = 10000
_CACHE_TTL = 3600


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cache_get(key: str) -> list[float] | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    vec, ts = entry
    if time.time() - ts > _CACHE_TTL:
        del _cache[key]
        return None
    return vec


def _cache_set(key: str, vec: list[float]) -> None:
    if len(_cache) >= _MAX_CACHE:
        oldest = min(_cache, key=lambda k: _cache[k][1])
        del _cache[oldest]
    _cache[key] = (vec, time.time())


def _can_use_api() -> bool:
    return bool(settings.effective_embedding_api_key) and bool(settings.effective_embedding_api_base)


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("embeddings: loading local model '%s'", settings.embedding_model)
        _local_model = SentenceTransformer(settings.embedding_model)
        logger.info("embeddings: local model loaded, dim=%d", _local_model.get_sentence_embedding_dimension())
    return _local_model


# -- Async --

async def embed_text_async(text: str, is_query: bool = False) -> list[float]:
    if _can_use_api():
        key = _cache_key(text)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        vec = await _embed_api(text)
        _cache_set(key, vec)
        return vec
    return _embed_local(text, is_query=is_query)


async def embed_texts_async(texts: list[str], is_query: bool = False) -> list[list[float]]:
    if _can_use_api():
        return await _embed_api_batch(texts)
    return _embed_local_batch(texts, is_query=is_query)


# -- Sync --

def embed_text(text: str, is_query: bool = False) -> list[float]:
    if _can_use_api():
        key = _cache_key(text)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        vec = _embed_api_sync(text)
        _cache_set(key, vec)
        return vec
    return _embed_local(text, is_query=is_query)


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    if _can_use_api():
        return _embed_api_batch_sync(texts)
    return _embed_local_batch(texts, is_query=is_query)


# -- Local model --

def _embed_local(text: str, is_query: bool = False) -> list[float]:
    # multilingual-e5-large é ASSIMÉTRICO: "query: " para buscas, "passage: " para documentos.
    # Usar o prefixo errado degrada o ranking (perguntas batem melhor que respostas).
    prefix = "query: " if is_query else "passage: "
    return _get_local_model().encode(f"{prefix}{text}").tolist()


def _embed_local_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    prefix = "query: " if is_query else "passage: "
    prefixed = [f"{prefix}{t}" for t in texts]
    return [v.tolist() for v in _get_local_model().encode(prefixed)]


# -- API helpers --

async def _embed_api(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.effective_embedding_api_base}/embeddings",
            headers={"Authorization": f"Bearer {settings.effective_embedding_api_key}", "Content-Type": "application/json"},
            json={"model": settings.embedding_api_model, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def _embed_api_batch(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.effective_embedding_api_base}/embeddings",
            headers={"Authorization": f"Bearer {settings.effective_embedding_api_key}", "Content-Type": "application/json"},
            json={"model": settings.embedding_api_model, "input": texts},
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]


def _embed_api_sync(text: str) -> list[float]:
    resp = httpx.post(
        f"{settings.effective_embedding_api_base}/embeddings",
        headers={"Authorization": f"Bearer {settings.effective_embedding_api_key}", "Content-Type": "application/json"},
        json={"model": settings.embedding_api_model, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_api_batch_sync(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        f"{settings.effective_embedding_api_base}/embeddings",
        headers={"Authorization": f"Bearer {settings.effective_embedding_api_key}", "Content-Type": "application/json"},
        json={"model": settings.embedding_api_model, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]
