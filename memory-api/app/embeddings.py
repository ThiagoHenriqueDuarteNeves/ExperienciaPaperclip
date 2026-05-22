"""Embedding providers: API-based (Deepseek-compatible, primary) with local fallback."""

from __future__ import annotations

import hashlib
import time

import httpx
from sentence_transformers import SentenceTransformer

from app.config import settings

_local_model: SentenceTransformer | None = None

# In-process embedding cache for API embeddings
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


def _get_local_model() -> SentenceTransformer:
    global _local_model
    if _local_model is None:
        _local_model = SentenceTransformer(settings.embedding_model)
    return _local_model


def _embedding_api_key() -> str:
    """Return the effective embedding API key or empty string.

    Uses the provider-aware property from settings so that:
    - Anthropic provider → returns "" (use local model)
    - Deepseek provider → returns Deepseek API key
    - Explicit MEMORY_EMBEDDING_API_KEY → always takes precedence
    """
    return settings.effective_embedding_api_key


def _embedding_api_base() -> str:
    """Return the effective embedding API base URL or empty string.

    Uses the provider-aware property from settings so that:
    - Anthropic provider → returns "" (use local model)
    - Deepseek provider → returns Deepseek's embedding endpoint
    - Explicit MEMORY_EMBEDDING_API_BASE → always takes precedence
    """
    return settings.effective_embedding_api_base


def _can_use_api() -> bool:
    """Check if API embeddings are available (both key and base URL required)."""
    return bool(_embedding_api_key()) and bool(_embedding_api_base())


# -- Async API embeddings (primary) --


async def embed_text_async(text: str) -> list[float]:
    if _can_use_api():
        key = _cache_key(text)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        vec = await _embed_api(text, _embedding_api_key())
        _cache_set(key, vec)
        return vec
    return _embed_local(text)


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    if _can_use_api():
        return await _embed_api_batch(texts, _embedding_api_key())
    return _embed_local_batch(texts)


# -- Sync embeddings (for existing sync code paths) --


def embed_text(text: str) -> list[float]:
    if _can_use_api():
        key = _cache_key(text)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        vec = _embed_api_sync(text, _embedding_api_key())
        _cache_set(key, vec)
        return vec
    return _embed_local(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _can_use_api():
        return _embed_api_batch_sync(texts, _embedding_api_key())
    return _embed_local_batch(texts)


# -- Local model (384-dim) --


def _embed_local(text: str) -> list[float]:
    return _get_local_model().encode(text).tolist()


def _embed_local_batch(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_local_model().encode(texts)]


# -- API helpers --


async def _embed_api(text: str, api_key: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_embedding_api_base()}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": settings.embedding_api_model, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def _embed_api_batch(texts: list[str], api_key: str) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{_embedding_api_base()}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": settings.embedding_api_model, "input": texts},
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]


def _embed_api_sync(text: str, api_key: str) -> list[float]:
    resp = httpx.post(
        f"{_embedding_api_base()}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": settings.embedding_api_model, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_api_batch_sync(texts: list[str], api_key: str) -> list[list[float]]:
    resp = httpx.post(
        f"{_embedding_api_base()}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": settings.embedding_api_model, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]
