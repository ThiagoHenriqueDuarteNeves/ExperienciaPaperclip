import os

import chromadb
from chromadb import Collection

from app.config import settings

_client: chromadb.HttpClient | chromadb.PersistentClient | None = None


def get_client() -> chromadb.HttpClient | chromadb.PersistentClient:
    global _client
    if _client is None:
        if settings.chromadb_use_local:
            persist_dir = settings.chromadb_local_persist_dir
            os.makedirs(persist_dir, exist_ok=True)
            _client = chromadb.PersistentClient(path=persist_dir)
        else:
            _client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port,
            )
    return _client


def get_or_create_collection() -> Collection:
    return get_client().get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def health() -> bool:
    try:
        get_client().heartbeat()
        return True
    except Exception:
        return False
