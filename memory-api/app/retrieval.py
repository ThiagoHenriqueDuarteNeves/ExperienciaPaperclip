import uuid

from app.chroma_client import get_or_create_collection
from app.config import settings
from app.embeddings import embed_text
try:
    from app.kg_retrieval import store_knowledge
    _KG_AVAILABLE = True
except ImportError:
    _KG_AVAILABLE = False

    def store_knowledge(*args, **kwargs) -> None:
        pass


def store_conversation(
    user_id: str,
    conversation_id: str,
    content: str,
    metadata: dict | None = None,
    extract_knowledge_graph: bool = True,
) -> str:
    memory_id = str(uuid.uuid4())
    embedding = embed_text(content)
    doc_metadata = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        **(metadata or {}),
    }
    collection = get_or_create_collection()
    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[doc_metadata],
    )

    # Extract entities and relationships into Neo4j in background
    if extract_knowledge_graph:
        try:
            store_knowledge(text=content, source_id=memory_id)
        except Exception:
            pass  # KG extraction is best-effort

    return memory_id


def retrieve_similar(
    query: str,
    user_id: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
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

    memories = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance  # cosine distance → similarity
        if similarity < settings.min_similarity_threshold:
            continue
        memories.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": round(similarity, 4),
        })

    return memories


def delete_memory(memory_id: str) -> None:
    collection = get_or_create_collection()
    collection.delete(ids=[memory_id])
