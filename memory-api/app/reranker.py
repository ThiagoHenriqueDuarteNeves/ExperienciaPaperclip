"""Optional cross-encoder re-ranking of recalled memories (Fase C).

A bi-encoder (the e5 embedding) ranks by cosine of independently-embedded texts;
a cross-encoder scores the (query, memory) pair jointly and is markedly more
precise about which few memories actually answer the question. We over-fetch
candidates, re-rank them here, and keep the top_n.

Off by default (settings.recall_rerank_enabled) so a heavy extra model isn't
forced onto a constrained host. If disabled, or if the model can't load, this
falls back to the incoming order (similarity) — it never breaks recall.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_model = None
_DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual (PT-friendly)


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        from app.config import settings

        name = getattr(settings, "recall_rerank_model", _DEFAULT_MODEL)
        logger.info("reranker: loading cross-encoder '%s'", name)
        _model = CrossEncoder(name)
    return _model


def rerank_memories(query: str, mems: list[dict], top_n: int) -> list[dict]:
    """Return the top_n memories re-ranked by cross-encoder relevance to query.

    Safe by construction: any failure falls back to the input order, capped to
    top_n, so callers can use it unconditionally.
    """
    if not mems or not query:
        return mems[:top_n]
    try:
        model = _get_model()
        scores = model.predict([(query, m.get("content", "") or "") for m in mems])
        order = sorted(range(len(mems)), key=lambda i: scores[i], reverse=True)
        return [mems[i] for i in order][:top_n]
    except Exception as exc:  # model missing / load error / predict error
        logger.warning("reranker: falling back (no rerank): %s", exc)
        return mems[:top_n]
