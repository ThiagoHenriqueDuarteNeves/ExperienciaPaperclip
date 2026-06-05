"""Unit tests for cross-encoder re-ranking (Fase C).

The model is mocked, so these test the ordering/selection/fallback logic only.
Run with:  pytest -m unit
"""

from __future__ import annotations

import pytest

from app import reranker

pytestmark = pytest.mark.unit


class _FakeModel:
    """Returns a preset score per memory content."""

    def __init__(self, scores_by_content):
        self.scores_by_content = scores_by_content

    def predict(self, pairs):
        return [self.scores_by_content[content] for _q, content in pairs]


def test_rerank_orders_by_score_desc(monkeypatch):
    mems = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    monkeypatch.setattr(
        reranker, "_get_model", lambda: _FakeModel({"a": 0.1, "b": 0.9, "c": 0.5})
    )
    out = reranker.rerank_memories("q", mems, top_n=3)
    assert [m["content"] for m in out] == ["b", "c", "a"]


def test_rerank_keeps_only_top_n(monkeypatch):
    mems = [{"content": x} for x in "abcd"]
    monkeypatch.setattr(
        reranker, "_get_model", lambda: _FakeModel({"a": 1, "b": 2, "c": 3, "d": 4})
    )
    out = reranker.rerank_memories("q", mems, top_n=2)
    assert [m["content"] for m in out] == ["d", "c"]


def test_rerank_empty_or_no_query_is_safe():
    assert reranker.rerank_memories("q", [], top_n=3) == []
    mems = [{"content": "a"}, {"content": "b"}]
    assert reranker.rerank_memories("", mems, top_n=1) == [{"content": "a"}]


def test_rerank_falls_back_on_model_error(monkeypatch):
    def boom():
        raise RuntimeError("model not available")

    monkeypatch.setattr(reranker, "_get_model", boom)
    mems = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    # Falls back to input order, capped to top_n — never raises.
    assert reranker.rerank_memories("q", mems, top_n=2) == [{"content": "a"}, {"content": "b"}]


# -- integration with assemble_memory_context (rerank flag) -----------------

def test_assemble_uses_rerank_when_enabled(monkeypatch):
    from app import chat_pipeline as cp

    monkeypatch.setattr(
        reranker, "_get_model",
        lambda: _FakeModel({"low relevance": 0.1, "HIGH relevance": 0.9}),
    )
    layers = {
        "episodic": [
            {"content": "low relevance", "similarity": 0.95},   # high cosine...
            {"content": "HIGH relevance", "similarity": 0.72},  # ...but cross-encoder prefers this
        ],
        "conversation": [],
    }
    ctx = cp.assemble_memory_context(layers, query="q", rerank=True, rerank_top_n=1)
    # Re-ranker wins over raw cosine order.
    assert "HIGH relevance" in ctx
    assert "low relevance" not in ctx
