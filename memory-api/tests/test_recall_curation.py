"""Unit tests for recall curation: gating, dedup, budget (pure, no DB/LLM).

These lock down Phase A — turning the raw 20-memory dump into a compact, relevant
context. Run with:  pytest -m unit
"""

from __future__ import annotations

import pytest

from app import chat_pipeline as cp

pytestmark = pytest.mark.unit


# -- _gate ------------------------------------------------------------------

def test_gate_drops_below_threshold():
    mems = [{"content": "a", "similarity": 0.9}, {"content": "b", "similarity": 0.5}]
    kept = cp._gate(mems, 0.7)
    assert [m["content"] for m in kept] == ["a"]


def test_gate_missing_similarity_treated_as_zero():
    assert cp._gate([{"content": "x"}], 0.1) == []


def test_gate_zero_threshold_keeps_all():
    mems = [{"content": "a", "similarity": 0.1}, {"content": "b", "similarity": 0.9}]
    assert len(cp._gate(mems, 0.0)) == 2


# -- _near_duplicate / merge_and_dedup --------------------------------------

def test_near_duplicate_containment():
    assert cp._near_duplicate("o gato preto", "o gato preto dormiu no sofa")


def test_near_duplicate_distinct_false():
    assert not cp._near_duplicate("eu gosto de pizza", "meu carro precisa de oleo")


def test_merge_dedup_sorts_by_similarity_desc():
    ep = [{"content": "low", "similarity": 0.4}]
    co = [{"content": "high", "similarity": 0.95}]
    out = cp.merge_and_dedup(ep, co)
    assert [m["content"] for m in out] == ["high", "low"]


def test_merge_dedup_collapses_overlap_keeping_strongest():
    ep = [{"content": "Thiago trabalha como QA", "similarity": 0.9}]
    co = [{"content": "Thiago trabalha como QA na empresa", "similarity": 0.6}]
    out = cp.merge_and_dedup(ep, co)
    # The two cover the same fact; only the strongest (containment) is kept.
    assert len(out) == 1
    assert out[0]["similarity"] == 0.9


def test_merge_dedup_skips_empty_content():
    out = cp.merge_and_dedup([{"content": "  ", "similarity": 0.9}], [])
    assert out == []


# -- apply_budget -----------------------------------------------------------

def test_apply_budget_stops_at_limit():
    mems = [{"content": "x" * 100}, {"content": "y" * 100}, {"content": "z" * 100}]
    out = cp.apply_budget(mems, 150)
    assert len(out) == 1  # first fits (100), second would exceed 150


def test_apply_budget_always_keeps_first_even_if_over():
    mems = [{"content": "x" * 9999}]
    assert len(cp.apply_budget(mems, 100)) == 1


def test_apply_budget_none_keeps_all():
    mems = [{"content": "a"}, {"content": "b"}]
    assert cp.apply_budget(mems, None) == mems


# -- assemble_memory_context with curation ----------------------------------

def test_assemble_applies_gate_and_budget():
    layers = {
        "episodic": [
            {"content": "RELEVANTE muito bom", "similarity": 0.92},
            {"content": "irrelevante fraco", "similarity": 0.30},
        ],
        "conversation": [],
    }
    ctx = cp.assemble_memory_context(layers, min_similarity=0.7, char_budget=10_000)
    assert "RELEVANTE" in ctx
    assert "irrelevante" not in ctx  # gated out


def test_assemble_defaults_keep_old_behavior():
    # No thresholds -> nothing gated/budgeted (back-compat for callers).
    layers = {"episodic": [{"content": "kept", "similarity": 0.1}], "conversation": []}
    assert "kept" in cp.assemble_memory_context(layers)
