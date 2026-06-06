"""Unit tests for semantic fact parsing (pure: no LLM/DB).

Run with:  pytest -m unit
"""

from __future__ import annotations

import pytest

from app import semantic_extractor as se

pytestmark = pytest.mark.unit


def test_parse_valid_facts():
    out = se._parse_facts('[{"key":"profissao","content":"Trabalha como QA.","importance":0.8}]')
    assert out == [{"key": "profissao", "content": "Trabalha como QA.", "importance": 0.8}]


def test_parse_empty_list():
    assert se._parse_facts("[]") == []
    assert se._parse_facts("") == []


def test_parse_strips_markdown_fence():
    out = se._parse_facts('```json\n[{"key":"nome","content":"Thiago"}]\n```')
    assert out[0]["key"] == "nome"
    assert out[0]["importance"] == 0.5  # default when missing


def test_key_is_normalized_to_snake_case():
    out = se._parse_facts('[{"key":"Objetivo De Carreira!","content":"IA"}]')
    assert out[0]["key"] == "objetivo_de_carreira"


def test_importance_is_clamped():
    out = se._parse_facts('[{"key":"a","content":"x","importance":5},{"key":"b","content":"y","importance":-3}]')
    assert out[0]["importance"] == 1.0
    assert out[1]["importance"] == 0.0


def test_malformed_items_skipped():
    out = se._parse_facts('[{"key":"ok","content":"v"},{"no_key":"x"},{"key":"k2"},"junk"]')
    assert [f["key"] for f in out] == ["ok"]


def test_duplicate_keys_collapsed_within_extraction():
    out = se._parse_facts('[{"key":"nome","content":"A"},{"key":"nome","content":"B"}]')
    assert len(out) == 1
    assert out[0]["content"] == "A"


def test_non_list_returns_empty():
    assert se._parse_facts('{"key":"x","content":"y"}') == []


def test_caps_at_ten_facts():
    items = ",".join(f'{{"key":"k{i}","content":"c"}}' for i in range(20))
    assert len(se._parse_facts(f"[{items}]")) == 10


# ---------------------------------------------------------------------------
# Key alias collapsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,canonical", [
    ("profissao_atual", "profissao"),
    ("trabalho_atual",  "profissao"),
    ("emprego_atual",   "profissao"),
    ("cargo_atual",     "profissao"),
    ("nome_completo",   "nome"),
    ("primeiro_nome",   "nome"),
    ("nome_usuario",    "nome"),
])
def test_key_aliases_collapsed_to_canonical(variant, canonical):
    out = se._parse_facts(f'[{{"key":"{variant}","content":"x"}}]')
    assert out[0]["key"] == canonical


def test_profissao_atual_deduped_against_profissao():
    """Two keys that both normalise to 'profissao' must produce only one fact."""
    out = se._parse_facts(
        '[{"key":"profissao","content":"QA"},{"key":"profissao_atual","content":"QA Senior"}]'
    )
    assert len(out) == 1
    assert out[0]["key"] == "profissao"
    # first occurrence wins (canonical "profissao" came first)
    assert out[0]["content"] == "QA"
