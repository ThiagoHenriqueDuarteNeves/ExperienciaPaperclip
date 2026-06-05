"""Unit tests for chat_pipeline assembly (pure: no DB, no LLM).

These lock down exactly how recalled memory becomes the system prompt — the
contract the inspector relies on to show "what is sent to the LLM".

Run with:  pytest -m unit
"""

from __future__ import annotations

import pytest

from app import chat_pipeline as cp

pytestmark = pytest.mark.unit


def test_build_system_without_context_is_prompt_only():
    system = cp.build_system("")
    assert len(system) == 1
    assert system[0]["text"] == cp.CHAT_SYSTEM_PROMPT


def test_build_system_with_context_appends_authoritative_block():
    system = cp.build_system("MEMORY HERE")
    assert len(system) == 2
    assert system[0]["text"] == cp.CHAT_SYSTEM_PROMPT
    assert "Retrieved memories (AUTHORITATIVE" in system[1]["text"]
    assert system[1]["text"].endswith("MEMORY HERE")


def test_assemble_empty_layers_is_empty():
    assert cp.assemble_memory_context({}) == ""
    assert cp.assemble_memory_context(
        {"episodic": [], "conversation": [], "identity": [], "aurora": []}
    ) == ""


def test_assemble_identity_first():
    ctx = cp.assemble_memory_context(
        {"identity": [{"content": "User's name is Thiago."}]}
    )
    assert ctx == "User's name is Thiago."


def test_assemble_merges_episodic_and_conversation_with_similarity():
    ctx = cp.assemble_memory_context(
        {
            "episodic": [{"content": "ep one", "similarity": 0.91}],
            "conversation": [{"content": "conv one", "similarity": 0.7}],
        }
    )
    assert "Memory 1 (similarity 0.91): ep one" in ctx
    assert "Memory 2 (similarity 0.70): conv one" in ctx


def test_assemble_aurora_block_includes_tone_and_bilhete():
    ctx = cp.assemble_memory_context(
        {
            "aurora": [
                {
                    "tipo": "confissao",
                    "tom_do_usuario": "vulneravel",
                    "importancia": 5,
                    "fato": "algo importante",
                    "minha_reacao_emocional": "aperto no peito",
                    "bilhete_interno": "lembrar disso",
                }
            ]
        }
    )
    assert "Arquivo Aurora" in ctx
    assert "[confissao, tom vulneravel, intensidade 5/5] algo importante" in ctx
    assert "Como senti: aperto no peito" in ctx
    assert "Bilhete interno: lembrar disso" in ctx


def test_assemble_full_order_identity_then_mems_then_aurora():
    ctx = cp.assemble_memory_context(
        {
            "identity": [{"content": "ID"}],
            "episodic": [{"content": "EP", "similarity": 0.5}],
            "aurora": [
                {
                    "tipo": "brincadeira",
                    "tom_do_usuario": "brincalhao",
                    "importancia": 2,
                    "fato": "AU",
                    "minha_reacao_emocional": "sorriso",
                }
            ],
        }
    )
    assert ctx.index("ID") < ctx.index("EP") < ctx.index("Arquivo Aurora")
