"""Unit tests for the semantic backfill turn-splitting (pure).

Run with:  pytest -m unit
"""

from __future__ import annotations

import pytest

from app.scripts.backfill_semantic import _split_turn

pytestmark = pytest.mark.unit


def test_split_standard_turn():
    user, assistant = _split_turn("User: oi tudo bem?\nAssistant: tudo, e você?")
    assert user == "oi tudo bem?"
    assert assistant == "tudo, e você?"


def test_split_multiline_reply():
    doc = "User: me fala sobre IA\nAssistant: claro!\nÉ uma área ampla."
    user, assistant = _split_turn(doc)
    assert user == "me fala sobre IA"
    assert assistant == "claro!\nÉ uma área ampla."


def test_split_without_assistant_marker():
    user, assistant = _split_turn("apenas um texto solto")
    assert user == "apenas um texto solto"
    assert assistant == ""


def test_split_does_not_double_label():
    # The whole point: the user part must not still contain the 'User:' label.
    user, _ = _split_turn("User: sou QA\nAssistant: legal")
    assert "User:" not in user
