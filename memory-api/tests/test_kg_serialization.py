"""Unit tests for Neo4j relationship-property (de)serialization.

The bug: Neo4j rejects a nested map as a relationship property value, so every
upsert_relationship with properties failed and the graph ended up with 0 edges.
The fix stores properties as a JSON string. These pure tests lock that contract.

Run with:  pytest -m unit
"""

from __future__ import annotations

import pytest

from app.neo4j_client import _props_from_storage, _props_to_storage

pytestmark = pytest.mark.unit


def test_nested_map_roundtrips():
    # This is the exact shape that used to crash Neo4j (22N01 type error).
    props = {"duration": "12 anos"}
    assert _props_from_storage(_props_to_storage(props)) == props


def test_mixed_value_types_roundtrip():
    props = {"years_experience": 5, "primary": True, "role": "engineer"}
    assert _props_from_storage(_props_to_storage(props)) == props


def test_to_storage_always_returns_str():
    assert isinstance(_props_to_storage({"a": 1}), str)
    assert _props_to_storage(None) == "{}"
    assert _props_to_storage({}) == "{}"


def test_from_storage_empty_json_is_empty_dict():
    assert _props_from_storage("{}") == {}


def test_from_storage_tolerates_legacy_dict():
    # If an older value was somehow stored as a real map, don't choke on it.
    assert _props_from_storage({"already": "dict"}) == {"already": "dict"}


def test_from_storage_tolerates_garbage():
    assert _props_from_storage("not json") == {}
    assert _props_from_storage(None) == {}
    assert _props_from_storage(42) == {}
