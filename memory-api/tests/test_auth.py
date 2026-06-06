"""Characterization tests for app.auth — PIN hashing + HMAC session tokens.

Pure unit tests: no database, no network, no heavy ML deps. These lock down the
current behavior of the auth layer so later refactors (e.g. splitting main.py
into routers) cannot silently change it.

Run with:  pytest -m unit
"""

from __future__ import annotations

import time

import pytest

from app import auth

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# PIN hashing
# ---------------------------------------------------------------------------


def test_hash_pin_returns_hex_hash_and_salt():
    pin_hash, pin_salt = auth.hash_pin("1234")
    # both are non-empty hex strings
    assert pin_hash and pin_salt
    bytes.fromhex(pin_hash)  # raises if not valid hex
    bytes.fromhex(pin_salt)
    # salt is 16 bytes -> 32 hex chars
    assert len(pin_salt) == 32


def test_hash_pin_is_salted_unique_per_call():
    h1, s1 = auth.hash_pin("1234")
    h2, s2 = auth.hash_pin("1234")
    # Same PIN, different salt => different hash (no global pepper reuse).
    assert s1 != s2
    assert h1 != h2


def test_verify_pin_accepts_correct_pin():
    pin_hash, pin_salt = auth.hash_pin("secret-pin")
    assert auth.verify_pin("secret-pin", pin_hash, pin_salt) is True


def test_verify_pin_rejects_wrong_pin():
    pin_hash, pin_salt = auth.hash_pin("secret-pin")
    assert auth.verify_pin("wrong-pin", pin_hash, pin_salt) is False


def test_verify_pin_rejects_malformed_salt():
    # Non-hex salt must fail closed, not raise.
    assert auth.verify_pin("1234", "deadbeef", "not-hex-salt") is False


# ---------------------------------------------------------------------------
# Session tokens (stateless, HMAC-signed)
# ---------------------------------------------------------------------------


def test_issue_and_verify_token_roundtrip():
    token = auth.issue_token("alice")
    assert auth.verify_token(token) == "alice"


def test_token_has_payload_dot_signature_shape():
    token = auth.issue_token("bob")
    assert token.count(".") == 1
    payload, sig = token.split(".")
    assert payload and sig


def test_verify_token_rejects_tampered_payload():
    token = auth.issue_token("alice")
    payload, sig = token.split(".")
    # Forge a different payload but keep the old signature.
    forged_payload = auth._b64e(f"admin|{int(time.time()) + 9999}".encode())
    tampered = f"{forged_payload}.{sig}"
    assert auth.verify_token(tampered) is None


def test_verify_token_rejects_tampered_signature():
    token = auth.issue_token("alice")
    payload, _ = token.split(".")
    assert auth.verify_token(f"{payload}.AAAAdeadbeef") is None


def test_verify_token_rejects_garbage():
    assert auth.verify_token("not-a-token") is None
    assert auth.verify_token("") is None
    assert auth.verify_token("a.b.c") is None


def test_verify_token_rejects_expired_token(monkeypatch):
    # Issue a token, then jump the clock past its TTL.
    token = auth.issue_token("alice")
    ttl_seconds = auth.settings.auth_token_ttl_hours * 3600
    future = time.time() + ttl_seconds + 60  # capture real time BEFORE patching
    monkeypatch.setattr(auth.time, "time", lambda: future)
    assert auth.verify_token(token) is None


def test_user_id_with_separator_char_roundtrips():
    # user_id|expiry is rsplit on the last '|', so a '|' inside the id survives.
    token = auth.issue_token("weird|name")
    assert auth.verify_token(token) == "weird|name"


# ---------------------------------------------------------------------------
# Authorization header parsing
# ---------------------------------------------------------------------------


def test_user_from_authorization_valid_bearer():
    token = auth.issue_token("carol")
    assert auth.user_from_authorization(f"Bearer {token}") == "carol"


def test_user_from_authorization_is_scheme_case_insensitive():
    token = auth.issue_token("carol")
    assert auth.user_from_authorization(f"bearer {token}") == "carol"


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "Basic abc",          # wrong scheme
        "Bearer",             # missing token
        "Token xyz",          # not bearer
    ],
)
def test_user_from_authorization_rejects_bad_headers(header):
    assert auth.user_from_authorization(header) is None
