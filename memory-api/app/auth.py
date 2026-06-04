"""Lightweight multi-user auth: PIN hashing + HMAC-signed session tokens.

No external dependencies — uses only the Python stdlib. PINs are stored as
pbkdf2-sha256 hashes with a per-user salt. Sessions are stateless: a token is
`base64url(user_id|expiry).base64url(hmac)`, signed with settings.auth_secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

from fastapi import Header, HTTPException

from app.config import settings

_PBKDF2_ROUNDS = 120_000


# ---------------------------------------------------------------------------
# PIN hashing
# ---------------------------------------------------------------------------


def hash_pin(pin: str) -> tuple[str, str]:
    """Return (hash_hex, salt_hex) for a PIN using pbkdf2-sha256."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, _PBKDF2_ROUNDS)
    return digest.hex(), salt.hex()


def verify_pin(pin: str, pin_hash: str, pin_salt: str) -> bool:
    """Constant-time check of a PIN against a stored hash + salt."""
    try:
        salt = bytes.fromhex(pin_salt)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, _PBKDF2_ROUNDS)
    return hmac.compare_digest(digest.hex(), pin_hash)


# ---------------------------------------------------------------------------
# Session tokens (stateless, HMAC-signed)
# ---------------------------------------------------------------------------


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(payload: str) -> str:
    mac = hmac.new(
        settings.effective_auth_secret.encode(), payload.encode(), hashlib.sha256
    ).digest()
    return _b64e(mac)


def issue_token(user_id: str) -> str:
    """Issue a signed session token for a user, valid for the configured TTL."""
    expiry = int(time.time()) + settings.auth_token_ttl_hours * 3600
    payload = _b64e(f"{user_id}|{expiry}".encode())
    return f"{payload}.{_sign(payload)}"


def verify_token(token: str) -> str | None:
    """Validate a token's signature and expiry. Return user_id or None."""
    try:
        payload, sig = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(payload)):
        return None
    try:
        user_id, expiry_str = _b64d(payload).decode().rsplit("|", 1)
        expiry = int(expiry_str)
    except (ValueError, UnicodeDecodeError):
        return None
    if time.time() > expiry:
        return None
    return user_id


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def user_from_authorization(authorization: str | None) -> str | None:
    """Extract and validate a Bearer token from an Authorization header value."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return verify_token(parts[1].strip())


def require_user(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: return the authenticated user_id or raise 401."""
    user_id = user_from_authorization(authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id
