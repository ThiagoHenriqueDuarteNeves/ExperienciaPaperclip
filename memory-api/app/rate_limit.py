"""Shared slowapi limiter.

Lives in its own module so routers can apply @limiter.limit(...) without importing
from app.main (which would be circular). main.py wires it onto the app
(app.state.limiter + the RateLimitExceeded handler).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
