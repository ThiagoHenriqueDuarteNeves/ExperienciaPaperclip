"""Create or promote an admin user for the memory inspector.

The inspector's data endpoints require a profile with is_admin=TRUE (see
app/inspect_api.require_admin). This script creates such a profile, or promotes
an existing one — idempotently — and (re)sets its PIN to the one you pass, so you
always end up with a known admin credential.

Usage (inside the running container, so it shares the DB + migrations):

    docker compose exec memory-api python -m app.scripts.create_admin <user_id> <pin> [display_name]

The migration 005_admin_flag.sql (adds user_profiles.is_admin) must have been
applied first — it runs automatically on memory-api startup.
"""

from __future__ import annotations

import asyncio
import sys

from app.auth import hash_pin
from app.pgvector_client import get_pool


async def upsert_admin(user_id: str, pin: str, display_name: str | None) -> bool:
    """Insert or promote `user_id` as an admin with the given PIN.

    Returns True if a new profile was created, False if an existing one was
    promoted/updated.
    """
    pin_hash, pin_salt = hash_pin(pin)
    display_name = display_name or user_id
    pool = await get_pool()
    async with pool.acquire() as conn:
        existed = await conn.fetchval(
            "SELECT TRUE FROM user_profiles WHERE user_id = $1", user_id
        )
        await conn.execute(
            """INSERT INTO user_profiles (user_id, display_name, pin_hash, pin_salt, is_admin)
               VALUES ($1, $2, $3, $4, TRUE)
               ON CONFLICT (user_id) DO UPDATE
                 SET pin_hash = EXCLUDED.pin_hash,
                     pin_salt = EXCLUDED.pin_salt,
                     is_admin = TRUE""",
            user_id, display_name, pin_hash, pin_salt,
        )
    return not existed


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(
            "usage: python -m app.scripts.create_admin <user_id> <pin> [display_name]",
            file=sys.stderr,
        )
        return 2
    user_id, pin = args[0], args[1]
    display_name = args[2] if len(args) > 2 else None

    created = asyncio.run(upsert_admin(user_id, pin, display_name))
    verb = "created" if created else "promoted/updated"
    print(f"OK: admin {verb} — user_id={user_id!r}, is_admin=TRUE")
    print("Login at /inspect with this user_id + PIN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
