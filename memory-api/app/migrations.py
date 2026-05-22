"""Programmatic migration runner.

Migrations are also mounted at /docker-entrypoint-initdb.d for initial provisioning,
but this runner handles upgrades and can be called from app startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.pgvector_client import get_pool

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


async def run_migrations() -> None:
    """Apply pending migrations. Idempotent — skips already-applied migrations."""
    if not MIGRATIONS_DIR.exists():
        logger.warning("Migrations directory not found: %s", MIGRATIONS_DIR)
        return

    pool = await get_pool()

    # Ensure the migration tracking table exists
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                name TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        migration_files = sorted(
            p for p in MIGRATIONS_DIR.glob("*.sql")
            if not p.name.endswith(".rollback.sql")
        )

        for path in migration_files:
            name = path.name
            already = await conn.fetchval(
                "SELECT 1 FROM _migrations WHERE name = $1", name
            )
            if already:
                continue

            sql = path.read_text()
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO _migrations (name) VALUES ($1)", name
                )

            logger.info("Applied migration: %s", name)


async def rollback_migration(name: str) -> None:
    """Roll back a single migration by name."""
    rollback_path = MIGRATIONS_DIR / name.replace(".sql", ".rollback.sql")
    if not rollback_path.exists():
        raise FileNotFoundError(f"Rollback script not found: {rollback_path}")

    pool = await get_pool()
    sql = rollback_path.read_text()

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "DELETE FROM _migrations WHERE name = $1", name
            )

    logger.info("Rolled back migration: %s", name)
