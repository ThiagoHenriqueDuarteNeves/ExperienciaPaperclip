-- Phase 4: User profiles for multi-user auth (PIN validated server-side)
-- The PIN is never stored in plaintext — only a pbkdf2 hash + per-user salt.

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT,
    pin_hash     TEXT NOT NULL,
    pin_salt     TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
