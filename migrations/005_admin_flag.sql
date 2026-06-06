-- Phase 4.1: admin flag for the memory inspector.
-- Inspector data endpoints require an authenticated user whose profile has is_admin.

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;
