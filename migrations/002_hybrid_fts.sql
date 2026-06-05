-- Migration 002: Add tsvector columns + GIN indexes for hybrid (dense + BM25) retrieval.
-- Uses ADD COLUMN IF NOT EXISTS — safe to run on fresh installs where init.sql
-- already created the columns via docker-entrypoint-initdb.d.

ALTER TABLE conversation_messages
    ADD COLUMN IF NOT EXISTS content_fts tsvector
    GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED;

ALTER TABLE semantic_memory
    ADD COLUMN IF NOT EXISTS content_fts tsvector
    GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_conv_content_fts
    ON conversation_messages USING GIN (content_fts);

CREATE INDEX IF NOT EXISTS idx_semantic_content_fts
    ON semantic_memory USING GIN (content_fts);
