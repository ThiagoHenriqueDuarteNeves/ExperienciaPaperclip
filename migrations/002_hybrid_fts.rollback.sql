-- Rollback for migration 002: remove hybrid FTS columns and indexes.

DROP INDEX IF EXISTS idx_conv_content_fts;
DROP INDEX IF EXISTS idx_semantic_content_fts;
ALTER TABLE conversation_messages DROP COLUMN IF EXISTS content_fts;
ALTER TABLE semantic_memory DROP COLUMN IF EXISTS content_fts;
