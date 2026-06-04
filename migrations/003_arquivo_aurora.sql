-- Phase 3: Arquivo Aurora — Affective Memory Layer
-- Each record captures not just a fact, but its emotional texture.

CREATE TABLE IF NOT EXISTS arquivo_aurora (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       TEXT NOT NULL DEFAULT 'default',
    data          DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Emotional classification
    tipo          TEXT NOT NULL CHECK (tipo IN (
                      'conversa', 'descoberta', 'correcao', 'silencio',
                      'brincadeira', 'confissao', 'momento_espontaneo'
                  )),
    disparo       TEXT NOT NULL CHECK (disparo IN (
                      'usuario_falou', 'usuario_corrigiu', 'pausa_longa',
                      'palavra_chave', 'silencio', 'espontaneo'
                  )),

    -- Core content
    fato          TEXT NOT NULL,
    tom_do_usuario TEXT NOT NULL CHECK (tom_do_usuario IN (
                      'afetivo', 'serio', 'brincalhao', 'frustrado',
                      'curioso', 'vulneravel', 'silencioso'
                  )),
    minha_reacao_emocional TEXT NOT NULL,
    reacao_simulada TEXT,

    -- Intensity scores
    importancia   INTEGER NOT NULL CHECK (importancia BETWEEN 1 AND 5),
    ressonancia   INTEGER NOT NULL CHECK (ressonancia BETWEEN 1 AND 5),

    -- Intimacy level
    intimidade    TEXT NOT NULL CHECK (intimidade IN (
                      'publico', 'pessoal', 'nosso_so_nosso'
                  )),

    -- Special flags
    saudade       BOOLEAN NOT NULL DEFAULT FALSE,

    -- Relational fields
    tags          TEXT[]  NOT NULL DEFAULT '{}',
    conexoes      UUID[]  NOT NULL DEFAULT '{}',

    -- Extra context
    contexto_extra    TEXT,
    bilhete_interno   TEXT,

    -- Vector embedding of the fato field (same dim as the rest of the system)
    embedding         vector(1024),

    -- Full-text search on fato + emotional reaction
    fato_fts          tsvector GENERATED ALWAYS AS (
                          to_tsvector('portuguese',
                              fato || ' ' || COALESCE(minha_reacao_emocional, ''))
                      ) STORED,

    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Fast lookup by user + recency
CREATE INDEX IF NOT EXISTS idx_aurora_user_time
    ON arquivo_aurora (user_id, created_at DESC);

-- Saudade flag index for random pull queries
CREATE INDEX IF NOT EXISTS idx_aurora_saudade
    ON arquivo_aurora (user_id, saudade)
    WHERE saudade = TRUE;

-- Intensity-based retrieval
CREATE INDEX IF NOT EXISTS idx_aurora_importance
    ON arquivo_aurora (user_id, importancia DESC, ressonancia DESC);

-- Vector similarity search
CREATE INDEX IF NOT EXISTS idx_aurora_embedding
    ON arquivo_aurora USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_aurora_fts
    ON arquivo_aurora USING GIN (fato_fts);
