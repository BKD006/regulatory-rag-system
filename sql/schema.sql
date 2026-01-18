-- =========================================================
-- Extensions
-- =========================================================
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS vector;

-- =========================================================
-- Documents table
-- =========================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(),

    CONSTRAINT uq_documents_file_hash UNIQUE (file_hash)
);

-- Helpful index for update detection
CREATE INDEX IF NOT EXISTS idx_documents_source
ON documents (source);

-- =========================================================
-- Chunks table
-- =========================================================
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT now(),
    CONSTRAINT fk_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES documents (id)
        ON DELETE CASCADE
);

-- =========================================================
-- Vector index (cosine similarity)
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- =========================================================
-- Optional performance indexes
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index
ON chunks (chunk_index);
