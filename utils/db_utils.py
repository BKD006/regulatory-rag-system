"""
Database utilities for PostgreSQL + pgvector.

Responsibilities:
- Connection pooling
- Schema initialization
- Idempotency helpers (hash / source checks)
- Safe cleanup on re-ingestion
"""

import os
import asyncpg
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Database configuration
# ------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

db_pool: Optional[asyncpg.Pool] = None


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

async def initialize_database():
    """Initialize connection pool."""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
        )
        logger.info("PostgreSQL connection pool initialized")


async def close_database():
    """Close connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None
        logger.info("PostgreSQL connection pool closed")


# ------------------------------------------------------------------
# Document lookup helpers
# ------------------------------------------------------------------

async def get_document_by_hash(file_hash: str):
    """Return document row if hash already exists."""
    async with db_pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT id::text
            FROM documents
            WHERE file_hash = $1
            """,
            file_hash,
        )


async def get_document_by_source(source: str):
    """Return document row if source already exists."""
    async with db_pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT id::text
            FROM documents
            WHERE source = $1
            """,
            source,
        )


# ------------------------------------------------------------------
# Cleanup helpers
# ------------------------------------------------------------------

async def delete_document_and_chunks(document_id: str):
    """
    Delete document and all its chunks.
    Cascades via FK constraint.
    """
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                DELETE FROM documents
                WHERE id = $1::uuid
                """,
                document_id,
            )
            logger.info(f"Deleted document and chunks: {document_id}")
