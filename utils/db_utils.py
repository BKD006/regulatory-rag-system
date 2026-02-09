"""Unified async DB utilities.

This file consolidates the previous `db_utils.py` and `db_utils_async.py` helpers
into a single module exposing both the lifecycle helpers used by the ingestion
pipeline and the document query helpers used by the API and UI.
"""

import os
import asyncpg
import logging
from typing import Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Database configuration
# ------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

# single pool used by all helpers
_pool: Optional[asyncpg.Pool] = None
# Backward-compatible name used by older modules (e.g. ingest.py)
# Keep this in sync with `_pool` when initializing/closing the pool.
db_pool: Optional[asyncpg.Pool] = None


async def init_db_pool():
    """Initialize the global asyncpg pool.

    This name is used by the API startup handler. For backward compatibility
    `initialize_database` is provided as an alias.
    """
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
        )
        logger.info("PostgreSQL connection pool initialized")
        # expose the pool under the legacy name
        global db_pool
        db_pool = _pool


async def initialize_database():
    return await init_db_pool()


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_db_pool() first.")
    return _pool


async def close_db_pool():
    """Close the global connection pool if it exists."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        # keep legacy symbol in sync
        global db_pool
        db_pool = None
        logger.info("PostgreSQL connection pool closed")


async def close_database():
    return await close_db_pool()


# ------------------------------------------------------------------
# Document queries / helpers
# ------------------------------------------------------------------


async def get_document_by_hash(file_hash: str):
    """Return document row if hash already exists."""
    pool = get_pool()
    async with pool.acquire() as conn:
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
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT id::text
            FROM documents
            WHERE source = $1
            """,
            source,
        )


async def delete_document_and_chunks(document_id: str):
    """
    Delete document and all its chunks.
    Cascades via FK constraint.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                DELETE FROM documents
                WHERE id = $1::uuid
                """,
                document_id,
            )
            logger.info(f"Deleted document and chunks: {document_id}")


async def list_documents() -> List[Dict[str, str]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, title
            FROM documents
            ORDER BY title
            """
        )
        return [{"id": r["id"], "title": r["title"]} for r in rows]


async def list_document_titles() -> List[str]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT title
            FROM documents
            ORDER BY title
            """
        )
        return [r["title"] for r in rows]


async def delete_document_by_title(title: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id
                FROM documents
                WHERE title = $1
                """,
                title,
            )

            if not row:
                raise ValueError(f"No document found with title '{title}'")

            await conn.execute(
                """
                DELETE FROM documents
                WHERE id = $1
                """,
                row["id"],
            )
