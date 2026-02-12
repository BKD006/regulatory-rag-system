"""
Unified async DB utilities.

- Owns DB pool lifecycle
- Exposes document-level helpers
- NO query-level logging
"""

import os
import asyncpg
from typing import Optional, List, Dict
from dotenv import load_dotenv
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

_pool: Optional[asyncpg.Pool] = None
db_pool: Optional[asyncpg.Pool] = None  # backward compatibility

# ------------------------------------------------------------------
# Pool lifecycle
# ------------------------------------------------------------------

async def init_db_pool():
    global _pool, db_pool

    if _pool is not None:
        return

    try:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
        )
        db_pool = _pool
        log.info("db_pool_initialized", min_size=1, max_size=10)

    except Exception as e:
        log.error("db_pool_initialization_failed", error=str(e))
        raise RegulatoryRAGException(e)


async def close_db_pool():
    global _pool, db_pool

    if _pool is None:
        return

    try:
        await _pool.close()
        _pool = None
        db_pool = None
        log.info("db_pool_closed")

    except Exception as e:
        log.error("db_pool_close_failed", error=str(e))
        raise RegulatoryRAGException(e)


# backward-compatible aliases
async def initialize_database():
    await init_db_pool()


async def close_database():
    await close_db_pool()


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RegulatoryRAGException(
            "DB pool not initialized. Call init_db_pool() first."
        )
    return _pool

# ------------------------------------------------------------------
# Document helpers
# ------------------------------------------------------------------

async def get_document_by_hash(file_hash: str):
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
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            try:
                await conn.execute(
                    """
                    DELETE FROM documents
                    WHERE id = $1::uuid
                    """,
                    document_id,
                )
                log.info(
                    "document_deleted_with_chunks",
                    document_id=document_id,
                )

            except Exception as e:
                log.error(
                    "document_delete_failed",
                    document_id=document_id,
                    error=str(e),
                )
                raise RegulatoryRAGException(e)


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

            try:
                await conn.execute(
                    """
                    DELETE FROM documents
                    WHERE id = $1
                    """,
                    row["id"],
                )

                log.info(
                    "document_deleted_by_title",
                    title=title,
                )

            except Exception as e:
                log.error(
                    "delete_document_by_title_failed",
                    title=title,
                    error=str(e),
                )
                raise RegulatoryRAGException(e)

async def get_chunk_by_id(chunk_id: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 
                id::text AS chunk_id,
                document_id::text,
                content,
                metadata,
                token_count
            FROM chunks
            WHERE id = $1::uuid
            """,
            chunk_id,
        )

        return row
