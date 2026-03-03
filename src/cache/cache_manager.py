import hashlib
import json
import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta

from cachetools import TTLCache
from utils import db_utils


class LLMAnswerCacheManager:
    """
    Hybrid cache:
    L1 → In-memory TTL cache
    L2 → PostgreSQL persistent cache
    """

    def __init__(
        self,
        memory_maxsize: int = 1000,
        memory_ttl_seconds: int = 300,
        postgres_ttl_minutes: int = 60,
    ):
        self.memory_cache = TTLCache(
            maxsize=memory_maxsize,
            ttl=memory_ttl_seconds,
        )
        self.postgres_ttl = timedelta(minutes=postgres_ttl_minutes)
        self._lock = asyncio.Lock()
        self._knowledge_cache = TTLCache(maxsize=1, ttl=5)

    # -----------------------------------------
    # Key generator
    # -----------------------------------------
    async def make_key(
        self,
        question: str,
        namespace: str = "default"
    ) -> str:
        knowledge_version = await self._get_knowledge_version()

        raw = f"{namespace}:{knowledge_version}:{question}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # -----------------------------------------
    # GET
    # -----------------------------------------
    async def get(self, key: str) -> Optional[Any]:

        # 🔹 L1 - Memory
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 🔹 L2 - Postgres
        async with db_utils.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT response, created_at
                FROM public.query_cache
                WHERE cache_key = $1
                """,
                key,
            )

            if not row:
                return None

            created_at = row["created_at"]

            # TTL validation
            if datetime.utcnow() - created_at > self.postgres_ttl:
                await conn.execute(
                    "DELETE FROM public.query_cache WHERE cache_key = $1",
                    key,
                )
                return None

            # 🔥 FIX HERE
            response = json.loads(row["response"])

            # Store back to memory cache
            self.memory_cache[key] = response

            return response

    # -----------------------------------------
    # SET
    # -----------------------------------------
    async def set(
        self,
        key: str,
        question: str,
        value: Any,
        filters: Optional[dict] = None,
    ):
        async with self._lock:

            # Always fetch knowledge version here
            knowledge_version = await self._get_knowledge_version()

            # L1
            self.memory_cache[key] = value

            # L2
            async with db_utils.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO public.query_cache
                    (cache_key, question, knowledge_version, filters, response)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (cache_key)
                    DO UPDATE SET
                        response = EXCLUDED.response,
                        knowledge_version = EXCLUDED.knowledge_version,
                        filters = EXCLUDED.filters,
                        created_at = now()
                    """,
                    key,
                    question,
                    knowledge_version,   # ✅ no longer null
                    json.dumps(filters or {}),
                    json.dumps(value),
                )

    # -----------------------------------------
    # Clear cache
    # -----------------------------------------
    async def clear(self):
        self.memory_cache.clear()
        async with db_utils.db_pool.acquire() as conn:
            await conn.execute("DELETE FROM public.query_cache")
    
    # -----------------------------------------
    # Knowledge Version
    # -----------------------------------------
    async def _get_knowledge_version(self) -> str:
        """
        Returns a fingerprint of current knowledge state.
        Uses MAX(updated_at) for lightweight invalidation.
        """
        if "version" in self._knowledge_cache:
            return self._knowledge_cache["version"]
        async with db_utils.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(updated_at) AS latest FROM public.documents"
            )

            if not row or not row["latest"]:
                return "no_docs"

            return row["latest"].isoformat()