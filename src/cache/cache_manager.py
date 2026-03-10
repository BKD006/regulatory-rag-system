import hashlib
from typing import Any, Optional
from cachetools import TTLCache

class LLMAnswerCacheManager:
    """
    Runtime-only in-memory cache using TTLCache.
    Cache is cleared when application restarts.
    """

    def __init__(
        self,
        memory_maxsize: int = 1000,
        memory_ttl_seconds: int = 300,
    ):
        self.memory_cache = TTLCache(
            maxsize=memory_maxsize,
            ttl=memory_ttl_seconds,
        )

    # -----------------------------------------
    # Key generator
    # -----------------------------------------
    def make_key(
        self,
        question: str,
        namespace: str = "default"
    ) -> str:

        raw = f"{namespace}:{question}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # -----------------------------------------
    # GET
    # -----------------------------------------
    async def get(self, key: str) -> Optional[Any]:

        if key in self.memory_cache:
            return self.memory_cache[key]

        return None

    # -----------------------------------------
    # SET
    # -----------------------------------------
    async def set(
        self,
        key: str,
        value: Any,
    ):
        self.memory_cache[key] = value

    # -----------------------------------------
    # CLEAR
    # -----------------------------------------
    async def clear(self):
        self.memory_cache.clear()