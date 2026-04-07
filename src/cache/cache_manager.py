import hashlib
from typing import Any, Optional
from cachetools import TTLCache


class LLMAnswerCacheManager:
    """
    In-memory cache manager for storing LLM responses with TTL expiration.

    Uses a TTLCache to store question-response pairs for faster retrieval
    and reduced redundant LLM calls.

    Attributes:
        memory_cache (TTLCache): Cache storing responses with size and TTL constraints.
    """

    def __init__(
        self,
        memory_maxsize: int = 1000,
        memory_ttl_seconds: int = 300,
    ):
        """
        Initializes the cache manager with size and TTL settings.

        Args:
            memory_maxsize (int): Maximum number of items in cache.
            memory_ttl_seconds (int): Time-to-live for each cache entry in seconds.
        """
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
        """
        Generates a unique cache key based on question and namespace.

        Args:
            question (str): Input query/question.
            namespace (str): Logical grouping for cache separation.

        Returns:
            str: SHA-256 hash key representing the input.
        """
        raw = f"{namespace}:{question}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # -----------------------------------------
    # GET
    # -----------------------------------------
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a cached value for the given key.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Any]: Cached value if present, otherwise None.
        """
        return self.memory_cache.get(key)

    # -----------------------------------------
    # SET
    # -----------------------------------------
    async def set(
        self,
        key: str,
        value: Any,
    ):
        """
        Stores a value in the cache under the given key.

        Args:
            key (str): Cache key.
            value (Any): Value to store.
        """
        
        self.memory_cache[key] = value

    # -----------------------------------------
    # CLEAR
    # -----------------------------------------
    async def clear(self):
        """
        Clears all entries from the cache.

        Returns:
            None
        """
        
        self.memory_cache.clear()