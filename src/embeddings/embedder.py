"""
Document embedding generation for vector search.

UPDATED:
- Uses LangChain Embeddings interface (AWS Bedrock Titan embeddings)
- No OpenAI dependency anymore
- Works with providers.py Option A (LangChain providers)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import hashlib
from dotenv import load_dotenv

from src.chunking.chunker import DocumentChunk
from utils.providers import get_embedding_client, get_embedding_model


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize embedding client from providers (LangChain BedrockEmbeddings)
embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()


class EmbeddingGenerator:
    """Generates embeddings for document chunks (Bedrock Titan via LangChain)."""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 64,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize embedding generator.

        Args:
            model: Embedding model id (e.g. amazon.titan-embed-text-v1)
            batch_size: Number of texts per batch
            max_retries: Retries on failures
            retry_delay: Delay between retries
        """
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Titan embeddings dimensions are typically 1536 for v1
        # But dimension is not required by BedrockEmbeddings calls.
        self.default_dimensions = int(os.getenv("EMBEDDING_DIM", "1536"))

    async def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        BedrockEmbeddings.embed_documents is sync.
        We run it in a thread so async pipeline does not block.
        """
        return await asyncio.to_thread(embedding_client.embed_documents, texts)

    async def _embed_query(self, text: str) -> List[float]:
        """
        BedrockEmbeddings.embed_query is sync.
        Run it in a thread for async compatibility.
        """
        return await asyncio.to_thread(embedding_client.embed_query, text)

    def _normalize_text(self, text: str) -> str:
        """Basic cleanup to avoid Bedrock failures."""
        if not text:
            return ""
        text = text.strip()
        # Bedrock dislikes extremely long payloads sometimes – keep bounded.
        max_chars = int(os.getenv("EMBEDDING_MAX_CHARS", "12000"))
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        text = self._normalize_text(text)

        if not text:
            return [0.0] * self.default_dimensions

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await self._embed_query(text)
            except Exception as e:
                last_error = e
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Embedding single text failed attempt {attempt+1}/{self.max_retries}: {e}. Retrying in {delay}s")
                await asyncio.sleep(delay)

        logger.error(f"Failed to embed single text after retries: {last_error}")
        return [0.0] * self.default_dimensions

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        processed = [self._normalize_text(t) for t in texts]

        # Fill empty texts with placeholders to preserve alignment
        empty_indices = [i for i, t in enumerate(processed) if not t]
        for i in empty_indices:
            processed[i] = " "  # Titan can fail on empty string

        last_error = None
        for attempt in range(self.max_retries):
            try:
                vectors = await self._embed_documents(processed)

                # Replace placeholders with zero vectors
                for i in empty_indices:
                    vectors[i] = [0.0] * self.default_dimensions

                return vectors

            except Exception as e:
                last_error = e
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Batch embedding failed attempt {attempt+1}/{self.max_retries}: {e}. Retrying in {delay}s")
                await asyncio.sleep(delay)

        logger.error(f"Batch embedding failed after retries: {last_error}")
        # Fallback to individual
        return await self._process_individually(processed)

    async def _process_individually(self, texts: List[str]) -> List[List[float]]:
        """Fallback: embed one by one."""
        embeddings = []
        for text in texts:
            emb = await self.generate_embedding(text)
            embeddings.append(emb)
            await asyncio.sleep(0.05)
        return embeddings

    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[DocumentChunk]:
        """Attach embeddings to DocumentChunk objects."""
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks with model={self.model}")

        embedded_chunks: List[DocumentChunk] = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [c.content for c in batch_chunks]

            try:
                vectors = await self.generate_embeddings_batch(batch_texts)

                for chunk, vector in zip(batch_chunks, vectors):
                    embedded_chunk = DocumentChunk(
                        content=chunk.content,
                        index=chunk.index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "embedding_model": self.model,
                            "embedding_provider": "aws_bedrock_titan",
                            "embedding_generated_at": datetime.now().isoformat(),
                        },
                        token_count=chunk.token_count,
                    )
                    embedded_chunk.embedding = vector
                    embedded_chunks.append(embedded_chunk)

                current_batch = (i // self.batch_size) + 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)

                logger.info(f"Processed batch {current_batch}/{total_batches}")

            except Exception as e:
                logger.error(f"Failed to embed batch {i//self.batch_size + 1}: {e}")

                for chunk in batch_chunks:
                    chunk.metadata.update({
                        "embedding_error": str(e),
                        "embedding_model": self.model,
                        "embedding_provider": "aws_bedrock_titan",
                        "embedding_generated_at": datetime.now().isoformat(),
                    })
                    chunk.embedding = [0.0] * self.default_dimensions
                    embedded_chunks.append(chunk)

        return embedded_chunks

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        return await self.generate_embedding(query)

    def get_embedding_dimension(self) -> int:
        """Return default embedding dimension."""
        return self.default_dimensions


# Cache for embeddings
class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, List[float]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.max_size = max_size

    def get(self, text: str) -> Optional[List[float]]:
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            self.access_times[text_hash] = datetime.now()
            return self.cache[text_hash]
        return None

    def put(self, text: str, embedding: List[float]):
        text_hash = self._hash_text(text)

        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[text_hash] = embedding
        self.access_times[text_hash] = datetime.now()

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()


def create_embedder(
    model: str = EMBEDDING_MODEL,
    use_cache: bool = True,
    **kwargs
) -> EmbeddingGenerator:
    """Factory function."""
    embedder = EmbeddingGenerator(model=model, **kwargs)

    if use_cache:
        cache = EmbeddingCache()
        original_generate = embedder.generate_embedding

        async def cached_generate(text: str) -> List[float]:
            cached = cache.get(text)
            if cached is not None:
                return cached
            embedding = await original_generate(text)
            cache.put(text, embedding)
            return embedding

        embedder.generate_embedding = cached_generate

    return embedder
