"""
Simple embedding generator for RAG.

- Uses ModelLoader.load_embeddings()
- Async-safe (runs sync calls in thread)
- No retries / caching / batching complexity
"""

import asyncio
import logging
from typing import List, Optional, Callable
from datetime import datetime

from src.chunking.chunker import DocumentChunk
from utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Minimal embedding generator.
    """

    def __init__(self):
        # Single source of truth
        self.embedding_client = ModelLoader().load_embeddings()

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    async def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        embed_documents is sync → run in thread
        """
        return await asyncio.to_thread(
            self.embedding_client.embed_documents,
            texts,
        )

    async def _embed_query(self, text: str) -> List[float]:
        """
        embed_query is sync → run in thread
        """
        return await asyncio.to_thread(
            self.embedding_client.embed_query,
            text,
        )

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DocumentChunk]:
        """
        Attach embeddings to chunks.
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks")

        texts = [c.content for c in chunks]
        vectors = await self._embed_documents(texts)

        embedded_chunks: List[DocumentChunk] = []

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
            chunk.embedding = vector
            chunk.metadata.update(
                {
                    "embedding_generated_at": datetime.now().isoformat(),
                }
            )
            embedded_chunks.append(chunk)

            if progress_callback:
                progress_callback(idx, len(chunks))

        return embedded_chunks

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for retrieval query.
        """
        return await self._embed_query(query)


# -------------------------------------------------
# Factory
# -------------------------------------------------

def create_embedder() -> EmbeddingGenerator:
    """
    Factory used across ingestion + retrieval.
    """
    return EmbeddingGenerator()
