"""
Simple embedding generator for RAG.

- Uses ModelLoader.load_embeddings()
- Async-safe (runs sync calls in thread)
- Batch-level logging only
"""

import asyncio
from typing import List, Optional, Callable
from datetime import datetime
from utils.models import DocumentChunk
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class EmbeddingGenerator:
    """
    Minimal, production-safe embedding generator.
    """

    def __init__(self):
        try:
            self.embedding_client = ModelLoader().load_embeddings()
            log.info("embedding_client_initialized")
        except Exception as e:
            log.error("embedding_client_init_failed", error=str(e))
            raise RegulatoryRAGException(e)

    # -------------------------------------------------
    # Internal helpers (NO logging here)
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

        log.info(
            "embedding_started",
            chunk_count=len(chunks),
        )

        start_time = datetime.now()

        try:
            texts = [c.content for c in chunks]
            vectors = await self._embed_documents(texts)

            for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
                chunk.embedding = vector
                chunk.metadata.update(
                    {
                        "embedding_generated_at": datetime.now().isoformat(),
                    }
                )

                if progress_callback:
                    progress_callback(idx, len(chunks))

            duration_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            log.info(
                "embedding_completed",
                chunk_count=len(chunks),
                duration_ms=int(duration_ms),
            )

            return chunks

        except Exception as e:
            log.error(
                "embedding_failed",
                chunk_count=len(chunks),
                error=str(e),
            )
            raise RegulatoryRAGException(e)

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for retrieval query.
        """

        try:
            return await self._embed_query(query)
        except Exception as e:
            log.error("query_embedding_failed", error=str(e))
            raise RegulatoryRAGException(e)


# -------------------------------------------------
# Factory
# -------------------------------------------------

def create_embedder() -> EmbeddingGenerator:
    """
    Factory used across ingestion + retrieval.
    """
    return EmbeddingGenerator()
