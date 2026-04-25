import asyncio
from typing import List, Optional, Callable
from datetime import datetime

from utils.models import DocumentChunk
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class EmbeddingGenerator:
    """
    Generates vector embeddings for documents and queries.

    Uses an underlying embedding model (e.g., Bedrock/OpenAI) to convert text
    into numerical vectors for downstream retrieval tasks.

    Attributes:
        embedding_client (object): Embedding model client used for vector generation.
    """
    
    def __init__(self):
        """
        Initializes the EmbeddingGenerator by loading the embedding model.

        Raises:
            RegulatoryRAGException: If embedding client initialization fails.
        """
        
        try:
            # Load embedding model (Bedrock)
            self.embedding_client = ModelLoader().load_embeddings()

            log.info("embedding_client_initialized")

        except Exception as e:
            log.error("embedding_client_init_failed", error=str(e))
            raise RegulatoryRAGException(e)

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    async def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text inputs.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors corresponding to input texts.
        """
        return await asyncio.to_thread(
            self.embedding_client.embed_documents,
            texts,
        )

    async def _embed_query(self, text: str) -> List[float]:
        """
        Generates embedding for a single query string.

        Args:
            text (str): Query text.

        Returns:
            List[float]: Embedding vector for the query.
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
        Generates embeddings for a list of document chunks and attaches them.

        Args:
            chunks (List[DocumentChunk]): List of document chunks to embed.
            progress_callback (Optional[Callable[[int, int], None]]):
                Optional function to report progress with signature (current, total).

        Returns:
            List[DocumentChunk]: List of chunks with embeddings attached.

        Raises:
            RegulatoryRAGException: If embedding generation fails.
        """

        if not chunks:
            return []

        log.info(
            "embedding_started",
            chunk_count=len(chunks),
        )

        start_time = datetime.now()

        try:
            # Extract text content from chunks
            texts = [c.content for c in chunks]

            # Generate embeddings
            vectors = await self._embed_documents(texts)

            # Attach embeddings to chunks
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
                chunk.embedding = vector

                # Add metadata for traceability
                chunk.metadata.update(
                    {
                        "embedding_generated_at": datetime.now().isoformat(),
                    }
                )

                # Optional progress tracking
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
        Generates embedding for a single query.

        Args:
            query (str): Input query string.

        Returns:
            List[float]: Embedding vector for the query.

        Raises:
            RegulatoryRAGException: If embedding generation fails.
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
    Factory function to create an EmbeddingGenerator instance.

    Returns:
        EmbeddingGenerator: Initialized embedding generator.
    """
    return EmbeddingGenerator()