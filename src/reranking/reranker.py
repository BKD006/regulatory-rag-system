from typing import List
from sentence_transformers import CrossEncoder

from utils.models import RetrievedChunk
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class CrossEncoderReranker:
    """
    Reranks retrieved document chunks using a cross-encoder model.

    Uses a transformer-based cross-encoder to compute relevance scores
    between a query and each chunk, improving retrieval quality.

    Attributes:
        model_name (str): Name of the cross-encoder model.
        top_k (int): Number of top chunks to return after reranking.
        min_score (float): Minimum score threshold (currently unused).
        model (CrossEncoder): Loaded cross-encoder model instance.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 6,
        min_score: float = 0,
        device: str = "cpu",
    ):
        """
        Initializes the CrossEncoderReranker with the specified model.

        Args:
            model_name (str): HuggingFace model name for cross-encoder.
            top_k (int): Number of top-ranked chunks to return.
            min_score (float): Minimum score threshold for filtering (not enforced).
            device (str): Device to load the model on ("cpu" or "cuda").

        Raises:
            RegulatoryRAGException: If model initialization fails.
        """

        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score

        try:
            # Load cross-encoder model (uses transformers + torch)
            self.model = CrossEncoder(
                model_name,
                device=device,
            )

            log.info(
                "reranker_initialized",
                model=model_name,
                top_k=top_k,
                min_score=min_score,
                device=device,
            )

        except Exception as e:
            log.error(
                "reranker_initialization_failed",
                model=model_name,
                error=str(e),
            )
            raise RegulatoryRAGException(e)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Reranks retrieved chunks based on relevance to the query.

        Args:
            query (str): User query.
            chunks (List[RetrievedChunk]): List of retrieved chunks.

        Returns:
            List[RetrievedChunk]: Top-k reranked chunks sorted by relevance.

        Notes:
            - Falls back to original chunks if reranking fails.
            - Always returns at most top_k chunks.
        """

        if not chunks:
            return []

        log.info(
            "reranking_started",
            input_chunks=len(chunks),
        )

        try:
            # Create (query, chunk) pairs
            pairs = [(query, c.content) for c in chunks]

            # Compute scores
            scores = self.model.predict(pairs)

            reranked: List[RetrievedChunk] = []

            for chunk, score in zip(chunks, scores):

                metadata = dict(chunk.metadata or {})
                metadata.update(
                    {
                        "rerank_score": float(score),
                        "reranked": True,
                    }
                )

                reranked.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        score=float(score),
                        source=chunk.source,
                        metadata=metadata,
                    )
                )

            # Sort by score
            reranked.sort(key=lambda c: c.score, reverse=True)

            # Always take top_k
            final = reranked[: self.top_k]

            # SAFETY FALLBACK (critical)
            if not final:
                log.warning("reranker_empty_fallback")
                return chunks[: self.top_k]

            log.info(
                "reranking_completed",
                input_chunks=len(chunks),
                output_chunks=len(final),
            )

            return final

        except Exception as e:
            log.error(
                "reranking_failed",
                input_chunks=len(chunks),
                error=str(e),
            )

            # HARD FALLBACK
            return chunks[: self.top_k]