"""
Cross-encoder reranker for RAG.

- Improves precision after retrieval
- Production-safe (summary logging only)
"""

from typing import List
from sentence_transformers import CrossEncoder
from utils.models import RetrievedChunk
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class CrossEncoderReranker:
    """
    Cross-encoder reranker with:
    - score thresholding
    - metadata enrichment
    - top-k control
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 6,
        min_score: float = 0.05,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score

        try:
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
        Rerank retrieved chunks using cross-encoder.
        """

        if not chunks:
            return []

        log.info(
            "reranking_started",
            input_chunks=len(chunks),
        )

        try:
            pairs = [(query, c.content) for c in chunks]
            scores = self.model.predict(pairs)

            reranked: List[RetrievedChunk] = []

            for chunk, score in zip(chunks, scores):
                if score < self.min_score:
                    continue

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

            reranked.sort(key=lambda c: c.score, reverse=True)
            final = reranked[: self.top_k]

            log.info(
                "reranking_completed",
                input_chunks=len(chunks),
                passed_threshold=len(reranked),
                output_chunks=len(final),
            )

            return final

        except Exception as e:
            log.error(
                "reranking_failed",
                input_chunks=len(chunks),
                error=str(e),
            )
            raise RegulatoryRAGException(e)
