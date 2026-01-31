"""
Cross-encoder reranker for RAG.
Improves precision after retrieval.
Production-safe version.
"""

from typing import List
from sentence_transformers import CrossEncoder

from utils.models import RetrievedChunk


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
        """
        Args:
            model_name: Cross-encoder model
            top_k: Max chunks after reranking
            min_score: Minimum relevance score
            device: cpu / cuda
        """
        self.model = CrossEncoder(
            model_name,
            device=device,
        )
        self.top_k = top_k
        self.min_score = min_score

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Args:
            query: User question
            chunks: RetrievedChunk list

        Returns:
            Reranked and filtered RetrievedChunk list
        """
        if not chunks:
            return []

        # Prepare pairs
        pairs = [(query, c.content) for c in chunks]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        reranked = []
        for chunk, score in zip(chunks, scores):
            if score < self.min_score:
                continue

            # Enrich metadata safely
            metadata = dict(chunk.metadata or {})
            metadata["rerank_score"] = float(score)
            metadata["reranked"] = True

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

        # Sort by rerank score
        reranked.sort(key=lambda c: c.score, reverse=True)

        return reranked[: self.top_k]
