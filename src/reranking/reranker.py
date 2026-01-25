"""
Cross-encoder reranker for RAG.
Improves precision after retrieval.
"""

from typing import List
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 6,
    ):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, chunks: List):
        """
        Args:
            query: user question
            chunks: list of RetrievedChunk

        Returns:
            Top-k reranked chunks
        """
        if not chunks:
            return []

        pairs = [(query, c.content) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [c for c, _ in ranked[: self.top_k]]
