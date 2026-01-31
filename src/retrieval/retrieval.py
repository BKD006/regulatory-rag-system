"""
Hybrid Retrieval using LangGraph
- BM25 + Vector (pgvector)
- Parallel execution
- Timeouts + fallback
- Weighted fusion
- STRICT metadata filters (NO leakage)
"""

import os
import json
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from collections import defaultdict

from langgraph.graph import StateGraph, START

from src.embeddings.embedder import create_embedder
from utils.models import RetrievedChunk, RetrievalState


class HybridRetriever:
    """
    Production-grade hybrid retriever with strict filtering.
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        bm25_k: int = 20,
        vector_k: int = 20,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        bm25_timeout: float = 30,
        vector_timeout: float = 30,
    ):
        self.top_k = top_k
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_timeout = bm25_timeout
        self.vector_timeout = vector_timeout

        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL must be set")

        self.embedder = create_embedder()
        self.graph = self._build_graph()

    # -------------------------------------------------
    # DB helpers
    # -------------------------------------------------

    async def _get_conn(self):
        return await asyncpg.connect(self.database_url)

    def _parse_json(self, val):
        return json.loads(val) if isinstance(val, str) else val

    # -------------------------------------------------
    # BM25 Node (FILTERED)
    # -------------------------------------------------

    async def _bm25_node(self, state: RetrievalState):
        async def _run():
            conn = await self._get_conn()

            filters = state.filters or {}
            title_filter = filters.get("title")

            rows = await conn.fetch(
                """
                SELECT
                    c.id::text AS chunk_id,
                    c.document_id::text,
                    c.content,
                    ts_rank_cd(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', $1)
                    ) AS score,
                    d.title AS source,
                    c.metadata
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE to_tsvector('english', c.content)
                      @@ plainto_tsquery('english', $1)
                  AND ($2::text IS NULL OR d.title = $2)
                ORDER BY score DESC
                LIMIT $3;
                """,
                state.user_query,
                title_filter,
                self.bm25_k,
            )

            await conn.close()

            return {
                "bm25_results": [
                    RetrievedChunk(
                        chunk_id=r["chunk_id"],
                        document_id=r["document_id"],
                        content=r["content"],
                        score=float(r["score"]),
                        source=r["source"],
                        metadata=self._parse_json(r["metadata"]),
                    )
                    for r in rows
                ]
            }

        try:
            return await asyncio.wait_for(_run(), timeout=self.bm25_timeout)
        except Exception:
            return {"bm25_results": []}

    # -------------------------------------------------
    # Vector Node (FILTERED)
    # -------------------------------------------------

    async def _vector_node(self, state: RetrievalState):
        async def _run():
            embedding = await self.embedder.embed_query(state.user_query)
            vector_literal = "[" + ",".join(map(str, embedding)) + "]"

            filters = state.filters or {}
            title_filter = filters.get("title")

            conn = await self._get_conn()

            rows = await conn.fetch(
                """
                SELECT
                    c.id::text AS chunk_id,
                    c.document_id::text,
                    c.content,
                    1 - (c.embedding <=> $1::vector) AS score,
                    d.title AS source,
                    c.metadata
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL
                  AND ($2::text IS NULL OR d.title = $2)
                ORDER BY c.embedding <=> $1::vector
                LIMIT $3;
                """,
                vector_literal,
                title_filter,
                self.vector_k,
            )

            await conn.close()

            return {
                "vector_results": [
                    RetrievedChunk(
                        chunk_id=r["chunk_id"],
                        document_id=r["document_id"],
                        content=r["content"],
                        score=float(r["score"]),
                        source=r["source"],
                        metadata=self._parse_json(r["metadata"]),
                    )
                    for r in rows
                ]
            }

        try:
            return await asyncio.wait_for(_run(), timeout=self.vector_timeout)
        except Exception:
            return {"vector_results": []}

    # -------------------------------------------------
    # Fusion Node (STRICT)
    # -------------------------------------------------

    def _fusion_node(self, state: RetrievalState):
        scores = defaultdict(float)
        chunks: Dict[str, RetrievedChunk] = {}

        def add(results, weight):
            for rank, r in enumerate(results, start=1):
                scores[r.chunk_id] += weight / (rank + 60)
                chunks[r.chunk_id] = r

        add(state.bm25_results, self.bm25_weight)
        add(state.vector_results, self.vector_weight)

        fused = sorted(
            chunks.values(),
            key=lambda r: scores[r.chunk_id],
            reverse=True,
        )[: self.top_k]

        return {"fused_results": fused}

    # -------------------------------------------------
    # Graph Wiring
    # -------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(RetrievalState)

        graph.add_node("bm25", self._bm25_node)
        graph.add_node("vector", self._vector_node)
        graph.add_node("fusion", self._fusion_node)

        graph.add_edge(START, "bm25")
        graph.add_edge(START, "vector")
        graph.add_edge("bm25", "fusion")
        graph.add_edge("vector", "fusion")

        return graph.compile()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:

        state = RetrievalState(
            user_query=query,
            filters=filters or {},
        )

        final_state = await self.graph.ainvoke(state)
        return final_state["fused_results"]
