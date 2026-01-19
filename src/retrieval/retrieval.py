"""
Hybrid Retrieval Pipeline (BM25 + Vector) using LangGraph.

This module encapsulates:
- BM25 retrieval (Postgres full-text search)
- Vector retrieval (pgvector cosine similarity)
- Reciprocal Rank Fusion (RRF)
- LangGraph wiring

Designed for regulatory / compliance RAG systems.
"""

import os
import json
import asyncpg
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

from src.embeddings.embedder import create_embedder
from utils.models import RetrievalState, RetrievedChunk
load_dotenv()

# ------------------------------------------------------------------
# Hybrid Retriever (single class)
# ------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retrieval using BM25 + Vector search, implemented
    as a LangGraph pipeline inside a single class.
    """

    def __init__(
        self,
        top_k: int = 10,
        bm25_k: int = 20,
        vector_k: int = 20,
    ):
        self.top_k = top_k
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.database_url = os.getenv("DATABASE_URL")

        if not self.database_url:
            raise RuntimeError("DATABASE_URL must be set")

        self.embedder = create_embedder()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # DB helper
    # ------------------------------------------------------------------

    async def _get_connection(self):
        return await asyncpg.connect(self.database_url)

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    async def _bm25_node(self, state: RetrievalState) -> RetrievalState:
        """
        BM25-style retrieval using PostgreSQL full-text search.
        """
        conn = await self._get_connection()

        query = """
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
        ORDER BY score DESC
        LIMIT $2;
        """

        rows = await conn.fetch(query, state.user_query, self.bm25_k)
        await conn.close()

        state.bm25_results = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=float(r["score"]),
                source=r["source"],
                metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
            )
            for r in rows
        ]

        return state

    async def _vector_node(self, state: RetrievalState) -> RetrievalState:
        """
        Vector retrieval using pgvector cosine similarity.
        """
        query_embedding = await self.embedder.embed_query(state.user_query)

        # Convert Python list → pgvector literal
        vector_literal = "[" + ",".join(map(str, query_embedding)) + "]"

        conn = await self._get_connection()

        query = """
        SELECT
            c.id::text AS chunk_id,
            c.document_id::text,
            c.content,
            1 - (c.embedding <=> $1::vector) AS score,
            d.title AS source,
            c.metadata
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2;
        """

        rows = await conn.fetch(query, vector_literal, self.vector_k)
        await conn.close()

        state.vector_results = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=float(r["score"]),
                source=r["source"],
                metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
            )
            for r in rows
        ]

        return state


    def _fusion_node(self, state: RetrievalState) -> RetrievalState:
        """
        Reciprocal Rank Fusion (RRF).
        """
        scores = defaultdict(float)
        chunk_map = {}

        def add_results(results, weight=1.0):
            for rank, r in enumerate(results, start=1):
                scores[r.chunk_id] += weight / (rank + 60)
                chunk_map[r.chunk_id] = r

        add_results(state.bm25_results, weight=1.0)
        add_results(state.vector_results, weight=1.0)

        fused = sorted(
            chunk_map.values(),
            key=lambda r: scores[r.chunk_id],
            reverse=True
        )

        state.fused_results = fused[: self.top_k]
        return state

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(RetrievalState)

        graph.add_node("bm25", self._bm25_node)
        graph.add_node("vector", self._vector_node)
        graph.add_node("fusion", self._fusion_node)

        graph.set_entry_point("bm25")
        graph.add_edge("bm25", "vector")
        graph.add_edge("vector", "fusion")

        return graph.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Run hybrid retrieval and return top-k fused chunks.
        """
        state = RetrievalState(user_query=query)
        final_state = await self.graph.ainvoke(state)
        return final_state["fused_results"]
