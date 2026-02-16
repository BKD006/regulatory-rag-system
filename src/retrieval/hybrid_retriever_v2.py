import os
import json
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, START
from src.embeddings.embedder import create_embedder
from utils.models import RetrievedChunk, RetrievalState
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class HybridRetriever:
    """
    Production-grade hybrid retriever (V2)
    Score-calibrated + strict filtering + adaptive weighting.
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
        """
        Initialize the HybridRetriever V2.

        This retriever performs hybrid retrieval using:
        - BM25 (PostgreSQL full-text search)
        - Dense vector similarity (pgvector)
        - Score-level weighted fusion with dynamic weighting
        - Strict metadata filtering enforcement

        Parameters:
            top_k (int): Final number of results returned after fusion.
            bm25_k (int): Number of candidates retrieved from BM25.
            vector_k (int): Number of candidates retrieved from vector search.
            bm25_weight (float): Default weight assigned to BM25 scores.
            vector_weight (float): Default weight assigned to vector scores.
            bm25_timeout (float): Timeout (seconds) for BM25 retrieval.
            vector_timeout (float): Timeout (seconds) for vector retrieval.

        Raises:
            RuntimeError: If DATABASE_URL environment variable is not set.
        """
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

        # Optional reranker hook
        self.reranker = None
        self.rerank_k = 20

        self.graph = self._build_graph()

        log.info(
            "hybrid_retriever_v2_initialized",
            top_k=top_k,
            bm25_k=bm25_k,
            vector_k=vector_k,
        )

    # -------------------------------------------------
    # DB Helpers
    # -------------------------------------------------

    async def _get_conn(self):
        """
        Create and return a new async PostgreSQL connection.

        Returns:
            asyncpg.Connection: Active database connection.

        Raises:
            Exception: If connection establishment fails.
        """
        try:
            return await asyncpg.connect(self.database_url)
        except Exception as e:
            log.error("db_connection_failed", error=str(e))
            raise

    def _parse_json(self, val):
        """
        Safely parse JSON metadata returned from the database.

        Parameters:
            val: JSON string or already-parsed object.

        Returns:
            Parsed Python object if string, otherwise original value.
        """
        return json.loads(val) if isinstance(val, str) else val

    # -------------------------------------------------
    # Score Utilities
    # -------------------------------------------------

    def _min_max_normalize(self, results):
        """
        Apply per-query min-max normalization to retrieval scores.

        Normalizes raw retrieval scores into the range [0, 1]
        to enable calibrated score-level fusion between BM25
        and dense vector results.

        Parameters:
            results (List[RetrievedChunk]): Retrieved chunks with raw scores.

        Returns:
            Dict[str, float]: Mapping of chunk_id to normalized score.
        """
        if not results:
            return {}

        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return {r.chunk_id: 1.0 for r in results}

        return {
            r.chunk_id: (r.score - min_s) / (max_s - min_s)
            for r in results
        }

    def _get_dynamic_weights(self, query: str):
        """
        Determine adaptive fusion weights based on query characteristics.
        Heuristic logic:
            - Numeric-heavy queries → favor BM25
            - Short queries → moderately favor BM25
            - Default → use configured weights

        Parameters:
            query (str): User query string.

        Returns:
            Tuple[float, float]: (bm25_weight, vector_weight)
        """
        if any(char.isdigit() for char in query):
            return 0.7, 0.3  # BM25 heavy
        if len(query.split()) <= 3:
            return 0.6, 0.4
        return self.bm25_weight, self.vector_weight

    def _validate_filters(self, results, filters):
        """
        Enforce strict metadata filter validation to prevent leakage.

        Ensures that all retrieved chunks comply with the
        requested metadata filters (e.g., document title).

        This acts as a secondary safety layer in addition
        to database-level filtering.

        Parameters:
            results (List[RetrievedChunk]): Final retrieved results.
            filters (Dict[str, Any]): Applied metadata filters.

        Raises:
            RegulatoryRAGException: If any result violates filters.
    """
        if not filters:
            return

        title_filter = filters.get("title")

        for r in results:
            if title_filter and r.source != title_filter:
                log.error(
                    "metadata_leak_detected",
                    chunk_id=r.chunk_id,
                    expected=title_filter,
                    actual=r.source,
                )
                raise RegulatoryRAGException(
                    "Strict metadata filter violation detected."
                )

    # -------------------------------------------------
    # BM25 Node
    # -------------------------------------------------

    async def _bm25_node(self, state: RetrievalState):
        """
        Execute BM25 full-text search retrieval.

        Performs PostgreSQL text search using `ts_rank_cd`
        with strict metadata filtering.

        Execution characteristics:
            - Async execution
            - Timeout protected
            - Graceful fallback on failure
            - Structured logging

        Parameters:
            state (RetrievalState): Current retrieval state.

        Returns:
            Dict containing:
                {
                    "bm25_results": List[RetrievedChunk]
                }
        """
        async def _run():
            conn = await self._get_conn()
            try:
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
            finally:
                await conn.close()

        try:
            result = await asyncio.wait_for(
                _run(), timeout=self.bm25_timeout
            )
            log.info("bm25_completed", count=len(result["bm25_results"]))
            return result

        except asyncio.TimeoutError:
            log.warning("bm25_timeout")
            return {"bm25_results": []}

        except Exception as e:
            log.error("bm25_failed", error=str(e))
            return {"bm25_results": []}

    # -------------------------------------------------
    # Vector Node
    # -------------------------------------------------

    async def _vector_node(self, state: RetrievalState):
        """
        Execute dense vector similarity retrieval using pgvector.

        Steps:
            1. Embed query using configured embedder
            2. Perform similarity search using cosine distance
            3. Apply strict metadata filtering
            4. Return top-k candidates

        Execution characteristics:
            - Async execution
            - Timeout protected
            - Graceful fallback on failure
            - Structured logging

        Parameters:
            state (RetrievalState): Current retrieval state.

        Returns:
            Dict containing:
                {
                    "vector_results": List[RetrievedChunk]
                }
        """
        async def _run():
            embedding = await self.embedder.embed_query(state.user_query)
            vector_literal = "[" + ",".join(map(str, embedding)) + "]"

            filters = state.filters or {}
            title_filter = filters.get("title")

            conn = await self._get_conn()
            try:
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
            finally:
                await conn.close()

        try:
            result = await asyncio.wait_for(
                _run(), timeout=self.vector_timeout
            )
            log.info("vector_completed", count=len(result["vector_results"]))
            return result

        except asyncio.TimeoutError:
            log.warning("vector_timeout")
            return {"vector_results": []}

        except Exception as e:
            log.error("vector_failed", error=str(e))
            return {"vector_results": []}

    # -------------------------------------------------
    # Fusion Node
    # -------------------------------------------------

    def _fusion_node(self, state: RetrievalState):
        """
        Perform score-level hybrid fusion of BM25 and vector results.

        Fusion process:
            1. Min-max normalize scores independently
            2. Apply dynamic query-adaptive weights
            3. Compute weighted linear combination
            4. Sort and select top_k results

        This replaces rank-based fusion (RRF) with
        calibrated score-level hybrid retrieval.

        Parameters:
            state (RetrievalState): Retrieval state containing
                                    bm25_results and vector_results.

        Returns:
            Dict containing:
                {
                    "fused_results": List[RetrievedChunk]
                }
        """
        bm25_results = state.bm25_results or []
        vector_results = state.vector_results or []

        if not bm25_results and not vector_results:
            log.warning("both_retrievers_empty")
            return {"fused_results": []}

        bm25_norm = self._min_max_normalize(bm25_results)
        vector_norm = self._min_max_normalize(vector_results)

        bm25_w, vector_w = self._get_dynamic_weights(state.user_query)

        all_chunks: Dict[str, RetrievedChunk] = {}
        final_scores = {}

        for r in bm25_results + vector_results:
            all_chunks[r.chunk_id] = r

        for chunk_id, chunk in all_chunks.items():
            score = (
                bm25_w * bm25_norm.get(chunk_id, 0.0)
                + vector_w * vector_norm.get(chunk_id, 0.0)
            )
            final_scores[chunk_id] = score

        fused = sorted(
            all_chunks.values(),
            key=lambda r: final_scores[r.chunk_id],
            reverse=True,
        )[: self.top_k]

        log.info(
            "fusion_completed_v2",
            bm25=len(bm25_results),
            vector=len(vector_results),
            fused=len(fused),
            bm25_weight=bm25_w,
            vector_weight=vector_w,
        )

        return {"fused_results": fused}

    # -------------------------------------------------
    # Graph Wiring
    # -------------------------------------------------

    def _build_graph(self):
        """
        Construct and compile the LangGraph retrieval pipeline.

        Graph Structure:
            START
            ├── BM25 Node
            ├── Vector Node
            ↓
            Fusion Node

        Both retrieval nodes execute in parallel.
        Fusion executes after both complete.

        Returns:
            Compiled LangGraph instance.
        """
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
        """
        Execute the full hybrid retrieval pipeline.

        Pipeline:
            1. Parallel BM25 + Vector retrieval
            2. Score-level fusion
            3. Strict metadata validation
            4. Optional reranking
            5. Return top_k results

        Parameters:
            query (str): User query string.
            filters (Optional[Dict[str, Any]]): Metadata filters
                                            (e.g., title).

        Returns:
            List[RetrievedChunk]: Final ranked retrieval results.

        Raises:
            RegulatoryRAGException: If pipeline fails or filter
                                    violation is detected.
        """

        log.info("retrieval_started", has_filters=bool(filters))

        try:
            state = RetrievalState(
                user_query=query,
                filters=filters or {},
            )

            final_state = await self.graph.ainvoke(state)

            results = final_state.get("fused_results", [])

            # Strict leakage validation
            self._validate_filters(results, filters)

            # Optional reranking hook
            if self.reranker and results:
                results = await self.reranker.rerank(query, results)

            log.info("retrieval_completed", result_count=len(results))

            return results

        except Exception as e:
            log.error("retrieval_pipeline_failed", error=str(e))
            raise RegulatoryRAGException(e)
