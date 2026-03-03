"""
End-to-end RAG LangGraph:
Retrieval → Reranking → Guardrails → Answer Generation
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START
from src.retrieval.hybrid_retriever_v2 import HybridRetriever
from src.generation.answer_generation import AnswerGenerator
from src.reranking.reranker import CrossEncoderReranker
from src.guardrails.guardrails import AnswerGuardrails, GuardrailViolation
from utils.models import RAGState, RetrievedChunk
from utils.helper_functions import format_citations
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException
import re
from src.cache.cache_manager import LLMAnswerCacheManager
from langsmith import traceable, get_current_run_tree

class RAGPipeline:
    """
    End-to-end RAG pipeline with HARD guardrails.
    """

    def __init__(self):
        self.cache = LLMAnswerCacheManager(memory_maxsize=1000,
                                           memory_ttl_seconds=300,     # 5 min L1
                                           postgres_ttl_minutes=60     # 1 hour L2
                                           )
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_k=6)
        self.answer_generator = AnswerGenerator()
        self.guardrails = AnswerGuardrails()
        self.graph = self._build_graph()

        log.info("rag_pipeline_initialized")

    # -------------------------------------------------
    # LangGraph Nodes
    # -------------------------------------------------
    @traceable(name="Retrieval")
    async def retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        try:
            chunks = await self.retriever.retrieve(
                query=state.user_query,
                filters=state.filters,
            )

            log.info(
                "retrieval_node_completed",
                chunk_count=len(chunks),
            )

            return {"retrieved_chunks": chunks}

        except Exception as e:
            log.error("retrieval_node_failed", error=str(e))
            raise
    
    @traceable(name="Rerank")
    async def rerank_node(self, state: RAGState) -> Dict[str, Any]:
        chunks = state.retrieved_chunks or []

        reranked = self.reranker.rerank(
            query=state.user_query,
            chunks=chunks,
        )

        log.info(
            "rerank_node_completed",
            input_chunks=len(chunks),
            output_chunks=len(reranked),
        )

        return {"reranked_chunks": reranked}

    async def guardrail_node(self, state: RAGState) -> Dict[str, Any]:
        """
        HARD enforcement BEFORE LLM:
        - Single document only
        - Minimum evidence
        """

        chunks: List[RetrievedChunk] = (
            state.reranked_chunks or state.retrieved_chunks or []
        )

        try:
            guarded_chunks = self.guardrails.apply_retrieval_guardrails(
                chunks=chunks,
                filters=state.filters,
                min_chunks=2,
            )

            log.info(
                "guardrails_passed",
                chunk_count=len(guarded_chunks),
            )

            return {"guarded_chunks": guarded_chunks}

        except GuardrailViolation as e:
            # EXPECTED, SAFE REFUSAL (not an error)
            log.warning(
                "guardrail_blocked",
                reason=str(e),
                chunk_count=len(chunks),
            )

            return {
                "answer": str(e),
                "citations": [],
            }

    @traceable(name="Answer_Generation")
    async def answer_node(self, state: RAGState) -> Dict[str, Any]:
        chunks = state.guarded_chunks or []

        if not chunks:
            log.warning("answer_node_no_chunks")

            return {
                "answer": (
                    "The question cannot be answered safely "
                    "from the selected document."
                ),
                "citations": [],
            }

        try:
            # ---------------------------------------------
            # Generate Answer
            # ---------------------------------------------
            result = await self.answer_generator.generate(
                question=state.user_query,
                retrieved_chunks=chunks,
            )

            answer_text = result.answer

            # ---------------------------------------------
            # Extract citation numbers from answer text
            # Example: [1], [5], [12]
            # ---------------------------------------------
            citation_matches = re.findall(r"\[(\d+)\]", answer_text)

            # Convert to zero-based indices
            cited_indices = {
                int(num) - 1
                for num in citation_matches
            }

            # ---------------------------------------------
            # Validate citation indices
            # ---------------------------------------------
            invalid_indices = [
                idx for idx in cited_indices
                if idx < 0 or idx >= len(chunks)
            ]

            if invalid_indices:
                log.warning(
                    "invalid_citation_indices_detected",
                    invalid_indices=invalid_indices,
                )
                raise GuardrailViolation(
                    f"Invalid citation numbers detected: {invalid_indices}"
                )

            # ---------------------------------------------
            # Map indices to actual chunks
            # ---------------------------------------------
            cited_chunks = [
                chunks[idx]
                for idx in sorted(cited_indices)
            ]

            # ---------------------------------------------
            # Deterministic citation validation
            # ---------------------------------------------
            self.guardrails.validate_citations(
                used_chunks=chunks,
                cited_chunk_ids=[c.chunk_id for c in cited_chunks],
            )

            # ---------------------------------------------
            # Grounding enforcement
            # ---------------------------------------------
            self.guardrails.enforce_answer_grounded(
                answer_text,
                cited_chunks,
            )

            log.info(
                "answer_generated",
                chunks_provided=len(chunks),
                citations_used=len(cited_chunks),
            )

            return {
                "answer": answer_text,
                "citations": cited_chunks,
            }

        except GuardrailViolation as e:
            log.warning(
                "answer_guardrail_blocked",
                reason=str(e),
            )

            return {
                "answer": str(e),
                "citations": [],
            }

        except Exception as e:
            log.error("answer_generation_failed", error=str(e))
            raise

    # -------------------------------------------------
    # Graph wiring
    # -------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("guardrails", self.guardrail_node)
        graph.add_node("answer", self.answer_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "guardrails")
        graph.add_edge("guardrails", "answer")

        return graph.compile()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    @traceable(name="RAG_Run")
    async def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        log.info(
            "rag_run_started",
            has_filters=bool(filters),
        )

        try:
            # -------------------------------------------------
            # 1. Generate deterministic cache key
            # -------------------------------------------------
            cache_key = await self.cache.make_key(
                question=query,
                namespace=f"rag:{filters}"
            )

            # -------------------------------------------------
            # 2. Check Hybrid Cache (L1 → L2)
            # -------------------------------------------------
            cached = await self.cache.get(cache_key)
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.metadata.update({
                    "has_filters": bool(filters),
                    "cache_hit": cached is not None,
                    "query_length": len(query),
                })
            if cached:
                log.info("rag_cache_hit")
                return cached

            log.info("rag_cache_miss")

            # -------------------------------------------------
            # 3. Run Graph (cold path)
            # -------------------------------------------------
            final_state = await self.graph.ainvoke(
                RAGState(
                    user_query=query,
                    filters=filters or {},
                )
            )

            raw_citations = format_citations(
                final_state.get("citations", [])
            )

            # Ensure JSON-safe
            safe_citations = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in raw_citations
            ]

            response = {
                "answer": final_state.get("answer"),
                "citations": safe_citations,
            }

            # -------------------------------------------------
            # 4. Store Only Valid Answers
            # -------------------------------------------------
            if response["answer"] and not response["answer"].startswith("The question cannot be answered safely"):
                await self.cache.set(
                    cache_key,
                    query,
                    response,
                )

            log.info("rag_run_completed")

            return response

        except Exception as e:
            log.error("rag_pipeline_failed", error=str(e))
            raise RegulatoryRAGException(e)