"""
End-to-end RAG LangGraph:
Retrieval → Reranking → Guardrails → Answer Generation
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from src.retrieval.retrieval import HybridRetriever
from src.generation.answer_generation import AnswerGenerator
from src.reranking.reranker import CrossEncoderReranker
from src.guardrails.guardrails import AnswerGuardrails, GuardrailViolation
from utils.models import RAGState, RetrievedChunk
from utils.helper_functions import format_citations
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class RAGPipeline:
    """
    End-to-end RAG pipeline with HARD guardrails.
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_k=6)
        self.answer_generator = AnswerGenerator()
        self.guardrails = AnswerGuardrails()
        self.graph = self._build_graph()

        log.info("rag_pipeline_initialized")

    # -------------------------------------------------
    # LangGraph Nodes
    # -------------------------------------------------

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
                min_chunks=1,
            )

            log.info(
                "guardrails_passed",
                chunk_count=len(guarded_chunks),
            )

            return {"guarded_chunks": guarded_chunks}

        except GuardrailViolation as e:
            # ⚠️ EXPECTED, SAFE REFUSAL (not an error)
            log.warning(
                "guardrail_blocked",
                reason=str(e),
                chunk_count=len(chunks),
            )

            return {
                "answer": str(e),
                "citations": [],
            }

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
            result = await self.answer_generator.generate(
                question=state.user_query,
                retrieved_chunks=chunks,
            )

            # 🔒 HARD citation filtering
            citations = self.guardrails.filter_citations(
                result.citations,
                allowed_chunks=chunks,
            )

            log.info(
                "answer_generated",
                chunks_used=len(chunks),
                citations=len(citations),
            )

            return {
                "answer": result.answer,
                "citations": citations,
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

        return graph.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        thread_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        log.info(
            "rag_run_started",
            has_filters=bool(filters),
            thread_id=thread_id,
        )

        try:
            final_state = await self.graph.ainvoke(
                RAGState(
                    user_query=query,
                    filters=filters or {},
                ),
                config={
                    "configurable": {
                        "thread_id": thread_id
                    }
                },
            )

            log.info("rag_run_completed")

            return {
                "answer": final_state.get("answer"),
                "citations": format_citations(
                    final_state.get("citations", [])
                ),
            }

        except Exception as e:
            # 🚨 True system failure (not a guardrail refusal)
            log.error("rag_pipeline_failed", error=str(e))
            raise RegulatoryRAGException(e)
