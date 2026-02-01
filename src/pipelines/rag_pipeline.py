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


class RAGPipeline:
    """
    End-to-end RAG pipeline with HARD guardrails.
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_k=6)
        self.answer_generator = AnswerGenerator()
        self.graph = self._build_graph()
        self.guardrails = AnswerGuardrails()

    # -------------------------------------------------
    # LangGraph Nodes
    # -------------------------------------------------

    async def retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        chunks = await self.retriever.retrieve(
            query=state.user_query,
            filters=state.filters,
        )
        return {"retrieved_chunks": chunks}

    async def rerank_node(self, state: RAGState) -> Dict[str, Any]:
        reranked = self.reranker.rerank(
            query=state.user_query,
            chunks=state.retrieved_chunks,
        )
        return {"reranked_chunks": reranked}

    async def guardrail_node(self, state: RAGState) -> Dict[str, Any]:
        """
        HARD enforcement BEFORE LLM:
        - Single document only
        - Minimum evidence
        """

        chunks: List[RetrievedChunk] = (
            state.reranked_chunks or state.retrieved_chunks
        )

        try:
            guarded_chunks = self.guardrails.apply_retrieval_guardrails(
                chunks=chunks,
                filters=state.filters,
                min_chunks=1,
            )
        except GuardrailViolation as e:
            # HARD STOP — safe refusal
            return {
                "answer": str(e),
                "citations": [],
            }

        return {"guarded_chunks": guarded_chunks}

    async def answer_node(self, state: RAGState) -> Dict[str, Any]:
        chunks = state.guarded_chunks or []

        if not chunks:
            return {
                "answer": "The question cannot be answered safely from the selected document.",
                "citations": [],
            }

        result = await self.answer_generator.generate(
            question=state.user_query,
            retrieved_chunks=chunks,
        )

        # 🔒 HARD citation filtering
        citations = self.guardrails.filter_citations(
            result.citations,
            allowed_chunks=chunks,
        )

        return {
            "answer": result.answer,
            "citations": citations,
        }


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

        return {
            "answer": final_state.get("answer"),
            "citations": format_citations(
                final_state.get("citations", [])
            ),
        }
