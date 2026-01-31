"""
End-to-end RAG LangGraph:
Retrieval → Reranking → Answer Generation with Citations
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from src.retrieval.retrieval import HybridRetriever
from src.generation.answer_generation import AnswerGenerator
from src.reranking.reranker import CrossEncoderReranker
from utils.models import RAGState, RetrievedChunk
from utils.helper_functions import format_citations


class RAGPipeline:
    """
    End-to-end RAG pipeline using LangGraph + checkpointed memory.
    Designed for regulatory / compliance-grade QA.
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_k=6)
        self.answer_generator = AnswerGenerator()
        self.graph = self._build_graph()

    # -------------------------------------------------
    # LangGraph Nodes
    # -------------------------------------------------

    async def retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Strict retrieval (filters enforced at DB level).
        """
        chunks: List[RetrievedChunk] = await self.retriever.retrieve(
            query=state.user_query,
            filters=state.filters,
        )
        return {"retrieved_chunks": chunks}

    async def rerank_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Rerank retrieved chunks.
        Falls back safely if reranker returns nothing.
        """
        if not state.retrieved_chunks:
            return {"reranked_chunks": []}

        reranked = self.reranker.rerank(
            query=state.user_query,
            chunks=state.retrieved_chunks,
        )

        return {
            "reranked_chunks": reranked or state.retrieved_chunks
        }

    async def answer_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Generate answer ONLY if evidence exists.
        """
        chunks = state.reranked_chunks or state.retrieved_chunks

        if not chunks:
            return {
                "answer": (
                    "No relevant information was found in the selected "
                    "document(s) to answer this question."
                ),
                "citations": [],
            }

        result = await self.answer_generator.generate(
            question=state.user_query,
            retrieved_chunks=chunks,
        )

        return {
            "answer": result.answer,
            "citations": result.citations,
        }

    # -------------------------------------------------
    # Graph Wiring
    # -------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("answer", self.answer_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "answer")

        return graph.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def run(self,query: str,*,thread_id: str,filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        filters = filters or {}

        final_state = await self.graph.ainvoke(
            RAGState(
                user_query=query,
                filters=filters,
            ),
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            },
        )

        answer = final_state.get("answer")
        raw_citations = final_state.get("citations", [])

        # -------------------------------------------------
        # 🔒 HARD CITATION FILTER (FAIL-CLOSED)
        # -------------------------------------------------

        if "title" in filters:
            allowed_title = filters["title"]

            filtered_citations = [
                c for c in raw_citations
                if c.source == allowed_title
            ]

            # If LLM tried to use other documents → drop them
            if not filtered_citations:
                return {
                    "answer": (
                        "The selected document does not contain sufficient "
                        "information to answer this question."
                    ),
                    "citations": [],
                }

            raw_citations = filtered_citations

        return {
            "answer": answer,
            "citations": format_citations(raw_citations),
        }
