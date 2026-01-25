"""
End-to-end RAG LangGraph:
Retrieval → Answer Generation with Citations
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from src.retrieval.retrieval import HybridRetriever
from src.generation.answer_generation import AnswerGenerator
from src.reranking.reranker import CrossEncoderReranker
from utils.models import RAGState
from utils.helper_functions import format_citations


class RAGPipeline:
    """
    End-to-end RAG pipeline using LangGraph + checkpointed memory.
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_k=6)
        self.answer_generator = AnswerGenerator()
        self.graph = self._build_graph()

    # ----------------------------
    # LangGraph Nodes
    # ----------------------------

    async def retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        chunks = await self.retriever.retrieve(
            query=state.user_query,
            filters=state.filters,
        )
        return {"retrieved_chunks": chunks}
    
    async def rerank_node(self, state: RAGState):
        reranked = self.reranker.rerank(
            query=state.user_query,
            chunks=state.retrieved_chunks,
        )
        return {"reranked_chunks": reranked}


    async def answer_node(self, state: RAGState) -> Dict[str, Any]:
        chunks = state.reranked_chunks or state.retrieved_chunks
        result = await self.answer_generator.generate(
            question=state.user_query,
            retrieved_chunks=chunks,
        )
        return {
            "answer": result.answer,
            "citations": result.citations,
        }

    # ----------------------------
    # Graph wiring
    # ----------------------------

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("answer", self.answer_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "answer")

        return graph.compile(checkpointer=self.checkpointer)

    # ----------------------------
    # Public API
    # ----------------------------

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
        raw_chunks = final_state.get("citations", [])
        return {
            "answer": final_state.get("answer"),
            "citations": format_citations(raw_chunks),
            # "retrieved_chunks": final_state.get("retrieved_chunks", [])
        }
