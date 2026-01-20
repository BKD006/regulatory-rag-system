"""
End-to-end RAG LangGraph:
Retrieval → Answer Generation with Citations
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from src.retrieval.retrieval import HybridRetriever
from src.generation.answer_generation import AnswerGenerator
from utils.models import RAGState


class RAGPipeline:
    """
    End-to-end RAG pipeline using LangGraph + checkpointed memory.
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.retriever = HybridRetriever()
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

    async def answer_node(self, state: RAGState) -> Dict[str, Any]:
        result = await self.answer_generator.generate(
            question=state.user_query,
            retrieved_chunks=state.retrieved_chunks,
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
        graph.add_node("answer", self.answer_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "answer")

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

        return {
            "answer": final_state.get("answer"),
            "citations": final_state.get("citations", []),
            "retrieved_chunks": final_state.get("retrieved_chunks", [])
        }
