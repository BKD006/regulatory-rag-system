from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START
from src.retrieval.hybrid_retriever_v2 import HybridRetriever
from src.memory.conversation_store import ConversationStore
from src.generation.answer_generation import AnswerGenerator
from src.reranking.reranker import CrossEncoderReranker
from utils.models import RAGState
from utils.helper_functions import format_citations
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException
import re
from src.cache.cache_manager import LLMAnswerCacheManager
from langsmith import traceable, get_current_run_tree
from src.guardrails.guardrails import AnswerGuardrails


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline using LangGraph.

    Coordinates query rewriting, retrieval, reranking, answer generation,
    caching, guardrails validation, and conversation persistence.

    Attributes:
        cache (LLMAnswerCacheManager): In-memory cache for responses.
        guardrails (AnswerGuardrails): Post-generation validation layer.
        conversation_store (ConversationStore): Stores and retrieves chat history.
        retriever (HybridRetriever): Hybrid retrieval component (vector + BM25).
        reranker (CrossEncoderReranker): Reranks retrieved chunks.
        answer_generator (AnswerGenerator): Generates final answers using LLM.
        graph (CompiledGraph): LangGraph execution graph.
    """
    
    def __init__(self):
        """
        Initializes all pipeline components and builds the execution graph.
        """
        self.cache = LLMAnswerCacheManager(memory_maxsize=1000,
                                           memory_ttl_seconds=300)
        self.guardrails = AnswerGuardrails()
        self.conversation_store = ConversationStore()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker(top_k=6)
        self.answer_generator = AnswerGenerator()
        self.graph = self._build_graph()

        log.info("rag_pipeline_initialized")

    # -------------------------------------------------
    # LangGraph Nodes
    # ------------------------------------------------

    @traceable(name="Retrieval")
    async def retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Retrieves relevant chunks using hybrid retrieval and applies safety filters.

        Args:
            state (RAGState): Current pipeline state.

        Returns:
            Dict[str, Any]: Retrieved chunks and updated previous chunks.

        Raises:
            Exception: If retrieval fails.
        """
        try:
            query = f"{state.user_query} {state.rewritten_query or ''}".strip()

            chunks = await self.retriever.retrieve(
                query=query,
                filters=state.filters,
            )

            current_doc = (state.filters or {}).get("title")

            if current_doc:
                chunks = [
                    c for c in chunks
                    if c.source == current_doc
                ]

            if len(chunks) < 2 and state.previous_chunks:
                safe_previous_chunks = [
                    c for c in state.previous_chunks
                    if c.source == current_doc
                ]

                if safe_previous_chunks:
                    log.warning("fallback_to_previous_chunks_same_doc")
                    chunks = safe_previous_chunks

            log.info(
                "retrieval_node_completed",
                query=query,
                retrieved_count=len(chunks),
                sample_chunk_ids=[c.chunk_id for c in chunks[:3]],
                sources=list(set([c.source for c in chunks])),
            )

            return {
                "retrieved_chunks": chunks,
                "previous_chunks": chunks,
            }

        except Exception as e:
            log.error("retrieval_node_failed", error=str(e))
            raise
    
    @traceable(name="Rerank")
    async def rerank_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Reranks retrieved chunks based on relevance to the query.

        Args:
            state (RAGState): Current pipeline state.

        Returns:
            Dict[str, Any]: Reranked chunks.
        """
        chunks = state.retrieved_chunks or []

        reranked = self.reranker.rerank(
            query=state.user_query,
            chunks=chunks,
        )

        log.info(
            "rerank_node_completed",
            input_chunks=len(chunks),
            output_chunks=len(reranked),
            top_scores=[round(c.score, 3) for c in reranked[:3]],
        )

        return {"reranked_chunks": reranked}

    async def rewrite_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Rewrites user query using chat history to improve retrieval quality.

        Args:
            state (RAGState): Current pipeline state.

        Returns:
            Dict[str, Any]: Rewritten query.
        """

        history = state.chat_history or []
        query = state.user_query
        current_doc = (state.filters or {}).get("title")

        if not history:
            return {"rewritten_query": query}

        def strip_citations(text: str) -> str:
            return re.sub(r"\[\d+\]", "", text)

        cleaned_history = []
        for h in history:
            content = strip_citations(h["content"])

            if current_doc and len(content) > 1000:
                content = content[:1000]

            cleaned_history.append({
                "role": h["role"],
                "content": content
            })

        history_text = "\n".join(
            f"{h['role']}: {h['content']}" for h in cleaned_history
        )

        previous_answer = ""
        last_assistant_msgs = [
            h["content"] for h in cleaned_history if h["role"] == "assistant"
        ]
        if last_assistant_msgs:
            previous_answer = last_assistant_msgs[-1]

        reference_keywords = ["paragraph", "section", "clause", "article"]
        is_reference_query = any(k in query.lower() for k in reference_keywords)

        vague_patterns = ["explain more", "elaborate", "more details", "explain it"]
        is_vague = any(v in query.lower() for v in vague_patterns)

        prompt = f"""
            You are a query rewriting system for a retrieval engine.

            Your task:
            Convert the follow-up question into a COMPLETE, standalone, retrieval-optimized query.

            STRICT RULES:
            - Resolve references like "this", "that", "it"
            - Include the SUBJECT from previous conversation
            - Expand vague queries into descriptive queries
            - Preserve domain-specific terms
            - DO NOT shorten the query
            - Output ONLY the rewritten query

            Conversation:
            {history_text}

            Previous Answer:
            {previous_answer}

            Follow-up Question:
            {query}

            Rewritten Query:
            """

        try:
            rewritten = await self.answer_generator.llm.ainvoke(prompt)
            rewritten_query = rewritten.content.strip()

            if not rewritten_query:
                rewritten_query = query

            if len(rewritten_query.split()) < 5:
                rewritten_query = f"{query} {previous_answer[:200]}"

            if is_reference_query and previous_answer:
                rewritten_query = f"{rewritten_query} {previous_answer[:300]}"

            if is_vague and previous_answer:
                rewritten_query = f"Detailed explanation of {previous_answer[:300]}"

            log.info(
                "query_rewritten",
                original=query,
                rewritten=rewritten_query,
                is_reference=is_reference_query,
                is_vague=is_vague
            )

            return {"rewritten_query": rewritten_query}

        except Exception as e:
            log.error("rewrite_failed", error=str(e))
            return {"rewritten_query": query}

    @traceable(name="Answer_Generation")
    async def answer_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Generates final answer with validation, retry logic, and guardrails enforcement.

        Args:
            state (RAGState): Current pipeline state.

        Returns:
            Dict[str, Any]: Final answer and validated citations.
        """

        attempts = [
            ("primary", state.reranked_chunks or []),
            ("fallback", state.retrieved_chunks or []),
        ]

        for attempt_name, chunks in attempts:

            if not chunks:
                continue

            cited_chunks = []

            try:
                log.info("answer_attempt_started", attempt=attempt_name, chunk_count=len(chunks))

                result = await self.answer_generator.generate(
                    question=state.user_query,
                    retrieved_chunks=chunks,
                    chat_history=state.chat_history,
                )

                answer_text = result.answer

                citation_matches = re.findall(r"\[(\d+)\]", answer_text)

                cited_indices = {
                    int(num) - 1 for num in citation_matches
                }

                raw_cited_chunks = [
                    chunks[idx]
                    for idx in cited_indices
                    if 0 <= idx < len(chunks)
                ]

                cited_chunks = self.guardrails.filter_citations(
                    raw_cited_chunks,
                    chunks,
                )

                self.guardrails.validate_citations(
                    used_chunks=chunks,
                    cited_chunk_ids=[c.chunk_id for c in cited_chunks],
                )

                self.guardrails.enforce_answer_grounded(
                    answer_text,
                    cited_chunks,
                )

                return {
                    "answer": answer_text,
                    "citations": cited_chunks,
                }

            except RegulatoryRAGException as e:
                log.warning(
                    "validation_failed",
                    attempt=attempt_name,
                    reason=str(e),
                    chunk_count=len(chunks),
                    cited_chunks=len(cited_chunks),
                )
                continue

        log.warning("all_attempts_failed", query=state.user_query)

        return {
            "answer": "The answer is not clearly supported by the document.",
            "citations": [],
        }

    # -------------------------------------------------
    # Graph wiring
    # -------------------------------------------------

    def _build_graph(self):
        """
        Builds the LangGraph workflow connecting all pipeline nodes.

        Returns:
            CompiledGraph: Executable LangGraph pipeline.
        """
        graph = StateGraph(RAGState)

        graph.add_node("rewrite", self.rewrite_node)
        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("answer", self.answer_node)

        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "answer")

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
        """
        Executes the full RAG pipeline for a given query.

        Handles caching, conversation history, pipeline execution,
        and persistence of results.

        Args:
            query (str): User query.
            filters (Optional[Dict[str, Any]]): Retrieval filters (e.g., document title).

        Returns:
            Dict[str, Any]: Final response containing answer and citations.

        Raises:
            RegulatoryRAGException: If pipeline execution fails.
        """

        log.info("rag_run_started", has_filters=bool(filters))

        try:
            current_doc = (filters or {}).get("title", "default")
            session_id = f"doc:{current_doc}"

            cache_key = self.cache.make_key(
                question=query,
                namespace=f"rag:{session_id}",
            )

            cached = await self.cache.get(cache_key)

            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.metadata.update({
                    "has_filters": bool(filters),
                    "cache_hit": cached is not None,
                    "query_length": len(query),
                    "document": current_doc,
                })

            if cached:
                log.info("rag_cache_hit", document=current_doc)
                return cached

            chat_history = await self.conversation_store.get_recent_history(
                session_id=session_id,
                limit=4
            )

            previous_chunks = []

            final_state = await self.graph.ainvoke(
                RAGState(
                    user_query=query,
                    filters=filters or {},
                    chat_history=chat_history,
                    previous_chunks=previous_chunks,
                )
            )

            raw_citations = format_citations(
                final_state.get("citations", [])
            )

            safe_citations = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in raw_citations
            ]

            response = {
                "answer": final_state.get("answer"),
                "citations": safe_citations,
            }

            if response["answer"] and not response["answer"].startswith(
                "The question cannot be answered safely"
            ):
                try:
                    await self.conversation_store.save_qa(
                        query=query,
                        answer=response["answer"],
                        citations=response["citations"],
                        metadata={
                            "filters": filters,
                            "session_id": session_id,
                            "document": current_doc,
                        }
                    )
                except Exception as e:
                    log.error("conversation_store_save_failed", error=str(e))

                await self.cache.set(cache_key, response)

            log.info("rag_run_completed", document=current_doc)

            return response

        except Exception as e:
            log.error("rag_pipeline_failed", error=str(e))
            raise RegulatoryRAGException(e)