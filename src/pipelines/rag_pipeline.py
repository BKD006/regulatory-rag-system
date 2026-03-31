from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START
from src.retrieval.hybrid_retriever_v2 import HybridRetriever
from src.memory.conversation_store import ConversationStore
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
    
    def __init__(self):

        self.cache = LLMAnswerCacheManager(memory_maxsize=1000,
                                           memory_ttl_seconds=300,     # 5 min L1
                                           )
        self.conversation_store = ConversationStore()
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
            query = state.rewritten_query or state.user_query
            chunks = await self.retriever.retrieve(
                query=query,
                filters=state.filters,
            )
            # Conversational fallback (handles weak + empty retrieval)
            if len(chunks) < 2 and state.previous_chunks:
                if state.chat_history:  # ensure it's a follow-up
                    log.warning("weak_retrieval_fallback_to_previous_chunks")
                    chunks = state.previous_chunks

            log.info(
                "retrieval_node_completed",
                chunk_count=len(chunks),
            )

            return {"retrieved_chunks": chunks,
                    "previous_chunks": chunks}

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

    async def rewrite_node(self, state: RAGState) -> Dict[str, Any]:
        history = state.chat_history
        query = state.user_query

        if not history:
            return {"rewritten_query": query}

        history_text = "\n".join(
            f"{h['role']}: {h['content']}" for h in history
        )

        prompt = f"""
                    You are a query rewriting system for a retrieval engine.

                    Convert the follow-up question into a COMPLETE standalone query.

                    STRICT RULES:
                    - Resolve all references like "it", "this", "that"
                    - Include domain-specific terms from conversation
                    - Expand vague queries into detailed searchable queries
                    - DO NOT return short queries

                    Conversation:
                    {history_text}

                    Follow-up:
                    {query}

                    Standalone Query:
                """
        rewritten = await self.answer_generator.llm.ainvoke(prompt)
        return {"rewritten_query": rewritten.content.strip()}

    async def guardrail_node(self, state: RAGState) -> Dict[str, Any]:
        chunks: List[RetrievedChunk] = (
            state.reranked_chunks or state.retrieved_chunks or []
        )

        try:
            min_required_chunks = 2

            # Relax rule for conversational follow-ups
            if state.chat_history:
                min_required_chunks = 1

            guarded_chunks = self.guardrails.apply_retrieval_guardrails(
                chunks=chunks,
                filters=state.filters,
                min_chunks=min_required_chunks,
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
            # Extract previous answer
            def strip_citations(text: str) -> str:
                return re.sub(r"\[\d+\]", "", text)

            previous_answer = ""
            if state.chat_history:
                previous_answer = strip_citations(
                    state.chat_history[-1]["content"]
                )
            cleaned_history = []
            for h in state.chat_history:
                cleaned_history.append({
                    "role": h["role"],
                    "content": strip_citations(h["content"])
                })
            # ---------------------------------------------
            # Generate Answer
            # ---------------------------------------------
            result = await self.answer_generator.generate(
                question=state.user_query,
                retrieved_chunks=chunks,
                chat_history=cleaned_history,
                previous_answer=previous_answer
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

        graph.add_node("rewrite", self.rewrite_node)
        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("guardrails", self.guardrail_node)
        graph.add_node("answer", self.answer_node)

        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "retrieve")
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
            session_id = f"doc:{(filters or {}).get('title', 'default')}"
            # -------------------------------------------------
            # 1. Generate deterministic cache key
            # -------------------------------------------------
            cache_key = self.cache.make_key(
                question=query,
                namespace=f"rag:{session_id}",
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
            # 3. Fetch conversation history (LAST 4 TURNS)
            # -------------------------------------------------
            chat_history = await self.conversation_store.get_recent_history(
                session_id=session_id,
                limit=4
                )
            # -------------------------------------------------
            # 4. Run Graph (cold path)
            # -------------------------------------------------
            final_state = await self.graph.ainvoke(
                RAGState(
                    user_query=query,
                    filters=filters or {},
                    chat_history=chat_history
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

            # Save assistant response to conversation storage
            if response["answer"] and not response["answer"].startswith("The question cannot be answered safely"):
                try:
                    await self.conversation_store.save_qa(
                        query=query,
                        answer=response["answer"],
                        citations=response["citations"],
                        metadata={
                                "filters": filters,
                                "session_id": session_id,
                                "citation_count": len(response["citations"]),
                                "query_length": len(query),
                                "has_cache": cached is not None,
                            }
                    )
                except Exception as e:
                    log.error("conversation_store_save_failed", error=str(e))

            # -------------------------------------------------
            # 4. Store Only Valid Answers
            # -------------------------------------------------
            if response["answer"] and not response["answer"].startswith("The question cannot be answered safely"):
                await self.cache.set(
                    cache_key,
                    response,
                )

            log.info("rag_run_completed")

            return response

        except Exception as e:
            log.error("rag_pipeline_failed", error=str(e))
            raise RegulatoryRAGException(e)