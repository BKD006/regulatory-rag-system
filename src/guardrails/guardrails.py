"""
Deterministic HARD guardrails for Regulatory RAG.

These guardrails:
- Run before or after LLM deterministically
- Do NOT depend on model behavior
- Can STOP the pipeline safely
"""

from typing import List, Dict, Any, Optional
from utils.models import RetrievedChunk


# ==========================================================
# Exception
# ==========================================================

class GuardrailViolation(Exception):
    """
    Raised when a HARD guardrail is violated.
    Should be caught at pipeline level.
    """
    pass


# ==========================================================
# Guardrails
# ==========================================================

class AnswerGuardrails:
    """
    Deterministic guardrails for RAG pipeline.

    Phases:
    1. Retrieval-time
    2. Generation-time (pre-LLM)
    3. Post-generation
    """

    # ==========================================================
    # Retrieval-time guardrails (BEFORE answer generation)
    # ==========================================================

    @staticmethod
    def enforce_single_document(
        chunks: List[RetrievedChunk],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        """
        If document-level filter is provided,
        ensure all chunks come from that document.
        """
        if not filters:
            return chunks

        title = filters.get("title")
        if not title:
            return chunks

        filtered = [c for c in chunks if c.source == title]

        if not filtered:
            raise GuardrailViolation(
                f"No evidence found for document '{title}'."
            )

        return filtered

    @staticmethod
    def enforce_min_chunks(
        chunks: List[RetrievedChunk],
        min_chunks: int = 1,
    ) -> List[RetrievedChunk]:
        """
        Ensure minimum evidence exists.
        """
        if len(chunks) < min_chunks:
            raise GuardrailViolation(
                "Insufficient evidence to answer the question."
            )
        return chunks

    @staticmethod
    def enforce_single_source(
        chunks: List[RetrievedChunk],
    ) -> None:
        """
        Ensure all chunks come from a single document.
        """
        sources = {c.source for c in chunks}

        if len(sources) > 1:
            raise GuardrailViolation(
                f"Multiple document sources detected: {sorted(sources)}"
            )

    # ==========================================================
    # Post-generation guardrails (AFTER LLM)
    # ==========================================================

    @staticmethod
    def validate_citations(
        used_chunks: Optional[List[RetrievedChunk]],
        cited_chunk_ids: Optional[List[str]],
    ) -> None:
        """
        Ensure citations only reference chunks actually used.
        """
        if not used_chunks or not cited_chunk_ids:
            return  # safe no-op

        allowed_ids = {c.chunk_id for c in used_chunks}

        invalid = [
            cid for cid in cited_chunk_ids
            if cid not in allowed_ids
        ]

        if invalid:
            raise GuardrailViolation(
                f"Invalid citations detected: {invalid}"
            )

    @staticmethod
    def filter_citations(
        citations: Optional[List[RetrievedChunk]],
        allowed_chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Remove any citations not present in allowed_chunks.
        """
        if not citations:
            return []

        allowed_ids = {c.chunk_id for c in allowed_chunks}

        return [
            c for c in citations
            if c.chunk_id in allowed_ids
        ]

    @staticmethod
    def enforce_answer_grounded(
        answer: str,
        used_chunks: List[RetrievedChunk],
        *,
        min_overlap_ratio: float = 0.02,
    ) -> None:
        """
        Basic grounding heuristic:
        Ensure some overlap exists between answer text
        and combined chunk content.

        This is lightweight and deterministic.
        """

        if not answer.strip():
            return

        combined_text = " ".join(c.content for c in used_chunks)

        # Simple token-based overlap
        answer_tokens = set(answer.lower().split())
        chunk_tokens = set(combined_text.lower().split())

        if not chunk_tokens:
            raise GuardrailViolation(
                "No evidence available for grounding validation."
            )

        overlap = answer_tokens.intersection(chunk_tokens)

        overlap_ratio = len(overlap) / max(len(answer_tokens), 1)

        if overlap_ratio < min_overlap_ratio:
            raise GuardrailViolation(
                "Answer may not be sufficiently grounded in evidence."
            )

    # ==========================================================
    # Convenience wrapper
    # ==========================================================

    @staticmethod
    def apply_retrieval_guardrails(
        chunks: List[RetrievedChunk],
        filters: Optional[Dict[str, Any]],
        *,
        min_chunks: int = 1,
    ) -> List[RetrievedChunk]:
        """
        Apply ALL retrieval-time guardrails in order.
        """
        chunks = AnswerGuardrails.enforce_single_document(chunks, filters)
        chunks = AnswerGuardrails.enforce_min_chunks(chunks, min_chunks)
        AnswerGuardrails.enforce_single_source(chunks) # It is a validation function, it will raise if violated but won't modify the chunks.
        return chunks
