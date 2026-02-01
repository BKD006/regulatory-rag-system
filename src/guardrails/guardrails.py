from typing import List, Dict, Any, Tuple
from utils.models import RetrievedChunk


class GuardrailViolation(Exception):
    """
    Raised when a HARD guardrail is violated.
    Should be caught at pipeline level.
    """
    pass


class AnswerGuardrails:
    """
    Raw Python guardrails for HARD safety enforcement.

    These guardrails:
    - Run BEFORE LLM generation
    - Can STOP the pipeline deterministically
    - Do NOT depend on LLM behavior
    """

    # ==========================================================
    # Retrieval-time guardrails (BEFORE answer generation)
    # ==========================================================

    @staticmethod
    def enforce_single_document(
        chunks: List[RetrievedChunk],
        filters: Dict[str, Any],
    ) -> List[RetrievedChunk]:
        """
        Enforce that all chunks come from a single document
        when a document-level filter is provided.
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

    # ==========================================================
    # Generation-time guardrails (AFTER retrieval, BEFORE LLM)
    # ==========================================================

    @staticmethod
    def enforce_single_source(
        chunks: List[RetrievedChunk],
    ):
        """
        Final HARD check: ensure only one document source exists.
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
        used_chunks: List[RetrievedChunk],
        cited_chunk_ids: List[str],
    ):
        """
        Ensure citations only reference chunks actually used.
        """
        allowed_ids = {c.chunk_id for c in used_chunks}

        invalid = [
            cid for cid in cited_chunk_ids
            if cid not in allowed_ids
        ]

        if invalid:
            raise GuardrailViolation(
                f"Invalid citations detected: {invalid}"
            )

    # ==========================================================
    # Convenience wrapper
    # ==========================================================

    @staticmethod
    def apply_retrieval_guardrails(
        chunks: List[RetrievedChunk],
        filters: Dict[str, Any],
        *,
        min_chunks: int = 1,
    ) -> List[RetrievedChunk]:
        """
        Apply ALL retrieval-time guardrails in order.
        """
        chunks = AnswerGuardrails.enforce_single_document(chunks, filters)
        chunks = AnswerGuardrails.enforce_min_chunks(chunks, min_chunks)
        AnswerGuardrails.enforce_single_source(chunks)
        return chunks

    @staticmethod
    def validate_citations(
        used_chunks: List[RetrievedChunk] | None,
        cited_chunk_ids: List[str] | None,
    ):
        """
        Ensure citations only reference chunks actually used.
        """
        if not used_chunks or not cited_chunk_ids:
            return  # ✅ SAFE NO-OP

        allowed_ids = {c.chunk_id for c in used_chunks}

        invalid = [
            cid for cid in cited_chunk_ids
            if cid not in allowed_ids
        ]

        if invalid:
            raise ValueError(
                f"Invalid citations detected: {invalid}"
            )
    @staticmethod
    def filter_citations(
        citations: List[RetrievedChunk] | None,
        allowed_chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:

        if not citations:
            return []

        allowed_ids = {c.chunk_id for c in allowed_chunks}

        return [
            c for c in citations
            if c.chunk_id in allowed_ids
        ]
