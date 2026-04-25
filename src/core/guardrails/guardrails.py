from typing import List, Optional
from utils.models import RetrievedChunk
from exception.custom_exception import RegulatoryRAGException
from logger import GLOBAL_LOGGER as log


class AnswerGuardrails:
    """
    Provides post-generation validation checks to ensure answer quality and grounding.

    Includes validation for citations and checks to ensure the generated answer
    is grounded in retrieved evidence.

    Attributes:
        None
    """

    # ==========================================================
    # Post-generation guardrails ONLY
    # ==========================================================

    @staticmethod
    def validate_citations(
        used_chunks: Optional[List[RetrievedChunk]],
        cited_chunk_ids: Optional[List[str]],
    ) -> None:
        """
        Validates that all cited chunk IDs exist within the used chunks.

        Args:
            used_chunks (Optional[List[RetrievedChunk]]): Chunks used to generate the answer.
            cited_chunk_ids (Optional[List[str]]): List of cited chunk IDs.

        Returns:
            None

        Raises:
            RegulatoryRAGException: If any cited chunk ID is not present in used_chunks.
        """
        
        if not used_chunks or not cited_chunk_ids:
            return

        allowed_ids = {c.chunk_id for c in used_chunks}

        invalid = [
            cid for cid in cited_chunk_ids
            if cid not in allowed_ids
        ]

        if invalid:
            log.warning("invalid_citations_detected", invalid_ids=invalid)
            raise RegulatoryRAGException(
                f"Invalid citations detected: {invalid}"
            )

    @staticmethod
    def filter_citations(
        citations: Optional[List[RetrievedChunk]],
        allowed_chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Filters citations to include only those present in allowed chunks.

        Args:
            citations (Optional[List[RetrievedChunk]]): List of citation chunks.
            allowed_chunks (List[RetrievedChunk]): Valid chunks allowed for citation.

        Returns:
            List[RetrievedChunk]: Filtered list of valid citation chunks.
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
        Ensures that the generated answer is sufficiently grounded in retrieved content.

        Computes token overlap between the answer and combined chunk content.

        Args:
            answer (str): Generated answer text.
            used_chunks (List[RetrievedChunk]): Chunks used as evidence.
            min_overlap_ratio (float): Minimum required overlap ratio.

        Returns:
            None

        Raises:
            RegulatoryRAGException:
                - If no evidence is available.
                - If overlap ratio is below threshold, indicating weak grounding.
        """

        if not answer.strip():
            return

        combined_text = " ".join(c.content for c in used_chunks)

        answer_tokens = set(answer.lower().split())
        chunk_tokens = set(combined_text.lower().split())

        if not chunk_tokens:
            raise RegulatoryRAGException(
                "No evidence available for grounding validation."
            )

        overlap = answer_tokens.intersection(chunk_tokens)
        overlap_ratio = len(overlap) / max(len(answer_tokens), 1)

        if overlap_ratio < min_overlap_ratio:
            log.warning("low_grounding_score", overlap_ratio=overlap_ratio)
            raise RegulatoryRAGException(
                "Answer may not be sufficiently grounded in evidence."
            )