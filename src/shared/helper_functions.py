from typing import List
from utils.models import RetrievedChunk, Citation


def format_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    """
    Convert internal RetrievedChunk objects into clean, user-facing citations.
    - Deduplicates chunks
    - Assigns stable numeric IDs
    - Hides internal fields
    """

    seen = set()
    citations: List[Citation] = []
    idx = 1

    for chunk in chunks:
        key = (chunk.document_id, chunk.chunk_id)
        if key in seen:
            continue

        seen.add(key)

        citations.append(
            Citation(
                id=idx,
                source=chunk.source,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
            )
        )
        idx += 1

    return citations
