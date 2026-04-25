from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    """Represents a retrieved document chunk with metadata and score."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalState(BaseModel):
    """Holds intermediate retrieval state for hybrid retrieval pipeline."""
    user_query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    bm25_results: List[RetrievedChunk] = Field(default_factory=list)
    vector_results: List[RetrievedChunk] = Field(default_factory=list)
    fused_results: List[RetrievedChunk] = Field(default_factory=list)