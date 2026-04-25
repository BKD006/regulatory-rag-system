from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.shared.schemas.retrieval import RetrievedChunk

# ------------------------------------------------------------------
# RAG State model
# ------------------------------------------------------------------

class RAGState(BaseModel):
    """Maintains state across the RAG pipeline execution."""
    user_query: str
    filters: Dict[str, Any] = Field(default_factory=dict)

    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: List[RetrievedChunk] = Field(default_factory=list)

    #Currently unused in pipeline execution
    # guarded_chunks: Optional[List[RetrievedChunk]] = Field(default_factory=list)

    answer: Optional[str] = None
    citations: List[RetrievedChunk] = Field(default_factory=list)

    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    rewritten_query: Optional[str] = None
    previous_chunks: List[RetrievedChunk] = Field(default_factory=list)