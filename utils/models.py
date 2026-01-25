"""
Pydantic models used across the ingestion pipeline.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Ingestion configuration
# ------------------------------------------------------------------

class IngestionConfig(BaseModel):
    """
    Configuration used by the ingestion pipeline.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    use_semantic_chunking: bool = True


# ------------------------------------------------------------------
# Ingestion result
# ------------------------------------------------------------------

class IngestionResult(BaseModel):
    """
    Result returned after ingesting a single document.
    """
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# Document model (DB representation)
# ------------------------------------------------------------------

class Document(BaseModel):
    """
    Represents a document stored in the database.
    """
    id: Optional[str] = None
    title: str
    source: str
    file_hash: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ------------------------------------------------------------------
# Chunk model (optional, useful for debugging / testing)
# ------------------------------------------------------------------

class Chunk(BaseModel):
    """
    Represents a chunk stored in the database.
    """
    id: Optional[str] = None
    document_id: str
    content: str
    chunk_index: int
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

# ------------------------------------------------------------------
# Retriever models
# ------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalState(BaseModel):
    # input
    user_query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    # retrieval
    bm25_results: List[RetrievedChunk] = Field(default_factory=list)
    vector_results: List[RetrievedChunk] = Field(default_factory=list)
    fused_results: List[RetrievedChunk] = Field(default_factory=list)
# ------------------------------------------------------------------
# Output models
# ------------------------------------------------------------------

class Citation(BaseModel):
    """
    Clean citation returned to the user.
    """
    id: int
    source: str
    document_id: str
    chunk_id: str


class AnswerOutput(BaseModel):
    answer: str
    evidence: List[str]
    citations: List[Citation]

# ------------------------------------------------------------------
# RAG State model
# ------------------------------------------------------------------
class RAGState(BaseModel):
    """
    State object used across the LangGraph RAG pipeline.
    """

    # Input
    user_query: str
    filters: Dict[str, Any] = Field(default_factory=dict)

    # Retrieval
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: List[RetrievedChunk] = Field(default_factory=list)

    # Output
    answer: Optional[str] = None
    citations: List[RetrievedChunk] = Field(default_factory=list)