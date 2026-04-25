from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------
# Ingestion configuration
# ------------------------------------------------------------------

class IngestionConfig(BaseModel):
    """Defines ingestion configuration parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    use_semantic_chunking: bool = True


# ------------------------------------------------------------------
# Ingestion result
# ------------------------------------------------------------------

class IngestionResult(BaseModel):
    """Represents the result of a document ingestion process."""
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# Retriever models
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Output models
# ------------------------------------------------------------------

class Citation(BaseModel):
    """Represents a user-facing citation reference."""
    id: int
    source: str
    document_id: str
    chunk_id: str


class AnswerResult(BaseModel):
    """Represents generated answer with supporting citations."""
    answer: str
    citations: List[RetrievedChunk]


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


class QueryRequest(BaseModel):
    """Represents incoming query request payload."""
    question: str
    thread_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class ChunkingConfig(BaseModel):
    """Defines configuration for document chunking."""
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_chunk_size: int = Field(default=2000, gt=0)
    min_chunk_size: int = Field(default=100, gt=0)
    use_semantic_splitting: bool = True
    preserve_structure: bool = True
    max_tokens: int = Field(default=512, gt=0)

    @model_validator(mode="after")
    def validate_config(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size")
        return self


class DocumentChunk(BaseModel):
    """Represents a chunk of document with optional embedding."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None

    @model_validator(mode="after")
    def compute_token_count(self):
        if self.token_count is None:
            self.token_count = max(1, len(self.content) // 4)
        return self