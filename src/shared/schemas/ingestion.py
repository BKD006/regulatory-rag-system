from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

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