from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from src.shared.schemas.retrieval import RetrievedChunk

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

class QueryRequest(BaseModel):
    """Represents incoming query request payload."""
    question: str
    thread_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None