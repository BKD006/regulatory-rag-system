from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator


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