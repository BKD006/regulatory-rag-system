"""
Document chunking strategies.

- Docling HybridChunker (preferred)
- RecursiveCharacterTextSplitter fallback
- NO per-chunk logging
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from utils.models import DocumentChunk, ChunkingConfig
from logger import GLOBAL_LOGGER as log

# ------------------------------------------------------------------
# Docling Hybrid Chunker
# ------------------------------------------------------------------

class DoclingHybridChunker:
    """
    Wrapper around Docling HybridChunker.

    Guarantees:
    - Token-aware chunking
    - Structure preservation
    - Summary-level logging only
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

        if config.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.chunker = HybridChunker(
                tokenizer=self.tokenizer,
                max_tokens=config.max_tokens,
                merge_peers=True,
            )

            log.info(
                "hybrid_chunker_initialized",
                max_tokens=config.max_tokens,
            )

        except Exception as e:
            log.error("hybrid_chunker_init_failed", error=str(e))
            raise

    async def chunk_document(
        self,
        *,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[DoclingDocument] = None,
    ) -> List[DocumentChunk]:

        if not content or not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {}),
        }

        if docling_doc is None:
            log.warning(
                "docling_doc_missing_fallback_used",
                title=title,
            )
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            raw_chunks = list(self.chunker.chunk(dl_doc=docling_doc))
            document_chunks: List[DocumentChunk] = []

            current_pos = 0

            for idx, chunk in enumerate(raw_chunks):
                text = self.chunker.contextualize(chunk=chunk)
                token_count = len(self.tokenizer.encode(text))

                end_pos = current_pos + len(text)

                document_chunks.append(
                    DocumentChunk(
                        content=text.strip(),
                        index=idx,
                        start_char=current_pos,
                        end_char=end_pos,
                        metadata={
                            **base_metadata,
                            "total_chunks": len(raw_chunks),
                            "token_count": token_count,
                            "has_context": True,
                        },
                        token_count=token_count,
                    )
                )

                current_pos = end_pos

            log.info(
                "hybrid_chunking_completed",
                title=title,
                chunk_count=len(document_chunks),
            )

            return document_chunks

        except Exception as e:
            log.warning(
                "hybrid_chunking_failed_fallback_used",
                title=title,
                error=str(e),
            )
            return self._simple_fallback_chunk(content, base_metadata)

    # ------------------------------------------------------------------
    # Fallback chunking
    # ------------------------------------------------------------------

    def _simple_fallback_chunk(
        self,
        content: str,
        base_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        texts = splitter.split_text(content)

        chunks: List[DocumentChunk] = []
        start_pos = 0

        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue

            found_at = content.find(text, start_pos)
            if found_at == -1:
                found_at = start_pos

            end_pos = found_at + len(text)
            start_pos = end_pos

            chunks.append(
                DocumentChunk(
                    content=text,
                    index=idx,
                    start_char=found_at,
                    end_char=end_pos,
                    metadata={
                        **base_metadata,
                        "chunk_method": "simple_fallback",
                        "total_chunks": len(texts),
                        "has_context": False,
                    },
                )
            )

        log.info(
            "fallback_chunking_completed",
            chunk_count=len(chunks),
        )

        return chunks


# ------------------------------------------------------------------
# Simple Recursive Chunker (explicit opt-in)
# ------------------------------------------------------------------

class SimpleChunker:
    """
    Simple non-semantic chunker.

    Use only when semantic chunking is disabled.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        log.info(
            "simple_chunker_initialized",
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )

    async def chunk_document(
        self,
        *,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **_,
    ) -> List[DocumentChunk]:

        if not content or not content.strip():
            return []

        texts = self.splitter.split_text(content)

        chunks: List[DocumentChunk] = []
        start_pos = 0

        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue

            found_at = content.find(text, start_pos)
            if found_at == -1:
                found_at = start_pos

            end_pos = found_at + len(text)
            start_pos = end_pos

            chunks.append(
                DocumentChunk(
                    content=text,
                    index=idx,
                    start_char=found_at,
                    end_char=end_pos,
                    metadata={
                        "title": title,
                        "source": source,
                        "chunk_method": "recursive",
                        "total_chunks": len(texts),
                        **(metadata or {}),
                    },
                )
            )

        log.info(
            "simple_chunking_completed",
            title=title,
            chunk_count=len(chunks),
        )

        return chunks


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_chunker(config: ChunkingConfig):
    """
    Create appropriate chunker based on configuration.
    """
    if config.use_semantic_splitting:
        return DoclingHybridChunker(config)
    return SimpleChunker(config)
