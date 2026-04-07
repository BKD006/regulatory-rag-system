from typing import List, Dict, Any, Optional
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
    Hybrid chunker that leverages Docling structure-aware chunking with token constraints.

    Falls back to simple recursive text splitting when structured chunking is unavailable
    or fails.

    Attributes:
        config (ChunkingConfig): Configuration controlling chunk sizes and overlap.
        tokenizer (AutoTokenizer): Tokenizer used for token-aware chunking.
        chunker (HybridChunker): Docling hybrid chunking engine.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initializes the DoclingHybridChunker with tokenizer and hybrid chunker.

        Args:
            config (ChunkingConfig): Configuration for chunking behavior.

        Raises:
            ValueError: If max_tokens is not greater than 0.
            Exception: If tokenizer or chunker initialization fails.
        """
        
        self.config = config

        if config.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

        try:
            # Initialize tokenizer for token-aware chunking
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

            # Initialize Docling hybrid chunker
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
        """
        Splits a document into structured chunks using Docling or fallback logic.

        Args:
            content (str): Full document content.
            title (str): Title of the document.
            source (str): Source identifier (e.g., file path).
            metadata (Optional[Dict[str, Any]]): Additional metadata to attach to each chunk.
            docling_doc (Optional[DoclingDocument]): Structured Docling document.

        Returns:
            List[DocumentChunk]: List of generated document chunks.

        Notes:
            - Uses Docling hybrid chunking if structured document is provided.
            - Falls back to simple chunking if docling_doc is None or chunking fails.
        """

        if not content or not content.strip():
            return []

        # Base metadata applied to all chunks
        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {}),
        }

        # Fallback if Docling structure is unavailable
        if docling_doc is None:
            log.warning(
                "docling_doc_missing_fallback_used",
                title=title,
            )
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            # Generate structured chunks using Docling
            raw_chunks = list(self.chunker.chunk(dl_doc=docling_doc))
            document_chunks: List[DocumentChunk] = []

            current_pos = 0

            for idx, chunk in enumerate(raw_chunks):
                # Convert chunk to contextualized text
                text = self.chunker.contextualize(chunk=chunk)

                # Token count for chunk
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
            # Fallback on failure
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
        """
        Splits content using a simple recursive character-based strategy.

        Args:
            content (str): Raw document content.
            base_metadata (Dict[str, Any]): Metadata applied to all chunks.

        Returns:
            List[DocumentChunk]: List of fallback chunks.
        """

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

            # Locate chunk position in original content
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
# Factory
# ------------------------------------------------------------------

def create_chunker(config: ChunkingConfig):
    """
    Factory function to create a DoclingHybridChunker instance.

    Args:
        config (ChunkingConfig): Configuration for chunking.

    Returns:
        DoclingHybridChunker: Initialized chunker instance.
    """
    
    return DoclingHybridChunker(config)