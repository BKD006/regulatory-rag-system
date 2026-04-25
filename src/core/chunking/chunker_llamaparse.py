from typing import List, Dict, Any, Optional

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding
from utils.model_loader import ModelLoader
from llama_index.core.schema import Document

from utils.models import DocumentChunk, ChunkingConfig
from logger import GLOBAL_LOGGER as log


# =========================================================
# Semantic Chunker
# =========================================================
class SemanticChunker:
    """
    Semantic chunker that splits documents based on meaning using embeddings.

    Uses LlamaIndex's SemanticSplitterNodeParser to create chunks based on
    semantic boundaries rather than fixed token/character limits.

    Attributes:
        config (ChunkingConfig): Configuration controlling chunking behavior.
        parser (SemanticSplitterNodeParser): Semantic node parser instance.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initializes the SemanticChunker with embedding model and semantic parser.

        Args:
            config (ChunkingConfig): Configuration for chunking.

        Raises:
            Exception: If embedding model or parser initialization fails.
        """
        self.config = config

        try:
            # Load embedding client (e.g., Bedrock / OpenAI / etc.)
            embedding_client = ModelLoader().load_embeddings()

            # Wrap LangChain embeddings for LlamaIndex compatibility
            embed_model = LangchainEmbedding(
                langchain_embeddings=embedding_client
            )

            # Initialize semantic splitter
            self.parser = SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95,
            )

            log.info(
                "semantic_chunker_initialized"
            )

        except Exception as e:
            log.error(
                "semantic_chunker_failed",
                error=str(e),
            )
            raise

    async def chunk_document(
        self,
        *,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[DocumentChunk]:
        """
        Splits document content into semantically meaningful chunks.

        Args:
            content (str): Full document content.
            title (str): Title of the document.
            source (str): Source identifier (e.g., file path).
            metadata (Optional[Dict[str, Any]]): Additional metadata for chunks.
            **kwargs: Additional unused parameters for compatibility.

        Returns:
            List[DocumentChunk]: List of semantic chunks.

        Notes:
            - Uses embedding-based semantic segmentation.
            - Returns empty list if chunking fails (fallback handled externally).
        """

        if not content.strip():
            return []

        try:
            # Wrap content into LlamaIndex Document
            doc = Document(text=content)

            # Generate semantic nodes
            nodes = self.parser.get_nodes_from_documents([doc])

            chunks: List[DocumentChunk] = []
            start_pos = 0

            for idx, node in enumerate(nodes):

                text = node.text.strip()

                if not text:
                    continue

                # Locate chunk position in original content
                found_at = content.find(
                    text,
                    start_pos,
                )

                if found_at == -1:
                    found_at = start_pos

                end_pos = found_at + len(text)
                start_pos = end_pos

                # Build chunk object
                chunks.append(
                    DocumentChunk(
                        content=text,
                        index=idx,
                        start_char=found_at,
                        end_char=end_pos,
                        metadata={
                            "title": title,
                            "source": source,
                            "chunk_method": "semantic",
                            **(metadata or {}),
                        },
                    )
                )

            log.info(
                "semantic_chunking_completed",
                title=title,
                chunk_count=len(chunks),
            )

            return chunks

        except Exception as e:
            log.warning(
                "semantic_failed_fallback",
                error=str(e),
            )
            return []  # fallback handled externally


# =========================================================
# Factory
# =========================================================

def create_chunker(config: ChunkingConfig):
    """
    Factory function to create a SemanticChunker instance.

    Args:
        config (ChunkingConfig): Configuration for chunking.

    Returns:
        SemanticChunker: Initialized semantic chunker.
    """
    return SemanticChunker(config)