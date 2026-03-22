"""
Semantic chunker (Docling removed)

Uses:
- LlamaIndex SemanticSplitter
- Recursive fallback
"""

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

    def __init__(self, config: ChunkingConfig):

        self.config = config

        try:

            embedding_client = ModelLoader().load_embeddings()

            embed_model = LangchainEmbedding(
                langchain_embeddings=embedding_client
            )

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

        # # fallback splitter
        # self.fallback = RecursiveCharacterTextSplitter(
        #     chunk_size=config.chunk_size,
        #     chunk_overlap=config.chunk_overlap,
        # )


    async def chunk_document(
        self,
        *,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[DocumentChunk]:

        if not content.strip():
            return []

        try:

            doc = Document(text=content)

            nodes = self.parser.get_nodes_from_documents(
                [doc]
            )

            chunks: List[DocumentChunk] = []

            start_pos = 0

            for idx, node in enumerate(nodes):

                text = node.text.strip()

                if not text:
                    continue

                found_at = content.find(
                    text,
                    start_pos,
                )

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

            # return await self._fallback(
            #     content,
            #     title,
            #     source,
            #     metadata,
            # )

    # -------------------------------------------------

    # async def _fallback(
    #     self,
    #     content,
    #     title,
    #     source,
    #     metadata,
    # ):

    #     texts = self.fallback.split_text(
    #         content
    #     )

    #     chunks = []

    #     start_pos = 0

    #     for idx, text in enumerate(texts):

    #         text = text.strip()

    #         if not text:
    #             continue

    #         found_at = content.find(
    #             text,
    #             start_pos,
    #         )

    #         if found_at == -1:
    #             found_at = start_pos

    #         end_pos = found_at + len(text)

    #         start_pos = end_pos

    #         chunks.append(
    #             DocumentChunk(
    #                 content=text,
    #                 index=idx,
    #                 start_char=found_at,
    #                 end_char=end_pos,
    #                 metadata={
    #                     "title": title,
    #                     "source": source,
    #                     "chunk_method": "recursive_fallback",
    #                     **(metadata or {}),
    #                 },
    #             )
    #         )

    #     return chunks


# =========================================================
# Factory
# =========================================================


def create_chunker(config: ChunkingConfig):
    return SemanticChunker(config)