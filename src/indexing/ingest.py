"""
Document ingestion pipeline.

Features:
- Document-only ingestion (PDF / MD / TXT)
- Idempotent ingestion using file hash
- Skip unchanged documents
- Update database when document content changes
- Uses DoclingHybridChunker / SimpleChunker
- Uses Titan embeddings via embedder.py
"""

import os
import logging
import json
import glob
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

from src.chunking.chunker import ChunkingConfig, create_chunker, DocumentChunk
from src.embeddings.embedder import create_embedder

from utils.db_utils import (
    initialize_database,
    close_database,
    get_document_by_hash,
    get_document_by_source,
    delete_document_and_chunks,
)
from utils import db_utils

from utils.models import IngestionConfig, IngestionResult

from docling.document_converter import DocumentConverter

# -------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into PostgreSQL + pgvector."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "data",
    ):
        self.config = config
        self.documents_folder = documents_folder

        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking,
        )

        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.converter = DocumentConverter()

        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        if not self._initialized:
            await initialize_database()
            self._initialized = True
            logger.info("Ingestion pipeline initialized")

    async def close(self):
        if self._initialized:
            await close_database()
            self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None,
    ) -> List[IngestionResult]:

        if not self._initialized:
            await self.initialize()

        document_files = self._find_document_files()

        if not document_files:
            logger.warning("No supported documents found")
            return []

        results: List[IngestionResult] = []

        for idx, file_path in enumerate(document_files):
            try:
                logger.info(f"Processing {idx + 1}/{len(document_files)}: {file_path}")
                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(idx + 1, len(document_files))

            except Exception as e:
                logger.exception(f"Failed to ingest {file_path}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        start_time = datetime.now()

        content, docling_doc = self._read_document(file_path)
        title = self._extract_title(content, file_path)
        source = os.path.relpath(file_path, self.documents_folder)
        metadata = self._extract_document_metadata(content, file_path)

        file_hash = self._compute_file_hash(content)

        # -------------------------------
        # Idempotency check (hash-based)
        # -------------------------------
        existing = await get_document_by_hash(file_hash)
        if existing:
            logger.info(f"Skipping unchanged document: {title}")
            return IngestionResult(
                document_id=existing["id"],
                title=title,
                chunks_created=0,
                processing_time_ms=0,
                errors=[],
            )

        # ----------------------------------------
        # Same source but changed content → update
        # ----------------------------------------
        existing_source = await get_document_by_source(source)
        if existing_source:
            logger.info(f"Updating modified document: {title}")
            await delete_document_and_chunks(existing_source["id"])

        # -------------------------------
        # Chunk
        # -------------------------------
        chunks = await self.chunker.chunk_document(
            content=content,
            title=title,
            source=source,
            metadata=metadata,
            docling_doc=docling_doc,
        )

        if not chunks:
            return IngestionResult(
                document_id="",
                title=title,
                chunks_created=0,
                processing_time_ms=0,
                errors=["No chunks created"],
            )

        # -------------------------------
        # Embed
        # -------------------------------
        embedded_chunks = await self.embedder.embed_chunks(chunks)

        # -------------------------------
        # Persist
        # -------------------------------
        document_id = await self._save_to_postgres(
            title=title,
            source=source,
            content=content,
            file_hash=file_hash,
            chunks=embedded_chunks,
            metadata=metadata,
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IngestionResult(
            document_id=document_id,
            title=title,
            chunks_created=len(chunks),
            processing_time_ms=processing_time,
            errors=[],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_document_files(self) -> List[str]:
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(
                glob.glob(os.path.join(self.documents_folder, f"**/*{ext}"), recursive=True)
            )
        return files

    def _compute_file_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_document(self, file_path: str):
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            result = self.converter.convert(file_path)
            return result.document.export_to_markdown(), result.document

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), None

    def _extract_title(self, content: str, file_path: str) -> str:
        for line in content.splitlines()[:10]:
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()
        return Path(file_path).stem

    def _extract_document_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            "file_path": file_path,
            "file_size": len(content),
            "line_count": len(content.splitlines()),
            "word_count": len(content.split()),
            "ingested_at": datetime.now().isoformat(),
        }

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        file_hash: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any],
    ) -> str:

        async with db_utils.db_pool.acquire() as conn:
            async with conn.transaction():
                doc = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, file_hash, content, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    file_hash,
                    content,
                    json.dumps(metadata),
                )

                document_id = doc["id"]

                for chunk in chunks:
                    embedding = (
                        "[" + ",".join(map(str, chunk.embedding)) + "]"
                        if chunk.embedding
                        else None
                    )

                    await conn.execute(
                        """
                        INSERT INTO chunks
                        (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        chunk.token_count,
                    )

                return document_id
