"""
Document ingestion pipeline.

Features:
- Document-only ingestion (PDF / MD / TXT)
- Idempotent ingestion using file hash
- Skip unchanged documents
- Update database when document content changes
"""

import os
import glob
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import asyncpg
from src.chunking.chunker import create_chunker
from src.embeddings.embedder import create_embedder
from utils.db_utils import (
    initialize_database,
    close_database,
    get_document_by_hash,
    get_document_by_source,
    delete_document_and_chunks,
)
from utils import db_utils
from utils.models import IngestionConfig, IngestionResult, DocumentChunk, ChunkingConfig
from docling.document_converter import DocumentConverter
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


load_dotenv()
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
        if self._initialized:
            return

        try:
            await initialize_database()
            self._initialized = True
            log.info("ingestion_pipeline_initialized")
        except Exception as e:
            log.error("db_initialization_failed", error=str(e))
            raise RegulatoryRAGException(e)

    async def close(self):
        if not self._initialized:
            return

        try:
            await close_database()
            self._initialized = False
            log.info("ingestion_pipeline_closed")
        except Exception as e:
            log.error("db_shutdown_failed", error=str(e))
            raise RegulatoryRAGException(e)

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
            log.warning("no_supported_documents_found")
            return []

        log.info(
            "ingestion_started",
            document_count=len(document_files),
        )

        results: List[IngestionResult] = []

        for idx, file_path in enumerate(document_files, start=1):
            try:
                log.info(
                    "document_processing_started",
                    file=str(file_path),
                    index=idx,
                    total=len(document_files),
                )

                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(idx, len(document_files))

            except RegulatoryRAGException as e:
                # Already enriched, just log once
                log.error("document_ingestion_failed", error=str(e))
                results.append(
                    IngestionResult(
                        document_id="",
                        title=Path(file_path).name,
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

        log.info(
            "ingestion_completed",
            processed=len(results),
        )

        return results

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        start_time = datetime.now()

        try:
            content, docling_doc = self._read_document(file_path)
            title = self._extract_title(content, file_path)

            # Force title into retrievable text
            content = f"# {title}\n\n{content}"

            source = os.path.relpath(file_path, self.documents_folder)
            metadata = self._extract_document_metadata(content, file_path)
            file_hash = self._compute_file_hash(content)

            # -------------------------------
            # Idempotency check
            # -------------------------------
            existing = await get_document_by_hash(file_hash)
            if existing:
                log.info(
                    "document_skipped_hash_match",
                    title=title,
                )
                return IngestionResult(
                    document_id=existing["id"],
                    title=title,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=[],
                )

            # -------------------------------
            # Update existing source
            # -------------------------------
            existing_source = await get_document_by_source(source)
            if existing_source:
                log.info(
                    "document_updated_existing_source",
                    title=title,
                )
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
                log.warning(
                    "no_chunks_created",
                    title=title,
                )
                return IngestionResult(
                    document_id="",
                    title=title,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=["No chunks created"],
                )

            log.info(
                "chunking_completed",
                title=title,
                chunk_count=len(chunks),
            )

            # -------------------------------
            # Embed
            # -------------------------------
            embedded_chunks = await self.embedder.embed_chunks(chunks)

            log.info(
                "embedding_completed",
                title=title,
                chunk_count=len(embedded_chunks),
            )

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

            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            log.info(
                "document_ingested",
                title=title,
                document_id=document_id,
                duration_ms=int(processing_time),
            )

            return IngestionResult(
                document_id=document_id,
                title=title,
                chunks_created=len(chunks),
                processing_time_ms=processing_time,
                errors=[],
            )

        except Exception as e:
            raise RegulatoryRAGException(e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_document_files(self) -> List[str]:
        files: List[str] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, f"**/*{ext}"),
                    recursive=True,
                )
            )
        return files

    def _compute_file_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_document(self, file_path: str):
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            result = self.converter.convert(file_path)
            content = result.document.export_to_markdown()
            return content, result.document

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), None

    def _extract_title(self, content: str, file_path: str) -> str:
        for line in content.splitlines()[:10]:
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()
        return Path(file_path).stem

    def _extract_document_metadata(
        self,
        content: str,
        file_path: str,
    ) -> Dict[str, Any]:
        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
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
                try:
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

                except asyncpg.exceptions.UniqueViolationError:
                    existing = await conn.fetchrow(
                        """
                        SELECT id::text FROM documents WHERE file_hash = $1
                        """,
                        file_hash,
                    )
                    if existing:
                        log.warning(
                            "concurrent_ingest_detected",
                            file_hash=file_hash,
                        )
                        return existing["id"]

                    raise
