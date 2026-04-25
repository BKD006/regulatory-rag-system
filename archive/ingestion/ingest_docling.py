import os
import glob
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from src.chunking.chunker_docling import create_chunker
from src.embeddings.embedder import create_embedder
from src.ingestion.cleaning import DocumentCleaner
from utils.db_utils import (
    initialize_database,
    close_database,
    get_document_by_hash,
    get_document_by_source,
    delete_document_and_chunks,
)
from utils import db_utils
from utils.models import IngestionConfig, IngestionResult, DocumentChunk, ChunkingConfig
from src.parsing.docling_parser import DoclingDocumentParser
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


load_dotenv()
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


class DocumentIngestionPipeline:
    """
    End-to-end pipeline for ingesting documents into the system.

    Handles parsing, cleaning, chunking, embedding, and storing documents
    into a PostgreSQL database.

    Attributes:
        config (IngestionConfig): Configuration for ingestion parameters.
        documents_folder (str): Root folder containing documents to ingest.
        cleaner (DocumentCleaner): Component for cleaning and normalizing content.
        chunker_config (ChunkingConfig): Configuration for chunking strategy.
        chunker (object): Chunking component for splitting documents.
        embedder (object): Embedding generator for chunks.
        parser (DoclingDocumentParser): Parser for extracting document content.
        _initialized (bool): Tracks whether the database is initialized.
    """

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "data",
    ):
        """
        Initializes the ingestion pipeline with required components.

        Args:
            config (IngestionConfig): Configuration controlling chunking and ingestion.
            documents_folder (str): Directory containing documents to process.
        """

        self.config = config
        self.documents_folder = documents_folder

        # Cleaning layer (layout + text normalization)
        self.cleaner = DocumentCleaner()

        # Chunking configuration
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking,
        )

        # Core pipeline components
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.parser = DoclingDocumentParser()

        # Internal state
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """
        Initializes database connections required for ingestion.

        Returns:
            None

        Raises:
            RegulatoryRAGException: If database initialization fails.
        """

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
        """
        Closes database connections used by the pipeline.

        Returns:
            None

        Raises:
            RegulatoryRAGException: If database shutdown fails.
        """

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
        """
        Ingests all supported documents from the configured folder.

        Processes documents sequentially and optionally reports progress.

        Args:
            progress_callback (Optional[callable]):
                Function to report progress with signature (current, total).

        Returns:
            List[IngestionResult]: List of ingestion results for each document.
        """

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

        log.info("ingestion_completed", processed=len(results))

        return results

    # ------------------------------------------------------------------
    # Core ingestion logic
    # ------------------------------------------------------------------

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Processes a single document through the full ingestion pipeline.

        Args:
            file_path (str): Path to the document file.

        Returns:
            IngestionResult: Result containing document ID, chunk count, and status.

        Raises:
            RegulatoryRAGException: If ingestion fails at any stage.
        """

        start_time = datetime.now()

        try:
            # -------------------------------
            # Parse document
            # -------------------------------
            content, docling_doc = self.parser.parse(file_path)

            # -------------------------------
            # Clean content
            # -------------------------------
            if docling_doc is not None:
                content, cleaning_metadata = self.cleaner.clean(content, docling_doc)
            else:
                cleaning_metadata = {"layout_cleaning": False}

            # -------------------------------
            # Title extraction
            # -------------------------------
            title = self._extract_title(content, file_path)

            content = f"# {title}\n\n{content}"

            # -------------------------------
            # Metadata + hashing
            # -------------------------------
            source = os.path.relpath(file_path, self.documents_folder)
            metadata = self._extract_document_metadata(content, file_path)
            metadata.update(cleaning_metadata)

            file_hash = self._compute_file_hash(content)

            # Idempotency check
            existing = await get_document_by_hash(file_hash)

            if existing:
                log.info("document_skipped_hash_match", title=title)

                return IngestionResult(
                    document_id=existing["id"],
                    title=title,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=[],
                )

            # Update existing document
            existing_source = await get_document_by_source(source)

            if existing_source:
                log.info("document_updated_existing_source", title=title)
                await delete_document_and_chunks(existing_source["id"])

            # Chunking
            chunks = await self.chunker.chunk_document(
                content=content,
                title=title,
                source=source,
                metadata=metadata,
                docling_doc=docling_doc,
            )

            if not chunks:
                log.warning("no_chunks_created", title=title)

                return IngestionResult(
                    document_id="",
                    title=title,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=["No chunks created"],
                )

            log.info("chunking_completed", title=title, chunk_count=len(chunks))

            # Embedding
            embedded_chunks = await self.embedder.embed_chunks(chunks)

            log.info(
                "embedding_completed",
                title=title,
                chunk_count=len(embedded_chunks),
            )

            # Persistence
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
        """
        Finds all supported document files in the documents folder.

        Returns:
            List[str]: List of file paths matching supported extensions.
        """
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
        """
        Computes a SHA-256 hash of document content.

        Args:
            content (str): Document content.

        Returns:
            str: Hexadecimal hash string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_document(self, file_path: str):
        """
        Reads document content directly from file.

        Args:
            file_path (str): Path to the document.

        Returns:
            Tuple[str, Optional[object]]: Content and optional structured document.
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            result = self.converter.convert(file_path)
            content = result.document.export_to_markdown()
            return content, result.document

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), None

    def _extract_title(self, content: str, file_path: str) -> str:
        """
        Extracts document title from content or filename.

        Args:
            content (str): Document content.
            file_path (str): Path to the file.

        Returns:
            str: Extracted or inferred title.
        """
        for line in content.splitlines()[:10]:
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()
        return Path(file_path).stem

    def _extract_document_metadata(
        self,
        content: str,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Extracts metadata from document content and file.

        Args:
            content (str): Document content.
            file_path (str): Path to the file.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
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
        """
        Saves document and its chunks into PostgreSQL database.

        Args:
            title (str): Document title.
            source (str): Relative source path.
            content (str): Full document content.
            file_hash (str): Unique content hash.
            chunks (List[DocumentChunk]): List of processed chunks.
            metadata (Dict[str, Any]): Metadata for the document.

        Returns:
            str: Inserted document ID.
        """

        async with db_utils.db_pool.acquire() as conn:
            async with conn.transaction():

                doc = await conn.fetchrow(
                    """
                    INSERT INTO public.documents
                    (title, source, file_hash, content, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (file_hash)
                    DO UPDATE SET updated_at = now()
                    RETURNING id::text
                    """,
                    title,
                    source,
                    file_hash,
                    content,
                    json.dumps(metadata),
                )

                document_id = doc["id"]

                await conn.execute(
                    """
                    DELETE FROM public.chunks
                    WHERE document_id = $1::uuid
                    """,
                    document_id,
                )

                for chunk in chunks:
                    embedding = (
                        "[" + ",".join(map(str, chunk.embedding)) + "]"
                        if chunk.embedding
                        else None
                    )

                    await conn.execute(
                        """
                        INSERT INTO public.chunks
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