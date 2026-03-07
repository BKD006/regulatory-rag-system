from fastapi import APIRouter, UploadFile, File, HTTPException, status
import os
import shutil

from src.indexing.ingest import DocumentIngestionPipeline
from utils.models import IngestionConfig
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException

router = APIRouter(
    prefix="",
    tags=["Ingestion"],
)

DATA_DIR = "data"

# Singleton pipeline
ingestion_pipeline = DocumentIngestionPipeline(
    config=IngestionConfig(),
    documents_folder=DATA_DIR,
)


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a regulatory document (PDF).

    Steps:
    - Save uploaded file
    - Run ingestion pipeline
    - Chunk + embed + store
    """

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        # -----------------------------
        # Save uploaded file
        # -----------------------------
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        log.info(
            "ingest_request_received",
            filename=file.filename,
        )

        # -----------------------------
        # Run ingestion pipeline
        # -----------------------------
        results = await ingestion_pipeline.ingest_documents()

        return {
            "message": "Ingestion completed",
            "uploaded_file": file.filename,
            "files_processed": len(results),
            "results": [r.model_dump() for r in results],
        }

    except RegulatoryRAGException as e:

        log.error(
            "ingestion_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed",
        )

    except Exception as e:

        log.error(
            "unexpected_ingestion_error",
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error during ingestion",
        )