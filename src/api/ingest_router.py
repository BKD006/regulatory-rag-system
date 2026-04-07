from fastapi import APIRouter, UploadFile, File, HTTPException, status
import os
import shutil

from src.ingestion.ingest_llamaparse import DocumentIngestionPipeline
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
    Uploads a document and triggers the ingestion pipeline.

    The file is saved locally and then processed through the ingestion pipeline,
    which includes parsing, chunking, embedding, and storing in the database.

    Args:
        file (UploadFile): File uploaded by the user.

    Returns:
        Dict[str, Any]: Response containing:
            - message (str): Status message.
            - uploaded_file (str): Name of the uploaded file.
            - files_processed (int): Number of processed files.
            - results (List[Dict]): Detailed ingestion results per file.

    Raises:
        HTTPException:
            - 500: If ingestion fails due to processing error.
            - 500: If an unexpected server error occurs.
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