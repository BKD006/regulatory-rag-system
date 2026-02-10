from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import shutil
import uuid
import json
from src.indexing.ingest import DocumentIngestionPipeline
from src.pipelines.rag_pipeline import RAGPipeline
from utils.models import IngestionConfig, QueryRequest
from utils.db_utils import (
    list_documents,
    list_document_titles,
    delete_document_by_title,
    init_db_pool,
    close_db_pool,
)
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException

# ------------------------------------------------------------------
# App init
# ------------------------------------------------------------------

app = FastAPI(
    title="Regulatory RAG API",
    version="1.0.0",
)

DATA_DIR = "data"

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# ------------------------------------------------------------------
# Pipelines (singletons)
# ------------------------------------------------------------------

ingestion_pipeline = DocumentIngestionPipeline(
    config=IngestionConfig(),
    documents_folder=DATA_DIR,
)

rag_pipeline = RAGPipeline()

# ------------------------------------------------------------------
# Health / UI
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    try:
        await init_db_pool()
        log.info("api_startup_completed")
    except Exception as e:
        log.error("api_startup_failed", error=str(e))
        raise


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await close_db_pool()
        log.info("db_pool_closed")
    except Exception as e:
        log.error("db_pool_close_failed", error=str(e))

    # cleanup uploaded files
    try:
        if os.path.isdir(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            log.info("data_dir_cleaned")
    except Exception as e:
        log.warning("data_dir_cleanup_failed", error=str(e))

# ------------------------------------------------------------------
# Ingest
# ------------------------------------------------------------------

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        log.info(
            "ingest_request_received",
            filename=file.filename,
        )

        results = await ingestion_pipeline.ingest_documents()

        return {
            "message": "Ingestion completed",
            "uploaded_file": file.filename,
            "files_processed": len(results),
            "results": [r.model_dump() for r in results],
        }

    except RegulatoryRAGException as e:
        # Domain failure (already logged internally)
        log.error("ingestion_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed",
        )

    except Exception as e:
        # Unexpected API-level failure
        log.error("unexpected_ingestion_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error during ingestion",
        )

# ------------------------------------------------------------------
# Query (RAG)
# ------------------------------------------------------------------

@app.post("/query")
async def query_rag(request: Request):
    payload = {}

    # Flexible body parsing (JSON or form)
    try:
        payload = await request.json()
    except Exception:
        try:
            form = await request.form()
            payload = dict(form)
        except Exception:
            payload = {}

    # Normalize filters
    if isinstance(payload.get("filters"), str):
        try:
            payload["filters"] = json.loads(payload["filters"])
        except Exception:
            pass

    try:
        qreq = QueryRequest.model_validate(payload)
    except Exception as e:
        log.warning("query_validation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    thread_id = qreq.thread_id or f"session-{uuid.uuid4()}"

    log.info(
        "query_received",
        thread_id=thread_id,
        has_filters=bool(qreq.filters),
    )

    try:
        result = await rag_pipeline.run(
            query=qreq.question,
            thread_id=thread_id,
            filters=qreq.filters,
        )

        return {
            "thread_id": thread_id,
            "answer": result.get("answer"),
            "citations": result.get("citations", []),
        }

    except RegulatoryRAGException as e:
        # True system failure
        log.error("rag_query_failed", error=str(e), thread_id=thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query",
        )

    except Exception as e:
        log.error("unexpected_query_error", error=str(e), thread_id=thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error",
        )

# ------------------------------------------------------------------
# Documents
# ------------------------------------------------------------------

@app.get("/documents")
async def get_documents():
    try:
        return await list_documents()
    except Exception as e:
        log.error("list_documents_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.get("/documents/titles")
async def get_document_titles():
    try:
        return await list_document_titles()
    except Exception as e:
        log.error("list_document_titles_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list document titles")


@app.delete("/documents/{title}")
async def delete_document(title: str):
    try:
        await delete_document_by_title(title)
        log.info("document_deleted", title=title)
        return {"message": f"Deleted document '{title}'"}

    except ValueError as e:
        log.warning("document_not_found", title=title)
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        log.error("delete_document_failed", title=title, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete document")
