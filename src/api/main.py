from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import shutil
import uuid

from src.indexing.ingest import DocumentIngestionPipeline
from src.pipelines.rag_pipeline import RAGPipeline
from utils.models import IngestionConfig, QueryRequest
import json
from fastapi import status
from utils.db_utils import (
    list_documents,
    list_document_titles,
    delete_document_by_title,
    init_db_pool,
    close_db_pool,
)

# -------------------------------------------------
# App init
# -------------------------------------------------

app = FastAPI(
    title="Regulatory RAG API",
    version="1.0.0",
)

DATA_DIR = "data"

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# -------------------------------------------------
# Pipelines (singletons)
# -------------------------------------------------

ingestion_pipeline = DocumentIngestionPipeline(
    config=IngestionConfig(),
    documents_folder=DATA_DIR,
)

rag_pipeline = RAGPipeline()

# -------------------------------------------------
# Health
# -------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index(request: Request):
    """Serve the HTMX single-page frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.on_event("startup")
async def on_startup():
    # initialize DB connection pool used by the async utils
    await init_db_pool()


@app.on_event("shutdown")
async def on_shutdown():
    # cleanly close DB pool
    try:
        await close_db_pool()
    except Exception:
        pass
    # remove temporary data folder created for ingested files
    try:
        # only remove the DATA_DIR inside the project to avoid accidental deletions
        if os.path.isdir(DATA_DIR):
            shutil.rmtree(DATA_DIR)
    except Exception:
        pass


# -------------------------------------------------
# Ingest
# -------------------------------------------------

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = await ingestion_pipeline.ingest_documents()

    return {
        "message": "Ingestion completed",
        "uploaded_file": file.filename,
        "files_processed": len(results),
        "results": [r.model_dump() for r in results],
    }


# -------------------------------------------------
# Query (RAG)
# -------------------------------------------------

@app.post("/query")
async def query_rag(request: Request):
    """Accept JSON or form POSTs, validate against QueryRequest, and run the RAG pipeline.

    This is more forgiving than FastAPI's automatic body parsing and will return
    clearer validation errors when the payload shape is incorrect.
    """
    payload = {}
    # try JSON body first
    try:
        payload = await request.json()
    except Exception:
        # try form data
        try:
            form = await request.form()
            payload = dict(form)
        except Exception:
            payload = {}

    # normalize filters if passed as a JSON string
    if isinstance(payload.get("filters"), str):
        try:
            payload["filters"] = json.loads(payload["filters"]) if payload.get("filters") else None
        except Exception:
            # leave as string; validation will catch if invalid
            pass

    try:
        qreq = QueryRequest.model_validate(payload)
    except Exception as e:
        # return structured 422 with validation detail
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    thread_id = qreq.thread_id or f"session-{uuid.uuid4()}"

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


# -------------------------------------------------
# Documents
# -------------------------------------------------

@app.get("/documents")
async def get_documents():
    return await list_documents()


@app.get("/documents/titles")
async def get_document_titles():
    return await list_document_titles()


@app.delete("/documents/{title}")
async def delete_document(title: str):
    try:
        await delete_document_by_title(title)
        return {"message": f"Deleted document '{title}'"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
