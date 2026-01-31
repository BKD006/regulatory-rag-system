"""
FastAPI entrypoint for Regulatory RAG system.

Endpoints:
- GET  /            → UI (Jinja, ingest + query)
- POST /query-ui    → UI-based RAG (same page)
- POST /ingest-ui   → UI-based ingestion (same page)
- POST /query       → API-based RAG (JSON)
- POST /ingest      → API-based ingestion (JSON)
- GET  /health      → Health check
"""

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Request,
    Form,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import shutil
import uuid
import json

from src.indexing.ingest import DocumentIngestionPipeline
from src.pipelines.rag_pipeline import RAGPipeline
from utils.models import IngestionConfig, QueryRequest

# -------------------------------------------------
# App initialization
# -------------------------------------------------

app = FastAPI(
    title="Regulatory RAG API",
    version="1.0.0",
)

# -------------------------------------------------
# Static + Templates
# -------------------------------------------------

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# -------------------------------------------------
# Pipelines
# -------------------------------------------------

INGEST_DATA_DIR = "data"

ingestion_pipeline = DocumentIngestionPipeline(
    config=IngestionConfig(),
    documents_folder=INGEST_DATA_DIR,
)

rag_pipeline = RAGPipeline()

# -------------------------------------------------
# UI (Jinja, no JS)
# -------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post("/ingest-ui", response_class=HTMLResponse)
async def ingest_ui(
    request: Request,
    file: UploadFile = File(...),
):
    os.makedirs(INGEST_DATA_DIR, exist_ok=True)
    file_path = os.path.join(INGEST_DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = await ingestion_pipeline.ingest_documents()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "ingest_message": f"✅ Successfully ingested {file.filename}",
            "ingest_results": results,
        },
    )


@app.post("/query-ui", response_class=HTMLResponse)
async def query_ui(
    request: Request,
    question: str = Form(...),
    thread_id: str = Form(""),
    filters: str = Form(""),
):
    thread_id = thread_id or f"session-{uuid.uuid4()}"

    try:
        filters_dict = json.loads(filters) if filters else {}
    except Exception:
        filters_dict = {}

    result = await rag_pipeline.run(
        query=question,
        thread_id=thread_id,
        filters=filters_dict,
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": question,
            "answer": result["answer"],
            "citations": result["citations"],
            "thread_id": thread_id,
            "filters": filters,
        },
    )

# -------------------------------------------------
# JSON APIs (optional, kept clean)
# -------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_api(file: UploadFile = File(...)):
    os.makedirs(INGEST_DATA_DIR, exist_ok=True)
    file_path = os.path.join(INGEST_DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = await ingestion_pipeline.ingest_documents()

    return {
        "message": "Ingestion completed",
        "uploaded_file": file.filename,
        "results": [r.model_dump() for r in results],
    }


@app.post("/query")
async def query_api(request: QueryRequest):
    thread_id = request.thread_id or f"session-{uuid.uuid4()}"

    result = await rag_pipeline.run(
        query=request.question,
        thread_id=thread_id,
        filters=request.filters,
    )

    return {
        "thread_id": thread_id,
        "answer": result["answer"],
        "citations": result["citations"],
    }
