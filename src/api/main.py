from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional, Dict, Any
import shutil
import os
import uuid

from src.indexing.ingest import DocumentIngestionPipeline
from src.pipelines.rag_pipeline import RAGPipeline
from utils.models import IngestionConfig, QueryRequest

app = FastAPI(
    title="Regulatory RAG API",
    version="1.0.0",
)

# ---------------------------
# Global pipeline instances
# ---------------------------

INGEST_DATA_DIR = "data"

ingestion_pipeline = DocumentIngestionPipeline(
    config=IngestionConfig(),
    documents_folder=INGEST_DATA_DIR,
)

rag_pipeline = RAGPipeline()

# ---------------------------
# Health check
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
):
    os.makedirs(INGEST_DATA_DIR, exist_ok=True)

    file_path = os.path.join(INGEST_DATA_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ingestion
    results = await ingestion_pipeline.ingest_documents()

    return {
        "message": "Ingestion completed",
        "files_processed": len(results),
        "results": [r.model_dump() for r in results],
    }

@app.post("/query")
async def query_rag(request: QueryRequest):
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
