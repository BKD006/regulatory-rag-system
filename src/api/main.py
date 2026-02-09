from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import uuid

from src.indexing.ingest import DocumentIngestionPipeline
from src.pipelines.rag_pipeline import RAGPipeline
from utils.models import IngestionConfig, QueryRequest
from utils.db_utils_async import (
    list_documents,
    list_document_titles,
    delete_document_by_title,
)

# -------------------------------------------------
# App init
# -------------------------------------------------

app = FastAPI(
    title="Regulatory RAG API",
    version="1.0.0",
)

DATA_DIR = "data"

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
