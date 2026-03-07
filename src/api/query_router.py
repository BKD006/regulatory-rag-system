from fastapi import APIRouter, HTTPException, Request, status
import json

from src.pipelines.rag_pipeline import RAGPipeline
from utils.models import QueryRequest
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException

router = APIRouter(
    prefix="",
    tags=["RAG"],
)

# Singleton pipeline
rag_pipeline = RAGPipeline()


@router.post("/query")
async def query_rag(request: Request):
    """
    RAG query endpoint.

    Accepts:
    - question
    - optional filters

    Returns:
    - generated answer
    - citations
    """

    payload = {}

    # ------------------------------
    # Flexible body parsing
    # ------------------------------
    try:
        payload = await request.json()
    except Exception:
        try:
            form = await request.form()
            payload = dict(form)
        except Exception:
            payload = {}

    # ------------------------------
    # Normalize filters
    # ------------------------------
    if isinstance(payload.get("filters"), str):
        try:
            payload["filters"] = json.loads(payload["filters"])
        except Exception:
            pass

    # ------------------------------
    # Validate request
    # ------------------------------
    try:
        qreq = QueryRequest.model_validate(payload)
    except Exception as e:
        log.warning("query_validation_failed", error=str(e))

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    log.info(
        "query_received",
        has_filters=bool(qreq.filters),
    )
    # Require document selection
    if not qreq.filters or "title" not in qreq.filters:
        return {
            "answer": "Please select a document before asking a question.",
            "citations": []
        }

    # ------------------------------
    # Run RAG pipeline
    # ------------------------------
    try:
        result = await rag_pipeline.run(
            query=qreq.question,
            filters=qreq.filters,
        )

        return {
            "answer": result.get("answer"),
            "citations": result.get("citations", []),
        }

    except RegulatoryRAGException as e:

        log.error(
            "rag_query_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query",
        )

    except Exception as e:

        log.error(
            "unexpected_query_error",
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error",
        )