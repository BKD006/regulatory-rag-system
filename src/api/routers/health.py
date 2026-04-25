from fastapi import APIRouter
from utils import db_utils
import boto3
import os

router = APIRouter(
    prefix="",
    tags=["Health"],
)


@router.get("/health")
async def health():
    """
    Performs a production health check for the service.

    Verifies the availability and connectivity of key system components:
    - API service
    - PostgreSQL database connection
    - pgvector extension
    - Bedrock embedding model

    Returns:
        Dict[str, str]: Health status of each component with values:
            - "ok": Component is functioning correctly.
            - "failed": Component check failed.
            - "unknown": Component not yet evaluated.
    """

    status = {
        "api": "ok",
        "database": "unknown",
        "vector_db": "unknown",
        "embedding_model": "unknown",
    }

    # -------------------------
    # Database check
    # -------------------------
    try:
        async with db_utils.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        status["database"] = "ok"
    except Exception:
        status["database"] = "failed"

    # -------------------------
    # pgvector check
    # -------------------------
    try:
        async with db_utils.db_pool.acquire() as conn:
            await conn.fetchval(
                "SELECT 1 FROM pg_extension WHERE extname='vector'"
            )
        status["vector_db"] = "ok"
    except Exception:
        status["vector_db"] = "failed"

    # -------------------------
    # Bedrock model check
    # -------------------------
    try:
        bedrock = boto3.client("bedrock-runtime")

        bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body='{"inputText": "health check"}',
            contentType="application/json",
            accept="application/json",
        )

        status["embedding_model"] = "ok"
    except Exception:
        status["embedding_model"] = "failed"

    return status