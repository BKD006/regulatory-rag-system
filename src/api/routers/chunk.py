from fastapi import APIRouter, HTTPException

from src.infra.db.db_utils import get_chunk_by_id
from logger import GLOBAL_LOGGER as log

router = APIRouter(
    prefix="",
    tags=["Chunks"],
)


# -------------------------------------------------
# Retrieve chunk by ID
# -------------------------------------------------
@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """
    Retrieves a document chunk by its unique identifier.

    Args:
        chunk_id (str): Unique identifier of the chunk.

    Returns:
        Dict[str, Any]: Dictionary containing chunk details including:
            - chunk_id (str)
            - document_id (str)
            - content (str)
            - metadata (dict)
            - token_count (int)

    Raises:
        HTTPException:
            - 400: If the chunk_id format is invalid.
            - 404: If the chunk is not found.
            - 500: If retrieval fails due to server error.
    """
    try:
        row = await get_chunk_by_id(chunk_id)

        if not row:
            log.warning(
                "chunk_not_found",
                chunk_id=chunk_id,
            )

            raise HTTPException(
                status_code=404,
                detail="Chunk not found",
            )

        return {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "content": row["content"],
            "metadata": row["metadata"],
            "token_count": row["token_count"],
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid chunk_id format",
        )

    except Exception as e:

        log.error(
            "get_chunk_failed",
            chunk_id=chunk_id,
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chunk",
        )