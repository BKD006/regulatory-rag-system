from fastapi import APIRouter, HTTPException

from src.infra.db.db_utils import (
    list_documents,
    list_document_titles,
    delete_document_by_title,
)

from logger import GLOBAL_LOGGER as log

router = APIRouter(
    prefix="",
    tags=["Documents"],
)


# -------------------------------------------------
# List all documents
# -------------------------------------------------
@router.get("/documents")
async def get_documents():
    """
    Retrieves all documents stored in the system.

    Returns:
        List[Dict[str, Any]]: List of documents with metadata.

    Raises:
        HTTPException:
            - 500: If document retrieval fails.
    """
    try:
        return await list_documents()

    except Exception as e:
        log.error(
            "list_documents_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to list documents",
        )


# -------------------------------------------------
# List document titles
# -------------------------------------------------
@router.get("/documents/titles")
async def get_document_titles():
    """
    Retrieves a list of document titles.

    Useful for UI filters and selection.

    Returns:
        List[str]: List of document titles.

    Raises:
        HTTPException:
            - 500: If retrieval fails.
    """
    try:
        return await list_document_titles()

    except Exception as e:
        log.error(
            "list_document_titles_failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to list document titles",
        )


# -------------------------------------------------
# Delete document
# -------------------------------------------------
@router.delete("/documents/{title}")
async def delete_document(title: str):
    """
    Deletes a document and all its associated chunks.

    Args:
        title (str): Title of the document to delete.

    Returns:
        Dict[str, str]: Confirmation message indicating successful deletion.

    Raises:
        HTTPException:
            - 404: If the document is not found.
            - 500: If deletion fails due to server error.
    """
    try:
        await delete_document_by_title(title)

        log.info(
            "document_deleted",
            title=title,
        )

        return {
            "message": f"Deleted document '{title}'"
        }

    except ValueError as e:

        log.warning(
            "document_not_found",
            title=title,
        )

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except Exception as e:

        log.error(
            "delete_document_failed",
            title=title,
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to delete document",
        )