from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil

from utils.db_utils import init_db_pool, close_db_pool
from logger import GLOBAL_LOGGER as log

# Routers
from src.api import (
    health_router,
    query_router,
    ingest_router,
    document_router,
    chunk_router,
)

# -------------------------------------------------
# App initialization
# -------------------------------------------------

app = FastAPI(
    title="Regulatory RAG API",
    version="1.0.0",
)

DATA_DIR = "data"

# Static UI files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/templates")

# -------------------------------------------------
# Register Routers
# -------------------------------------------------

app.include_router(health_router.router)
app.include_router(query_router.router)
app.include_router(ingest_router.router)
app.include_router(document_router.router)
app.include_router(chunk_router.router)

# -------------------------------------------------
# Root UI
# -------------------------------------------------

@app.get("/")
async def index():
    """
    Simple UI landing page
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": {}},
    )

# -------------------------------------------------
# Lifecycle Events
# -------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """
    Initialize database pool on startup
    """
    try:
        await init_db_pool()

        log.info(
            "api_startup_completed",
        )

    except Exception as e:

        log.error(
            "api_startup_failed",
            error=str(e),
        )

        raise


@app.on_event("shutdown")
async def on_shutdown():
    """
    Gracefully close resources
    """

    # Close DB pool
    try:
        await close_db_pool()

        log.info(
            "db_pool_closed",
        )

    except Exception as e:

        log.error(
            "db_pool_close_failed",
            error=str(e),
        )

    # Cleanup uploaded files
    try:
        if os.path.isdir(DATA_DIR):
            shutil.rmtree(DATA_DIR)

            log.info(
                "data_dir_cleaned",
            )

    except Exception as e:

        log.warning(
            "data_dir_cleanup_failed",
            error=str(e),
        )