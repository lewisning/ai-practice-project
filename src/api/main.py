from fastapi import FastAPI
import structlog
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

from src.utils.settings import settings
from src.rag.qdrant_store import qdrant_store

logger = structlog.get_logger()

app = FastAPI(title=settings.app_name, version="0.2.0")
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the FastAPI application.
    :param app: FastAPI application instance
    """
    qdrant_store.ensure_collection()
    logger.info(
        "startup_done",
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port
    )
    yield
    logger.info("shutdown_done")

app.router.lifespan_context = lifespan

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/qdrant/health")
def qdrant_health():
    ok = qdrant_store.is_healthy()
    return {
        "connected": ok,
        "host": settings.qdrant_host,
        "port": settings.qdrant_port,
        "collection": settings.qdrant_collection,
    }

@app.get("/")
def root():
    return {"service": "tucows-rag-assistant", "message": "Hello, Qdrant"}
