from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog
import uuid
import numpy as np

from prometheus_fastapi_instrumentator import Instrumentator

from src.utils.settings import settings
from src.rag.qdrant_store import qdrant_store
from src.rag.embedding import embed_texts
from src.api.schemas import IngestItem, SearchQuery

logger = structlog.get_logger()

app = FastAPI(title=settings.app_name, version="0.3.0")
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant_store.ensure_collection()
    logger.info("startup_done", qdrant_host=settings.qdrant_host, qdrant_port=settings.qdrant_port)
    yield
    logger.info("shutdown_done")

app.router.lifespan_context = lifespan

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/qdrant/health")
def qdrant_health():
    ok = qdrant_store.is_healthy()
    return {"connected": ok, "host": settings.qdrant_host, "port": settings.qdrant_port, "collection": settings.qdrant_collection}

@app.post("/ingest")
def ingest(item: IngestItem):
    # Prepare payload
    pid = item.id or str(uuid.uuid4())
    payload = item.model_dump()
    payload["id"] = pid # Ensure id is included in payload

    # Generate embedding
    vec = embed_texts([item.text])

    # Upsert to Qdrant
    qdrant_store.upsert(ids=[pid], vectors=vec, payloads=[payload])
    return {"ok": True, "id": pid}

@app.post("/search")
def search(q: SearchQuery):
    vec = embed_texts([q.query])
    filters = {}
    if q.product: filters["product"] = q.product
    if q.lang: filters["lang"] = q.lang
    hits = qdrant_store.search(vec[0], top_k=q.top_k, filters=filters or None)
    return {"query": q.query, "hits": hits}
