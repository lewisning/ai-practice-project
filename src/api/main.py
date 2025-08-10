from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog
import uuid
from fastapi import UploadFile, File
from pathlib import Path

from prometheus_fastapi_instrumentator import Instrumentator

from src.utils.settings import settings
from src.rag.qdrant_store import qdrant_store
from src.rag.embedding import embed_texts
from src.rag.bm25_store import bm25_store
from src.rag.merged_retriever import search_merged
from src.api.schemas import TicketRequest, TicketResponse, IngestPathRequest, IngestItem, SearchQuery
from src.core.orchestrator import resolve_ticket
from src.rag.file_ingest import ingest_file, ingest_folder

logger = structlog.get_logger()

app = FastAPI(title=settings.app_name, version="0.3.0")
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant_store.ensure_collection()
    pairs = [(row["id"], row["text"]) for row in qdrant_store.scroll_all_texts()]
    bm25_store.build(pairs)
    logger.info("bm25_built", count=len(pairs))
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

    # Upsert to Qdrant and add to BM25 store
    qdrant_store.upsert(ids=[pid], vectors=vec, payloads=[payload])
    bm25_store.add(pid, item.text)

    return {"ok": True, "id": pid}

@app.post("/search")
def search(q: SearchQuery):
    vec = embed_texts([q.query])
    filters = {}
    if q.product: filters["product"] = q.product
    if q.lang: filters["lang"] = q.lang
    hits = qdrant_store.search(vec[0], top_k=q.top_k, filters=filters or None)
    return {"query": q.query, "hits": hits}

@app.post("/search_merged")
def search_v2(q: SearchQuery):
    filters = {}
    if q.product: filters["product"] = q.product
    if q.lang: filters["lang"] = q.lang
    hits = search_merged(q.query, top_k=q.top_k, filters=filters or None, alpha=0.7)
    return {"query": q.query, "hits": hits}

@app.post("/ingest-file")
async def ingest_file_api(file: UploadFile = File(...), product: str = "domains", lang: str = "en"):
    tmp = Path("/tmp") / file.filename
    content = await file.read()
    tmp.write_bytes(content)
    try:
        res = ingest_file(tmp, product=product, lang=lang)
        return res
    finally:
        try: tmp.unlink()
        except Exception: pass

@app.post("/ingest-path")
def ingest_path_api(req: IngestPathRequest):
    p = Path(req.path)
    if not p.exists():
        return {"ok": False, "error": f"path not found: {p}"}
    if p.is_file():
        return ingest_file(p, product=req.product or "domains", lang=req.lang or "en")
    else:
        return {"ok": True, "results": ingest_folder(p, product=req.product or "domains", lang=req.lang or "en")}

@app.post("/resolve-ticket", response_model=TicketResponse)
def resolve_ticket_api(req: TicketRequest):
    result = resolve_ticket(req.ticket_text, top_k=req.top_k)
    return result
