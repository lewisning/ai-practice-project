from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from src.utils.settings import settings

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=5.0,
        )

    def ensure_collection(self):
        collections = {c.name for c in self.client.get_collections().collections}
        if settings.qdrant_collection not in collections:
            self.client.recreate_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=qm.VectorParams(
                    size=settings.embedding_dim,
                    distance=qm.Distance.COSINE,
                ),
            )

    def is_healthy(self) -> bool:
        try:
            _ = self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert(self, ids: List[str], vectors, payloads: List[Dict[str, Any]]):
        """
        Upsert vectors into the Qdrant collection.
        :param ids: List of unique identifiers for the vectors.
        :param vectors: List of vectors to be upserted.
        :param payloads: List of payloads associated with each vector.
        """
        points = qm.Batch(
            ids=ids,
            vectors=vectors.tolist(),
            payloads=payloads
        )
        self.client.upsert(
            collection_name=settings.qdrant_collection,
            points=points
        )

    def search(self, query_vec, top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
        cond = None
        if filters:
            must = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in filters.items()]
            cond = qm.Filter(must=must)

        res = self.client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
            query_filter=cond
        )
        return [
            {
                "id": str(r.id),
                "score": float(r.score),
                "payload": r.payload
            }
            for r in res
        ]

qdrant_store = QdrantStore()
