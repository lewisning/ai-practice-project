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
        """
        Ensure that the Qdrant collection exists. If it does not exist, create it.
        """
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

qdrant_store = QdrantStore()
