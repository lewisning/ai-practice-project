from typing import Dict, Any, List, Optional
import numpy as np

from src.rag.embedding import embed_texts
from src.rag.qdrant_store import qdrant_store
from src.rag.bm25_store import bm25_store

def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def search_merged(query: str, top_k: int = 8, filters: Optional[Dict[str, Any]] = None,
                  alpha: float = 0.7) -> List[Dict[str, Any]]:
    """
    alpha: semantic weight; (1-alpha) is BM25 weight
    """
    # 1. Calculate query vector
    q_vec = embed_texts([query])[0]

    # 2. Qdrant semantic recall
    sem_hits = qdrant_store.search(q_vec, top_k=top_k * 4, filters=filters)

    # 3. BM25 keyword recall
    bm25_hits = bm25_store.search(query, top_k=top_k * 4)

    # 4. Prepare combined candidates
    cand: Dict[str, Dict[str, Any]] = {}
    for h in sem_hits:
        pid = h["id"]
        cand.setdefault(pid, {}).update({"id": pid, "semantic": float(h["score"]), "payload": h.get("payload")})

    for h in bm25_hits:
        pid = h["id"]
        cand.setdefault(pid, {}).update({"id": pid, "bm25": float(h["bm25"])})

    # 5. Get vectors for BM25-only candidates
    only_bm25_ids = [pid for pid, v in cand.items() if "semantic" not in v]
    if only_bm25_ids:
        vecs = qdrant_store.get_vectors_by_ids(only_bm25_ids)
        for pid in only_bm25_ids:
            vec = vecs.get(pid)
            if vec is not None:
                # calculate semantic score for BM25-only candidates
                sem = float(np.dot(q_vec, np.array(vec, dtype=np.float32)))
                cand[pid]["semantic"] = sem
            else:
                cand[pid]["semantic"] = 0.0

    # 6. Normalize semantic and BM25 scores, then merge
    sem_scores = [v.get("semantic", 0.0) for v in cand.values()]
    bm25_scores = [v.get("bm25", 0.0) for v in cand.values()]
    sem_norm = _minmax_norm(sem_scores)
    bm25_norm = _minmax_norm(bm25_scores)

    for (pid, v), s, b in zip(cand.items(), sem_norm, bm25_norm):
        v["score_merged"] = alpha * s + (1 - alpha) * b

    # 7. Sort candidates by merged score and return top_k
    merged = sorted(cand.values(), key=lambda x: x["score_merged"], reverse=True)[:top_k]

    return merged
