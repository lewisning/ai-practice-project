import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

_token = re.compile(r"[a-z0-9]+", re.I)

def _tok(s: str) -> List[str]:
    return _token.findall(s.lower())

class BM25Store:
    """
    A simple BM25-based text search store.
    This class allows building a BM25 index from a list of (id, text) tuples,
    adding new items, and searching for the top-k relevant documents based on a query.
    """
    def __init__(self):
        self.ids: List[str] = []
        self.docs_tokens: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, items: List[Tuple[str, str]]):
        # items: [(id, text), ...]
        self.ids = [i for i, _ in items]
        self.docs_tokens = [_tok(t) for _, t in items]
        if self.docs_tokens:
            self._bm25 = BM25Okapi(self.docs_tokens)

    def add(self, _id: str, text: str):
        self.ids.append(_id)
        self.docs_tokens.append(_tok(text))
        # Rebuild BM25 index, efficient for small datasets
        self._bm25 = BM25Okapi(self.docs_tokens)

    def search(self, query: str, top_k: int = 10):
        if not self._bm25 or not self.ids:
            return []
        q = _tok(query)
        scores = self._bm25.get_scores(q)
        # Get top-k indices based on scores
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"id": self.ids[i], "bm25": float(scores[i])} for i in idxs]

bm25_store = BM25Store()
