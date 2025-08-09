import numpy as np
from openai import OpenAI
from typing import List
from src.utils.settings import settings


# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    batch embedding texts using OpenAI's API.
    """
    res = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        encoding_format="float"
    )
    arr = np.array([d.embedding for d in res.data], dtype=np.float32)
    # Normalize the embeddings
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr
