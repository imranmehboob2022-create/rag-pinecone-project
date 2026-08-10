"""
embedder.py
-----------
Wraps a SentenceTransformer model. Keeps embedding logic isolated so
the model could be swapped (e.g. for an OpenAI embedding endpoint)
without touching chunking / retrieval code.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed a list of strings. Returns plain python lists
        (Pinecone's client expects list[float], not numpy arrays)."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # so dot-product == cosine similarity
        )
        return np.asarray(vectors).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> Embedder:
    """Cache the loaded model process-wide — loading a SentenceTransformer
    is expensive and Streamlit reruns this module on every interaction."""
    return Embedder(model_name)
