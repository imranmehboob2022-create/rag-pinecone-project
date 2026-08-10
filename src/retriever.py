"""
retriever.py
------------
Thin orchestration layer: embed the user's query, then ask the
VectorStore for the top-k most similar chunks. Kept separate from
vector_store.py so retrieval-specific logic (thresholding, filters)
doesn't get tangled with raw Pinecone calls.
"""

from typing import List, Dict, Any, Optional

from src.embedder import Embedder
from src.vector_store import VectorStore, RetrievedChunk


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int = 5,
        similarity_threshold: float = 0.35,
        page_filter: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector = self.embedder.embed_query(query)

        metadata_filter: Optional[Dict[str, Any]] = None
        if page_filter is not None:
            metadata_filter = {"page_number": {"$eq": page_filter}}

        results = self.vector_store.query(
            query_vector=query_vector,
            namespace=namespace,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
        )
        # Highest similarity first
        results.sort(key=lambda r: r.score, reverse=True)
        return results
