"""
vector_store.py
----------------
All Pinecone-specific logic lives here: index creation, namespace
usage, upserting vectors, querying, and metadata management. Nothing
outside this file should import the `pinecone` package directly.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from pinecone import Pinecone, ServerlessSpec

from src.chunker import Chunk


class PineconeConnectionError(Exception):
    pass


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    page_number: int
    document_name: str
    score: float  # cosine similarity, 0..1


class VectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        try:
            self.pc = Pinecone(api_key=api_key)
        except Exception as exc:
            raise PineconeConnectionError(f"Could not initialise Pinecone client: {exc}") from exc

        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    # ---------- Index lifecycle ----------

    def _ensure_index(self) -> None:
        """Create the index if it doesn't already exist (idempotent)."""
        try:
            existing = [i["name"] for i in self.pc.list_indexes()]
        except Exception as exc:
            raise PineconeConnectionError(
                f"Could not reach Pinecone (check API key / network): {exc}"
            ) from exc

        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
            # Wait until the index reports ready before using it
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    # ---------- Upsert ----------

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        namespace: str,
        batch_size: int = 100,
    ) -> int:
        """Upsert chunk vectors + metadata into a namespace. Namespaces
        let us keep each uploaded document (or each user session)
        logically separate inside a single Pinecone index."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        records = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                {
                    "id": chunk.chunk_id,
                    "values": vector,
                    "metadata": {
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "document_name": chunk.document_name,
                        "chunk_index": chunk.chunk_index,
                    },
                }
            )

        total_upserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)

        return total_upserted

    # ---------- Query ----------

    def query(
        self,
        query_vector: List[float],
        namespace: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Query top-k most similar chunks, optionally filtered by
        metadata (e.g. {"page_number": 3}) and a minimum similarity
        threshold (cosine)."""
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=metadata_filter,
        )

        results: List[RetrievedChunk] = []
        for match in response.get("matches", []):
            score = match.get("score", 0.0)
            if score < similarity_threshold:
                continue
            meta = match.get("metadata", {})
            results.append(
                RetrievedChunk(
                    chunk_id=match["id"],
                    text=meta.get("text", ""),
                    page_number=meta.get("page_number", -1),
                    document_name=meta.get("document_name", "unknown"),
                    score=score,
                )
            )
        return results

    # ---------- Management ----------

    def delete_namespace(self, namespace: str) -> None:
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception:
            # Deleting a namespace that has no vectors yet raises in some
            # Pinecone client versions — safe to ignore.
            pass

    def list_namespaces(self) -> List[str]:
        stats = self.index.describe_index_stats()
        return list(stats.get("namespaces", {}).keys())
