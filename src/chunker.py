"""
chunker.py
----------
Turns per-page text into overlapping chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter for "intelligent"
chunking (splits on paragraph -> sentence -> word boundaries, in that
order of preference) while keeping a stable mapping back to page
number and document name for source attribution.
"""

import hashlib
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loader import PageText


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    document_name: str
    chunk_index: int  # position within the document, for ordering/debugging


def _make_chunk_id(document_name: str, page_number: int, chunk_index: int, text: str) -> str:
    """Deterministic, unique chunk id: doc + page + index + short text hash.
    Deterministic ids let re-upserts of the same document overwrite the
    same Pinecone vectors instead of duplicating them."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    safe_name = document_name.replace(" ", "_")
    return f"{safe_name}_p{page_number}_c{chunk_index}_{h}"


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    """
    Split page-level text into overlapping chunks.

    chunk_size / chunk_overlap are exposed as parameters so the UI can
    let the user adjust them (mandatory "intermediate enhancement").
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[Chunk] = []
    global_index = 0
    for page in pages:
        pieces = splitter.split_text(page.text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = _make_chunk_id(page.document_name, page.page_number, global_index, piece)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    page_number=page.page_number,
                    document_name=page.document_name,
                    chunk_index=global_index,
                )
            )
            global_index += 1

    return chunks
