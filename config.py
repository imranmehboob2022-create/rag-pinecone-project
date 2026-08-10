"""
config.py
---------
Centralised configuration and environment-variable handling.
Every other module imports settings from here instead of calling
os.getenv() directly. This keeps env handling in one place and makes
the rest of the codebase easy to test / mock.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # loads variables from a local .env file if present


def _get_env(name: str, default: str = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str

    # Embeddings
    embedding_model: str
    embedding_dimension: int

    # LLM
    llm_provider: str
    groq_api_key: str
    groq_model: str
    openai_api_key: str
    openai_model: str

    # App-level defaults (overridable from the Streamlit UI)
    default_chunk_size: int = 800
    default_chunk_overlap: int = 120
    default_top_k: int = 5
    default_similarity_threshold: float = 0.35
    max_pdf_size_mb: int = 20


def load_settings() -> Settings:
    """Load and validate all settings. Raises a clear error if something
    required is missing, rather than failing deep inside the pipeline."""
    return Settings(
        pinecone_api_key=_get_env("PINECONE_API_KEY", required=True),
        pinecone_index_name=_get_env("PINECONE_INDEX_NAME", "rag-intermediate-index"),
        pinecone_cloud=_get_env("PINECONE_CLOUD", "aws"),
        pinecone_region=_get_env("PINECONE_REGION", "us-east-1"),
        embedding_model=_get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dimension=int(_get_env("EMBEDDING_DIMENSION", "384")),
        llm_provider=_get_env("LLM_PROVIDER", "groq").lower(),
        groq_api_key=_get_env("GROQ_API_KEY", ""),
        groq_model=_get_env("GROQ_MODEL", "llama-3.1-8b-instant"),
        openai_api_key=_get_env("OPENAI_API_KEY", ""),
        openai_model=_get_env("OPENAI_MODEL", "gpt-4o-mini"),
    )
