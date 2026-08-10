# System Architecture Diagram

Paste the Mermaid block below into [mermaid.live](https://mermaid.live) or a
Markdown viewer that supports Mermaid (GitHub does) to render it, or export
it as a PNG for your submission.

```mermaid
flowchart TD
    A[User uploads PDF via Streamlit UI] --> B[loader.py: Text Extraction<br/>pdfplumber, page-by-page]
    B --> C[chunker.py: Intelligent Chunking<br/>RecursiveCharacterTextSplitter]
    C --> D[embedder.py: Embedding Generation<br/>SentenceTransformer all-MiniLM-L6-v2]
    D --> E[vector_store.py: Pinecone Upsert<br/>namespace = per document<br/>metadata: page, doc name, chunk id]
    E --> F[(Pinecone Vector Index)]

    G[User types a question] --> H[embedder.py: Embed Query]
    H --> I[retriever.py: Semantic Search<br/>top-k, cosine similarity, threshold, metadata filter]
    F --> I
    I --> J[generator.py: LLM Answer Generation<br/>Groq / OpenAI, strict-context prompt]
    J --> K{Context sufficient?}
    K -- Yes --> L[Answer + Source References<br/>page number, excerpt, similarity score]
    K -- No --> M["'Answer not available in document'"]
    L --> N[Streamlit UI: Answer tab + Query History]
    M --> N
```

## Module responsibility map

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI orchestration only — no business logic |
| `config.py` | Environment variable loading & validation |
| `src/loader.py` | PDF → cleaned, page-tagged text |
| `src/chunker.py` | Page text → overlapping chunks with stable IDs |
| `src/embedder.py` | Text → embedding vectors (cached model) |
| `src/vector_store.py` | All Pinecone calls: index creation, upsert, query, namespace/metadata mgmt |
| `src/retriever.py` | Query embedding + Pinecone query orchestration, thresholding |
| `src/generator.py` | Strict-context LLM prompting, hallucination guardrail, confidence scoring |
| `src/utils.py` | Query logging (CSV), text truncation, namespace sanitising |
