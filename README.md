# Intermediate RAG System Using Pinecone Vector Database

A Retrieval-Augmented Generation system that answers questions strictly
from uploaded PDF content, with page-level source attribution and
hallucination-prevention built into the prompting strategy.

```
PDF Upload → Text Extraction → Text Chunking → Embedding Generation →
Pinecone Vector Indexing → Semantic Retrieval → LLM Response Generation →
Answer with Source Reference
```

## 1. Project Structure

```
rag-pinecone-project/
├── app.py                     # Streamlit UI — orchestrates the pipeline
├── config.py                  # Environment variable loading & validation
├── requirements.txt
├── .env.example                # Copy to .env and fill in your keys
├── src/
│   ├── loader.py               # PDF → cleaned, page-tagged text
│   ├── chunker.py               # Text → overlapping chunks
│   ├── embedder.py              # Text → embedding vectors (SentenceTransformers)
│   ├── vector_store.py          # All Pinecone logic (index/namespace/upsert/query)
│   ├── retriever.py             # Query embedding + retrieval orchestration
│   ├── generator.py             # Strict-context LLM answer generation
│   └── utils.py                 # Query logging, formatting helpers
├── docs/
│   ├── architecture_diagram.md  # Mermaid diagram + module responsibility map
│   └── technical_report_template.md
├── data/                        # Put sample PDFs here for local testing
└── logs/
    └── query_log.csv            # Created automatically once you run queries
```

## 2. Setup

### 2.1 Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2.2 Install dependencies
```bash
pip install -r requirements.txt
```

### 2.3 Configure environment variables
```bash
cp .env.example .env
```
Then edit `.env`:
- `PINECONE_API_KEY` — from the [Pinecone console](https://app.pinecone.io/)
- `GROQ_API_KEY` — free tier available at [console.groq.com](https://console.groq.com/) (recommended, fast)
  — or set `LLM_PROVIDER=openai` and fill in `OPENAI_API_KEY` instead

The Pinecone index is created automatically on first run if it doesn't
already exist (serverless, cosine metric, dimension 384 to match the
default embedding model).

### 2.4 Run the app
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`).

## 3. Using the app
1. **Upload & Index tab** — upload one or more PDFs (≤20MB each), click
   "Process & Index Document(s)". Each document gets its own Pinecone
   namespace so documents never mix.
2. **Ask Questions tab** — pick which documents to search (sidebar),
   optionally filter to a specific page, then ask a question. The
   answer is generated strictly from retrieved context; if nothing
   relevant is found you'll see *"The answer is not available in the
   provided document."*
3. **Query History tab** — every question asked this session, with its
   confidence score and grounding status. All queries are also logged
   to `logs/query_log.csv`.

## 4. Key design choices (see `docs/technical_report_template.md` for the full report)
- **Embedding model:** `all-MiniLM-L6-v2` (384-dim, fast, free, local)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`, adjustable
  chunk size/overlap from the UI
- **Vector DB:** Pinecone serverless index, cosine similarity, one
  namespace per document, metadata = `text`, `page_number`,
  `document_name`, `chunk_index`
- **Hallucination prevention:** strict system prompt restricting the
  LLM to provided context only, explicit fallback message, similarity
  threshold filtering weak matches before they ever reach the LLM
- **Source attribution:** every answer shows the originating page
  number, an excerpt, and a cosine similarity score

## 5. Troubleshooting
| Symptom | Likely cause |
|---|---|
| `Missing required environment variable: PINECONE_API_KEY` | `.env` not created/filled — see step 2.3 |
| `Could not reach Pinecone` | Bad API key, or no network access |
| `'file.pdf' produced no extractable text` | Scanned/image-only PDF — needs OCR (not included by default) |
| Answers always say "not available" | Lower the similarity threshold in the sidebar, or increase top-k |
