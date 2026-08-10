# Technical Report: Intermediate RAG System Using Pinecone

**Course:** Artificial Intelligence / NLP / Applied LLM Systems
**Student Name:**
**Date:**

---

## 1. Introduction
Briefly describe the goal of the assignment and what the system does
(1 paragraph).

## 2. System Architecture
Include the architecture diagram from `docs/architecture_diagram.md`
and walk through the pipeline stage by stage:
PDF Upload → Text Extraction → Chunking → Embedding → Pinecone Indexing
→ Retrieval → LLM Generation → Answer with Source Reference.

## 3. Design Decisions
- Why `pdfplumber` was chosen for extraction over alternatives (e.g. PyPDF2)
- Why `RecursiveCharacterTextSplitter` was used for chunking, and how
  chunk size / overlap were chosen (trade-off: recall vs. context noise)
- Why answers are restricted to retrieved context only, and how the
  system avoids hallucination (system prompt design, fallback message,
  confidence scoring)

## 4. Embedding Model
- **Model used:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Dimension:** 384
- **Why this model:** fast, runs locally/free, good semantic quality
  for short passages, widely used baseline for RAG systems
- Note any alternatives considered (e.g. OpenAI `text-embedding-3-small`)

## 5. Pinecone Configuration
- **Index type:** Serverless (cloud/region as configured in `.env`)
- **Metric:** cosine
- **Namespace strategy:** one namespace per uploaded document, so
  multiple documents can be indexed without collisions and can be
  selectively searched or cleared
- **Metadata stored per vector:** `text`, `page_number`, `document_name`,
  `chunk_index`
- Include a screenshot of your Pinecone console showing the created
  index and a populated namespace

## 6. Retrieval & Generation Strategy
- Top-k retrieval with adjustable `k`
- Adjustable cosine-similarity threshold to filter weak matches
- Optional metadata filter (search within a specific page)
- LLM prompting strategy: system prompt forces context-only answers;
  fallback message `"The answer is not available in the provided document."`
  used when retrieval confidence/context is insufficient

## 7. Intermediate-Level Enhancements Implemented
List which of the mandatory enhancements you implemented (this project
ships with all of the following — pick at least 3 to highlight, or
describe all):
1. Multi-document support (per-document Pinecone namespaces)
2. Query history (session memory, shown in its own tab)
3. Adjustable chunk size / overlap from the sidebar UI
4. Adjustable top-k retrieval from the sidebar UI
5. Metadata filtering by page number
6. Confidence scoring displayed with each answer
7. Logging of user queries to `logs/query_log.csv`

## 8. Challenges Faced
Describe real issues you hit while building/running this (e.g. scanned
PDFs with no extractable text, tuning chunk size, rate limits on the
LLM API, Pinecone index readiness delays) and how you resolved them.

## 9. Performance Analysis
- Time to index a sample N-page PDF
- Average query latency (embedding + Pinecone query + LLM generation)
- Qualitative retrieval accuracy: a few example questions, whether the
  correct page/chunk was retrieved, and the confidence score returned
- Any false "not available" responses observed, and threshold tuning
  done to reduce them

## 10. Conclusion
Summarize what the system demonstrates and possible future improvements
(e.g. re-ranking, hybrid search, OCR for scanned PDFs, streaming answers).

---
*Target length: 3–5 pages.*
