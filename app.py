"""
app.py
------
Streamlit front-end wiring the full pipeline together:

PDF Upload -> Text Extraction -> Chunking -> Embeddings ->
Pinecone Indexing -> Semantic Retrieval -> LLM Generation ->
Answer with Source Reference

Run with:  streamlit run app.py
"""

import streamlit as st

from config import load_settings
from src.loader import extract_text_by_page, InvalidPDFError, PDFTooLargeError
from src.chunker import chunk_pages
from src.embedder import get_embedder
from src.vector_store import VectorStore, PineconeConnectionError
from src.retriever import Retriever
from src.generator import AnswerGenerator, NOT_FOUND_MESSAGE
from src.utils import log_query, truncate, safe_namespace

st.set_page_config(page_title="RAG over PDFs (Pinecone)", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------
# Settings & cached resources
# ----------------------------------------------------------------------
try:
    settings = load_settings()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    return VectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )


@st.cache_resource(show_spinner=False)
def get_generator() -> AnswerGenerator:
    return AnswerGenerator(
        provider=settings.llm_provider,
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = {}   # {display_name: namespace}
if "query_history" not in st.session_state:
    st.session_state.query_history = []  # list of dicts: question/answer/etc.

# ----------------------------------------------------------------------
# Sidebar: settings the user can adjust (mandatory enhancements)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Retrieval Settings")

    chunk_size = st.slider("Chunk size (characters)", 300, 2000, settings.default_chunk_size, step=50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, settings.default_chunk_overlap, step=20)
    top_k = st.slider("Top-K chunks to retrieve", 1, 15, settings.default_top_k)
    similarity_threshold = st.slider(
        "Similarity threshold (cosine)", 0.0, 1.0, settings.default_similarity_threshold, step=0.05
    )

    st.divider()
    st.header("📚 Indexed Documents")
    if st.session_state.indexed_docs:
        selected_docs = st.multiselect(
            "Search across:",
            options=list(st.session_state.indexed_docs.keys()),
            default=list(st.session_state.indexed_docs.keys()),
        )
        page_filter_enabled = st.checkbox("Filter by specific page number")
        page_filter_value = None
        if page_filter_enabled:
            page_filter_value = st.number_input("Page number", min_value=1, step=1, value=1)
    else:
        selected_docs = []
        page_filter_value = None
        st.caption("No documents indexed yet — upload one to get started.")

    st.divider()
    if st.button("🗑️ Clear all indexed documents"):
        vs = get_vector_store()
        for ns in st.session_state.indexed_docs.values():
            vs.delete_namespace(ns)
        st.session_state.indexed_docs = {}
        st.session_state.query_history = []
        st.success("Cleared all documents from this session.")
        st.rerun()

# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
st.title("📄 Intermediate RAG System — Pinecone Vector Database")
st.caption(
    "Upload one or more PDFs, then ask questions. Answers are generated "
    "strictly from the document content, with page-level source attribution."
)

tab_upload, tab_ask, tab_history = st.tabs(["1️⃣ Upload & Index", "2️⃣ Ask Questions", "3️⃣ Query History"])

# ---------------- Tab 1: Upload ----------------
with tab_upload:
    st.subheader("Upload PDF document(s)")
    uploaded_files = st.file_uploader(
        "Supports multiple PDFs, up to 20 MB each",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Document(s)"):
            embedder = get_embedder(settings.embedding_model)
            try:
                vector_store = get_vector_store()
            except PineconeConnectionError as e:
                st.error(f"Pinecone connection failed: {e}")
                st.stop()

            for f in uploaded_files:
                with st.status(f"Processing '{f.name}'...", expanded=True) as status:
                    file_bytes = f.read()

                    try:
                        st.write("Extracting text from PDF...")
                        pages = extract_text_by_page(
                            file_bytes, f.name, max_size_mb=settings.max_pdf_size_mb
                        )
                    except (InvalidPDFError, PDFTooLargeError) as e:
                        status.update(label=f"Failed: {f.name}", state="error")
                        st.error(str(e))
                        continue

                    st.write(f"Extracted text from {len(pages)} page(s). Chunking...")
                    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.write(f"Created {len(chunks)} chunk(s). Generating embeddings...")

                    texts = [c.text for c in chunks]
                    vectors = embedder.embed_texts(texts)

                    namespace = safe_namespace(f.name)
                    st.write(f"Upserting vectors into Pinecone namespace '{namespace}'...")
                    try:
                        vector_store.upsert_chunks(chunks, vectors, namespace=namespace)
                    except Exception as e:
                        status.update(label=f"Failed: {f.name}", state="error")
                        st.error(f"Pinecone upsert failed: {e}")
                        continue

                    st.session_state.indexed_docs[f.name] = namespace
                    status.update(label=f"Indexed: {f.name} ✅", state="complete")

            st.success("Done. Switch to the 'Ask Questions' tab.")

    if st.session_state.indexed_docs:
        st.divider()
        st.write("**Currently indexed:**")
        for name, ns in st.session_state.indexed_docs.items():
            st.write(f"- `{name}`  →  namespace: `{ns}`")

# ---------------- Tab 2: Ask ----------------
with tab_ask:
    st.subheader("Ask a question about your document(s)")

    if not st.session_state.indexed_docs:
        st.info("Upload and index a document first.")
    else:
        question = st.text_input("Your question", placeholder="e.g. What is the termination clause?")
        ask_clicked = st.button("🔍 Get Answer", type="primary")

        if ask_clicked:
            if not question or not question.strip():
                st.warning("Please enter a non-empty question.")
            elif not selected_docs:
                st.warning("Select at least one document in the sidebar to search.")
            else:
                embedder = get_embedder(settings.embedding_model)
                vector_store = get_vector_store()
                retriever = Retriever(embedder, vector_store)
                generator = get_generator()

                all_chunks = []
                with st.spinner("Retrieving relevant context..."):
                    for doc_name in selected_docs:
                        namespace = st.session_state.indexed_docs[doc_name]
                        try:
                            results = retriever.retrieve(
                                query=question,
                                namespace=namespace,
                                top_k=top_k,
                                similarity_threshold=similarity_threshold,
                                page_filter=page_filter_value,
                            )
                            all_chunks.extend(results)
                        except ValueError as e:
                            st.warning(str(e))

                all_chunks.sort(key=lambda r: r.score, reverse=True)
                top_chunks = all_chunks[:top_k]

                with st.spinner("Generating answer..."):
                    result = generator.generate(question, top_chunks)

                st.markdown("### Answer")
                if result.grounded:
                    st.success(result.answer)
                    st.caption(f"Confidence score: **{result.confidence:.2f}** (avg. similarity of top sources)")
                else:
                    st.warning(result.answer if result.answer else NOT_FOUND_MESSAGE)

                if result.sources:
                    st.markdown("### 📎 Source References")
                    for i, src in enumerate(result.sources, start=1):
                        with st.expander(
                            f"Excerpt {i} — {src.document_name}, page {src.page_number} "
                            f"(similarity: {src.score:.2f})"
                        ):
                            st.write(truncate(src.text, 800))

                log_query(
                    namespace=",".join(selected_docs),
                    question=question,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    grounded=result.grounded,
                    confidence=result.confidence,
                    num_sources=len(result.sources),
                )
                st.session_state.query_history.append(
                    {
                        "question": question,
                        "answer": result.answer,
                        "grounded": result.grounded,
                        "confidence": result.confidence,
                        "num_sources": len(result.sources),
                    }
                )

# ---------------- Tab 3: History ----------------
with tab_history:
    st.subheader("Session query history")
    if not st.session_state.query_history:
        st.caption("No queries yet this session.")
    else:
        for i, item in enumerate(reversed(st.session_state.query_history), start=1):
            icon = "✅" if item["grounded"] else "⚠️"
            with st.expander(f"{icon} Q: {truncate(item['question'], 80)}"):
                st.write(f"**Answer:** {item['answer']}")
                st.write(f"**Confidence:** {item['confidence']:.2f}")
                st.write(f"**Sources used:** {item['num_sources']}")
