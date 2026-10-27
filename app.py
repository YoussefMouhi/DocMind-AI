from pathlib import Path
import hashlib

import streamlit as st
import pandas as pd

from src.ingestion.document_loader import load_document
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import DocumentEmbedder
from src.retrieval.vector_store import VectorStore
from src.storage.persistence import (
    save_session,
    load_session,
    clear_session,
)
from src.generation.rag_engine import RAGEngine


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocMind AI | Document Intelligence",
    page_icon="assets/docmind_icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: #0B1020;
        color: #E8EDF7;
    }

    [data-testid="stSidebar"] {
        background: #111A30;
        border-right: 1px solid #263452;
    }

    [data-testid="stSidebar"] * {
        color: #E8EDF7;
    }

    h1, h2, h3 {
        color: #F4F7FF !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(
            115deg,
            #17264A,
            #10243B,
            #251A4B
        );
        border: 1px solid #31466B;
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        color: white;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #AABAD5;
        font-size: 15px;
    }

    [data-testid="stMetric"] {
        background: #151F36;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #2B3B5B;
    }

    [data-testid="stMetricLabel"] {
        color: #AABAD5;
    }

    [data-testid="stMetricValue"] {
        color: #70D9FF;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg,
            #276EF1,
            #7357E8
        );
        border: none;
        border-radius: 10px;
    }

    [data-testid="stChatMessage"] {
        background: #151F36;
        border: 1px solid #2B3B5B;
        border-radius: 12px;
    }

    .info-card {
        background: #151F36;
        border: 1px solid #2B3B5B;
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 14px;
    }

    .info-card h3 {
        margin-top: 0;
        color: #70D9FF !important;
    }

    .info-card p {
        color: #C1CCE0;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

@st.cache_resource
def load_embedder():
    return DocumentEmbedder()


embedder = load_embedder()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "vector_store": None,
    "chunks": [],
    "documents": [],
    "indexed_files": [],
    "messages": [],
    "model_name": "qwen2.5:3b",
    "top_k": 4,
    "chunk_size": 800,
    "overlap": 150,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Restore saved index once per Streamlit session.
if "storage_restored" not in st.session_state:
    st.session_state.storage_restored = True

    try:
        saved = load_session()

        if saved is not None:
            for key, value in saved.items():
                st.session_state[key] = value

    except (OSError, ValueError, KeyError) as error:
        st.warning(f"Could not restore saved documents: {error}")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.image(
        "assets/docmind_logo_transparent.png",
        width="stretch",
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "AI Workspace",
            "Document Explorer",
            "Semantic Search",
            "Model Evaluation",

        ],
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("Clear Session", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.chunks = []
        st.session_state.documents = []
        st.session_state.indexed_files = []
        st.session_state.messages = []
        clear_session()
        st.rerun()


# --------------------------------------------------
# HEADER
# --------------------------------------------------




# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Intelligence Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Indexed Documents",
        len(st.session_state.indexed_files),
    )

    col2.metric(
        "Text Chunks",
        len(st.session_state.chunks),
    )

    col3.metric(
        "Embedding Dimensions",
        384,
    )

    col4.metric(
        "Active LLM",
        st.session_state.model_name,
    )

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("System Architecture")

        st.markdown("""
        <div class="info-card">
            <h3>Document Intelligence Pipeline</h3>
            <p>
            Document Upload → Text Extraction →
            Chunking → Embeddings → FAISS →
            Semantic Retrieval → LLM → Answer
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>Retrieval-Augmented Generation</h3>
            <p>
            DocMind AI combines semantic document retrieval
            with a local language model to generate answers
            grounded in uploaded documents.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.subheader("Technology Stack")

        stack = pd.DataFrame({
            "Component": [
                "Language",
                "Embeddings",
                "Vector Database",
                "LLM",
                "Interface",
            ],
            "Technology": [
                "Python",
                "Sentence Transformers",
                "FAISS",
                st.session_state.model_name,
                "Streamlit",
            ],
        })

        st.dataframe(
            stack,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# AI WORKSPACE
# --------------------------------------------------

elif page == "AI Workspace":

    st.subheader("AI Document Workspace")

    st.caption(
        "Upload documents and ask questions using "
        "retrieval-augmented generation."
    )

    with st.expander("📁 Document Upload", expanded=True):

        uploaded_files = st.file_uploader(
            "Select documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if st.button(
            "Process Documents",
            type="primary",
            use_container_width=True,
        ):
            if not uploaded_files:
                st.warning("Upload at least one document.")

            else:
                documents = []
                upload_dir = Path("data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)

                with st.spinner("Processing documents..."):

                    for uploaded_file in uploaded_files:
                        content = uploaded_file.getvalue()

                        digest = hashlib.sha256(
                            content
                        ).hexdigest()[:12]

                        safe_name = Path(
                            uploaded_file.name
                        ).name

                        file_path = (
                            upload_dir /
                            f"{digest}_{safe_name}"
                        )

                        file_path.write_bytes(content)

                        extracted = load_document(file_path)

                        for item in extracted:
                            item["source"] = safe_name

                        documents.extend(extracted)

                    chunks = chunk_documents(
                        documents,
                        chunk_size=st.session_state.chunk_size,
                        overlap=st.session_state.overlap,
                    )

                    if chunks:
                        embeddings = embedder.encode(
                            [chunk["text"] for chunk in chunks]
                        )

                        store = VectorStore(
                            embeddings.shape[1]
                        )

                        store.add_documents(
                            chunks,
                            embeddings,
                        )

                        st.session_state.vector_store = store
                        st.session_state.documents = documents
                        st.session_state.chunks = chunks
                        st.session_state.indexed_files = [
                            file.name for file in uploaded_files
                        ]
                        st.session_state.messages = []

                        save_session(
                            store,
                            documents,
                            chunks,
                            st.session_state.indexed_files,
                        )

                        st.success(
                            f"Successfully indexed "
                            f"{len(chunks)} chunks."
                        )
                    else:
                        st.error(
                            "No readable text was extracted."
                        )

    st.divider()

    st.subheader("AI Assistant")

    if st.session_state.vector_store is None:
        st.info(
            "Upload and process documents to activate "
            "the AI assistant."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"[{source['citation']}] "
                            f"{source['source']}"
                        )

    question = st.chat_input(
        "Ask DocMind AI...",
        disabled=st.session_state.vector_store is None,
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                query_embedding = embedder.encode(
                    [question]
                )

                results = (
                    st.session_state.vector_store.search(
                        query_embedding,
                        top_k=st.session_state.top_k,
                    )
                )

                rag = RAGEngine(
                    model_name=st.session_state.model_name
                )

                response = rag.generate_answer(
                    question,
                    results,
                )

            st.markdown(response["answer"])

            with st.expander("Retrieved Sources"):
                for source in response["sources"]:
                    st.write(
                        f"[{source['citation']}] "
                        f"{source['source']}"
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })


# --------------------------------------------------
# DOCUMENT EXPLORER
# --------------------------------------------------

elif page == "Document Explorer":

    st.subheader("Document Explorer")

    if not st.session_state.documents:
        st.info("No documents have been indexed.")

    else:
        selected = st.selectbox(
            "Select document",
            st.session_state.indexed_files,
        )

        for document in st.session_state.documents:

            if document["source"] == selected:

                page_number = document.get("page")

                if page_number:
                    st.caption(f"Page {page_number}")

                st.text_area(
                    "Extracted Text",
                    document["text"],
                    height=350,
                    disabled=True,
                    key=f"{selected}_{page_number}",
                )


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

elif page == "Semantic Search":

    st.subheader("Semantic Search Engine")

    if st.session_state.vector_store is None:
        st.info("Index documents before searching.")

    else:
        query = st.text_input(
            "Enter a semantic search query"
        )

        if st.button("Search", type="primary") and query:

            query_embedding = embedder.encode(
                [query]
            )

            results = (
                st.session_state.vector_store.search(
                    query_embedding,
                    top_k=st.session_state.top_k,
                )
            )

            for index, result in enumerate(
                results,
                start=1,
            ):

                st.markdown(
                    f"### Result {index}"
                )

                st.progress(
                    max(
                        0.0,
                        min(
                            1.0,
                            result["similarity_score"],
                        ),
                    )
                )

                st.caption(
                    f"Source: {result['source']} | "
                    f"Similarity: "
                    f"{result['similarity_score']:.4f}"
                )

                st.write(result["text"])

                st.divider()


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

elif page == "Model Evaluation":
    from src.dashboard.evaluation_page import render_evaluation_page
    render_evaluation_page()

elif page == "Settings":

    st.subheader("Model Configuration")

    st.session_state.model_name = st.selectbox(
        "Language Model",
        [
            "qwen2.5:3b",
            "llama3.1:8b",
        ],
        index=[
            "qwen2.5:3b",
            "llama3.1:8b",
        ].index(st.session_state.model_name),
    )

    st.session_state.top_k = st.slider(
        "Retrieved Chunks (Top K)",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
    )

    st.session_state.chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1500,
        value=st.session_state.chunk_size,
        step=100,
    )

    st.session_state.overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=st.session_state.overlap,
        step=25,
    )

    st.warning(
        "Changing chunk size or overlap requires "
        "reprocessing your documents."
    )

    st.divider()

    st.subheader("Active Models")

    st.write("**Embedding:** all-MiniLM-L6-v2")
    st.write(f"**LLM:** {st.session_state.model_name}")
    st.write("**Vector Store:** FAISS")
    st.write("**Retrieval:** Cosine Similarity")


# --------------------------------------------------
# COPYRIGHT FOOTER
# --------------------------------------------------

st.markdown("""
<style>
.docmind-footer {
    margin-top: 60px;
    padding: 22px 0;
    border-top: 1px solid rgba(255,255,255,0.12);
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
}
</style>

<div class="docmind-footer">
    © 2026 DocMind AI · Developed by Youssef MOUHI · All rights reserved.
</div>
""", unsafe_allow_html=True)
