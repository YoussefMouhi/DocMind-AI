import numpy as np

from src.ingestion.document_loader import load_document
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import DocumentEmbedder
from src.retrieval.vector_store import VectorStore


def test_complete_rag_retrieval_pipeline(tmp_path):
    """
    Integration test:
    TXT -> Extraction -> Chunking -> Embeddings -> FAISS -> Retrieval.
    """

    # 1. Create an isolated test document.
    document_path = tmp_path / "test_document.txt"

    document_path.write_text(
        "Retrieval-Augmented Generation combines information retrieval "
        "with language model generation. "
        "FAISS enables efficient similarity search over vector embeddings.",
        encoding="utf-8",
    )

    # 2. Extract document content.
    documents = load_document(str(document_path))

    assert documents, "Document extraction returned no content."

    # 3. Split extracted text into chunks.
    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=30,
    )

    assert len(chunks) > 0, "Chunking produced no chunks."

    # 4. Generate embeddings.
    embedder = DocumentEmbedder()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = np.asarray(embedder.encode(texts), dtype=np.float32)

    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(chunks)

    # 5. Create an isolated FAISS vector store.
    store = VectorStore(dimension=embeddings.shape[1])
    store.add_documents(chunks, embeddings)

    assert store.index.ntotal == len(chunks)

    # 6. Retrieve relevant document chunks.
    question = "What does FAISS do?"

    query_embedding = np.asarray(
        embedder.encode([question]),
        dtype=np.float32,
    )

    results = store.search(query_embedding[0], top_k=1)

    assert len(results) == 1
    assert "FAISS" in results[0]["text"]

    # 7. Verify index persistence in a temporary directory.
    index_directory = tmp_path / "faiss_test_index"
    store.save(index_directory)

    assert (index_directory / "index.faiss").exists()
    assert (index_directory / "metadata.json").exists()

    # 8. Reload the saved FAISS index.
    restored_store = VectorStore.load(index_directory)

    assert restored_store.index.ntotal == len(chunks)

    # 9. Search the restored index.
    restored_results = restored_store.search(
        query_embedding[0],
        top_k=1,
    )

    assert len(restored_results) == 1
    assert "FAISS" in restored_results[0]["text"]
