from src.ingestion.document_loader import load_document
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import DocumentEmbedder
from src.retrieval.vector_store import VectorStore
from src.generation.rag_engine import RAGEngine


print("Loading documents...")
documents = load_document("data/uploads/sample.txt")

print("Creating chunks...")
chunks = chunk_documents(documents, chunk_size=250, overlap=50)

print("Generating embeddings...")
embedder = DocumentEmbedder()
embeddings = embedder.encode([chunk["text"] for chunk in chunks])

print("Building FAISS index...")
store = VectorStore(embeddings.shape[1])
store.add_documents(chunks, embeddings)

print("Initializing local LLM...")
rag = RAGEngine()

questions = [
    "What is DocMind AI?",
    "How does DocMind AI retrieve relevant information?",
    "Who founded DocMind AI?",
]

for question in questions:
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")

    query_embedding = embedder.encode([question])
    results = store.search(query_embedding, top_k=3)

    response = rag.generate_answer(question, results)

    print(f"\nANSWER:\n{response['answer']}")
    print("\nRETRIEVED SOURCES:")

    for source in response["sources"]:
        score = source.get("similarity_score")
        score_text = f"{score:.4f}" if score is not None else "N/A"

        print(
            f"[{source['citation']}] "
            f"{source['source']} "
            f"(similarity: {score_text})"
        )

print("\nDocMind AI RAG test completed!")
