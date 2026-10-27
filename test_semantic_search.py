from src.ingestion.document_loader import load_document
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import DocumentEmbedder
from src.retrieval.vector_store import VectorStore


print("\nLoading document...")
documents = load_document("data/uploads/sample.txt")

print("Creating text chunks...")
chunks = chunk_documents(documents, chunk_size=150, overlap=30)

print("Loading embedding model...")
embedder = DocumentEmbedder()

print("Generating embeddings...")
embeddings = embedder.encode([chunk["text"] for chunk in chunks])

print("Creating FAISS index...")
store = VectorStore(embeddings.shape[1])
store.add_documents(chunks, embeddings)

query = "How does DocMind AI find relevant information?"

print(f"\nQuestion: {query}")

query_embedding = embedder.encode([query])
results = store.search(query_embedding, top_k=3)

for rank, result in enumerate(results, start=1):
    print(f"\n{'=' * 60}")
    print(f"RESULT {rank}")
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Text:\n{result['text']}")

print("\nSemantic search completed successfully!")
